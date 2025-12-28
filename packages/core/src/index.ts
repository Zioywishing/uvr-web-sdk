
export class UAR {
  options: {
    modelUrl: string;
    workerUrl: string;
    provider?: 'wasm' | 'webgpu';
    workerCount?: number;
  };

  private status: 'UNINITIALIZED' | 'FREE' | 'BUSY' = 'UNINITIALIZED';
  private workers: Worker[] = [];
  private initPromise: Promise<void> | null = null;
  private audioCtx: AudioContext | null = null;

  public constructor(options: {
    modelUrl: string;
    workerUrl: string;
    provider?: 'wasm' | 'webgpu';
    workerCount?: number;
  }) {
    this.options = {
      ...options,
      workerCount: options.workerCount || 1
    };
  }

  public async init() {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = (async () => {
      if (this.workers.length > 0) {
        return;
      }

      const count = this.options.workerCount || 1;
      console.log(`[UAR] 正在创建 ${count} 个 Worker...`, this.options.workerUrl);

      const promises: Promise<Worker>[] = [];
      for (let i = 0; i < count; i++) {
        promises.push(this.createWorker(i));
      }

      try {
        this.workers = await Promise.all(promises);
      } catch (e) {
        this.workers.forEach(w => w.terminate());
        this.workers = [];
        this.initPromise = null;
        throw e;
      }

      // 预创建并启动 AudioContext 以热身（减少首次解码延迟）
      if (!this.audioCtx) {
        console.log('[UAR] 正在预创建 AudioContext (44100Hz)...');
        this.audioCtx = new AudioContext({ sampleRate: 44100 });
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume().catch(e => console.warn('[UAR] AudioContext resume failed during init:', e));
        }
      }

      this.status = 'FREE';
    })();

    return this.initPromise;
  }

  private async createWorker(index: number): Promise<Worker> {
    const worker = new Worker(this.options.workerUrl, { type: 'module' });

    // 1. 等待 Worker 就绪
    await new Promise<void>((resolve, reject) => {
      const initHandler = (e: MessageEvent) => {
        const { type, error } = e.data;
        if (type === 'worker_ready') {
          console.log(`[UAR] Worker ${index} 就绪`);
          worker.removeEventListener('message', initHandler);
          resolve();
        } else if (type === 'error') {
          worker.removeEventListener('message', initHandler);
          reject(new Error(error || `Worker ${index} init error`));
        }
      };
      worker.addEventListener('message', initHandler);
    });

    // 2. 预加载模型
    console.log(`[UAR] Worker ${index} 正在预加载模型...`, this.options.modelUrl);
    await new Promise<void>((resolve, reject) => {
      const preloadHandler = (e: MessageEvent) => {
        const { type, error } = e.data;
        if (type === 'preloaded') {
          worker.removeEventListener('message', preloadHandler);
          resolve();
        } else if (type === 'error') {
          worker.removeEventListener('message', preloadHandler);
          reject(new Error(error || `Worker ${index} preload error`));
        }
      };
      worker.addEventListener('message', preloadHandler);

      worker.postMessage({
        type: 'preload',
        data: {
          modelUrl: this.options.modelUrl,
          provider: this.options.provider
        }
      });
    });

    return worker;
  }

  public destroy() {
    console.log('[UAR] Destroying...');
    this.workers.forEach(w => w.terminate());
    this.workers = [];
    if (this.audioCtx) {
      this.audioCtx.close().catch(console.error);
      this.audioCtx = null;
    }
    this.status = 'UNINITIALIZED';
    this.initPromise = null;
  }

  public process(AudioData: ArrayBuffer): ReadableStream<Float32Array> {
    if (this.status === 'BUSY') {
      throw new Error('UAR is busy');
    }
    console.log('[UAR] 开始处理音频数据, 大小:', AudioData.byteLength);
    this.status = 'BUSY';

    let controller: ReadableStreamDefaultController<Float32Array>;

    return new ReadableStream<Float32Array>({
      start: async (c) => {
        controller = c;
        try {
          if (this.workers.length === 0) {
            await this.init();
          }

          if (!this.audioCtx) {
            this.audioCtx = new AudioContext({ sampleRate: 44100 });
          }
          const audioCtx = this.audioCtx;
          
          if (audioCtx.state === 'suspended') {
            await audioCtx.resume();
          }

          console.log('[UAR] 解码音频数据...');
          const audioBuffer = await audioCtx.decodeAudioData(AudioData.slice(0));
          console.log('[UAR] 音频解码完成, 时长:', audioBuffer.duration, '采样率:', audioBuffer.sampleRate);

          const chL = audioBuffer.getChannelData(0);
          const chR = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : chL;

          // 对齐采样率：如果不是 44100，进行重采样
          let finalChL = chL;
          let finalChR = chR;
          if (audioBuffer.sampleRate !== 44100) {
            console.log(`[UAR] 采样率不匹配 (${audioBuffer.sampleRate} -> 44100), 正在重采样...`);
            const resampled = await this.resampleChannelsTo44100(chL, chR, audioBuffer.sampleRate);
            finalChL = resampled.chL as any;
            finalChR = resampled.chR as any;
          }

          await this.runParallelStream(finalChL, finalChR, 44100, controller);

        } catch (err: unknown) {
          const error = err instanceof Error ? err : new Error(String(err));
          console.error('[UAR] 处理过程出错:', error);
          controller.error(error);
          this.status = 'FREE';
        }
      }
    });
  }

  /**
   * 直接处理左右声道数据（Float32Array）
   * 适用于已经解码的音频或实时音频流
   */
  public processChannels(chL: Float32Array, chR: Float32Array, sampleRate: number): ReadableStream<Float32Array> {
    if (this.status === 'BUSY') {
      throw new Error('UAR is busy');
    }
    console.log('[UAR] 开始处理声道数据, 长度:', chL.length, '采样率:', sampleRate);
    this.status = 'BUSY';

    let controller: ReadableStreamDefaultController<Float32Array>;

    return new ReadableStream<Float32Array>({
      start: async (c) => {
        controller = c;
        try {
          if (this.workers.length === 0) {
            await this.init();
          }

          let finalChL = chL;
          let finalChR = chR;
          if (sampleRate !== 44100) {
            console.log(`[UAR] 输入采样率不匹配 (${sampleRate} -> 44100), 正在重采样...`);
            const resampled = await this.resampleChannelsTo44100(chL, chR, sampleRate);
            finalChL = resampled.chL;
            finalChR = resampled.chR;
          }

          await this.runParallelStream(finalChL, finalChR, 44100, controller);

        } catch (err: unknown) {
          const error = err instanceof Error ? err : new Error(String(err));
          console.error('[UAR] 处理过程出错:', error);
          controller.error(error);
          this.status = 'FREE';
        }
      }
    });
  }

  private async runParallelStream(
    chL: Float32Array,
    chR: Float32Array,
    sampleRate: number,
    controller: ReadableStreamDefaultController<Float32Array>
  ) {
    const workerCount = this.workers.length;
    console.log(`[UAR] 使用 ${workerCount} 个 Worker 处理`);

    if (workerCount <= 1) {
      await this.runSingleWorkerStream(chL, chR, sampleRate, controller);
      return;
    }

    await this.runMultiWorkerSegmentedStream(chL, chR, sampleRate, controller);
  }

  private async runSingleWorkerStream(
    chL: Float32Array,
    chR: Float32Array,
    sampleRate: number,
    controller: ReadableStreamDefaultController<Float32Array>
  ) {
    const worker = this.workers[0];

    await new Promise<void>((resolve, reject) => {
      const startHandler = (e: MessageEvent) => {
        const payload = e.data as unknown;
        const type = this.getMessageType(payload);
        if (type === 'stream_started') {
          worker.removeEventListener('message', startHandler);
          resolve();
          return;
        }
        if (type === 'error') {
          worker.removeEventListener('message', startHandler);
          reject(new Error(this.getMessageError(payload) || 'Worker stream_start error'));
        }
      };
      worker.addEventListener('message', startHandler);
      worker.postMessage({
        type: 'stream_start',
        data: {
          modelUrl: this.options.modelUrl,
          provider: this.options.provider,
          sampleRate
        }
      });
    });

    const finished = new Promise<void>((resolve, reject) => {
      const streamHandler = (e: MessageEvent) => {
        const payload = e.data as unknown;
        const type = this.getMessageType(payload);
        if (type === 'stream_result') {
          const data = this.getStreamResultData(payload);
          if (!data) return;

          const len = data.chL.length;
          const interleaved = new Float32Array(len * 2);
          for (let i = 0; i < len; i++) {
            interleaved[i * 2] = data.chL[i];
            interleaved[i * 2 + 1] = data.chR[i];
          }
          controller.enqueue(interleaved);
          return;
        }
        if (type === 'stream_ended') {
          worker.removeEventListener('message', streamHandler);
          resolve();
          return;
        }
        if (type === 'error') {
          worker.removeEventListener('message', streamHandler);
          reject(new Error(this.getMessageError(payload) || 'Worker error'));
        }
      };
      worker.addEventListener('message', streamHandler);
    });

    const postChunkSize = 16384;
    for (let offset = 0; offset < chL.length; offset += postChunkSize) {
      const end = Math.min(offset + postChunkSize, chL.length);
      const subL = chL.slice(offset, end);
      const subR = chR.slice(offset, end);
      worker.postMessage({ type: 'stream_data', data: { chL: subL, chR: subR } }, [subL.buffer, subR.buffer]);
    }

    worker.postMessage({ type: 'stream_end' });

    try {
      await finished;
      controller.close();
      this.status = 'FREE';
    } catch (err: unknown) {
      controller.error(err instanceof Error ? err : new Error(String(err)));
      this.status = 'FREE';
    }
  }

  private async runMultiWorkerSegmentedStream(
    chL: Float32Array,
    chR: Float32Array,
    sampleRate: number,
    controller: ReadableStreamDefaultController<Float32Array>
  ) {
    const workerCount = this.workers.length;
    let streamParams: { nfft: number; chunkSize: number; segStep: number } | null = null;

    const startPromises = this.workers.map((worker, index) => {
      return new Promise<void>((resolve, reject) => {
        const startHandler = (e: MessageEvent) => {
          const payload = e.data as unknown;
          const type = this.getMessageType(payload);
          if (type === 'stream_started') {
            const params = this.getStreamStartedData(payload);
            if (!streamParams && params) {
              streamParams = params;
            }
            worker.removeEventListener('message', startHandler);
            resolve();
            return;
          }
          if (type === 'error') {
            worker.removeEventListener('message', startHandler);
            reject(new Error(this.getMessageError(payload) || `Worker ${index} stream_start error`));
          }
        };
        worker.addEventListener('message', startHandler);
        worker.postMessage({
          type: 'stream_start',
          data: {
            modelUrl: this.options.modelUrl,
            provider: this.options.provider,
            sampleRate
          }
        });
      });
    });

    await Promise.all(startPromises);

    const fallbackNfft = 6144;
    const fallbackChunkSize = 1024 * 255;
    const used = streamParams ?? {
      nfft: fallbackNfft,
      chunkSize: fallbackChunkSize,
      segStep: fallbackChunkSize - fallbackNfft
    };

    const segStep = used.segStep;
    const overlap = used.chunkSize;

    const segments: Array<{ segmentIndex: number; segStart: number; segEnd: number; sendStart: number; sendEnd: number }> = [];
    const totalLen = chL.length;
    const totalSteps = Math.ceil(totalLen / segStep);
    const stepsPerWorker = Math.ceil(totalSteps / workerCount);

    for (let i = 0; i < workerCount; i++) {
      const segStart = Math.min(totalLen, i * stepsPerWorker * segStep);
      const segEnd = Math.min(totalLen, (i + 1) * stepsPerWorker * segStep);
      const hasData = segEnd > segStart;
      const sendStart = hasData ? Math.max(0, segStart - overlap) : segStart;
      const sendEnd = hasData ? Math.min(totalLen, segEnd + overlap) : segStart;
      segments.push({ segmentIndex: i, segStart, segEnd, sendStart, sendEnd });
    }

    let nextSegmentToOutput = 0;
    const segmentBuffers = new Map<number, Float32Array[]>();
    const segmentEnded = new Set<number>();
    for (const segment of segments) {
      segmentBuffers.set(segment.segmentIndex, []);
    }

    const flush = () => {
      while (segmentBuffers.has(nextSegmentToOutput)) {
        const buffer = segmentBuffers.get(nextSegmentToOutput);
        if (buffer && buffer.length > 0) {
          for (const chunk of buffer) controller.enqueue(chunk);
          buffer.length = 0;
        }

        if (!segmentEnded.has(nextSegmentToOutput)) return;

        segmentEnded.delete(nextSegmentToOutput);
        segmentBuffers.delete(nextSegmentToOutput);
        nextSegmentToOutput++;
      }
    };

    const resultPromises = this.workers.map((worker, workerIndex) => {
      const segment = segments[workerIndex];
      const trimStartFrames = segment.segStart - segment.sendStart;
      const keepFrames = segment.segEnd - segment.segStart;
      let skipFrames = trimStartFrames;
      let remainingFrames = keepFrames;

      return new Promise<void>((resolve, reject) => {
        const streamHandler = (e: MessageEvent) => {
          const payload = e.data as unknown;
          const type = this.getMessageType(payload);

          if (type === 'stream_result') {
            if (remainingFrames <= 0) return;
            const data = this.getStreamResultData(payload);
            if (!data) return;

            const len = data.chL.length;
            if (data.chR.length !== len) {
              reject(new Error(`Worker ${workerIndex} 返回的左右声道长度不一致`));
              worker.removeEventListener('message', streamHandler);
              return;
            }

            const interleaved = new Float32Array(len * 2);
            for (let i = 0; i < len; i++) {
              interleaved[i * 2] = data.chL[i];
              interleaved[i * 2 + 1] = data.chR[i];
            }

            const chunkFrames = len;
            if (skipFrames >= chunkFrames) {
              skipFrames -= chunkFrames;
              return;
            }

            const startFrame = skipFrames;
            skipFrames = 0;
            const availableFrames = chunkFrames - startFrame;
            const takeFrames = Math.min(availableFrames, remainingFrames);
            remainingFrames -= takeFrames;

            const start = startFrame * 2;
            const end = start + takeFrames * 2;

            const trimmed = new Float32Array(takeFrames * 2);
            trimmed.set(interleaved.subarray(start, end));
            if (segment.segmentIndex === nextSegmentToOutput) {
              controller.enqueue(trimmed);
            } else {
              const buffer = segmentBuffers.get(segment.segmentIndex);
              if (buffer) buffer.push(trimmed);
            }
            return;
          }

          if (type === 'stream_ended') {
            worker.removeEventListener('message', streamHandler);
            segmentEnded.add(segment.segmentIndex);
            flush();
            resolve();
            return;
          }

          if (type === 'error') {
            worker.removeEventListener('message', streamHandler);
            reject(new Error(this.getMessageError(payload) || `Worker ${workerIndex} 运行时错误`));
          }
        };

        worker.addEventListener('message', streamHandler);
      });
    });

    const postChunkSize = 16384;

    for (let workerIndex = 0; workerIndex < workerCount; workerIndex++) {
      const segment = segments[workerIndex];
      const worker = this.workers[workerIndex];
      for (let offset = segment.sendStart; offset < segment.sendEnd; offset += postChunkSize) {
        const end = Math.min(offset + postChunkSize, segment.sendEnd);
        const subL = chL.slice(offset, end);
        const subR = chR.slice(offset, end);
        worker.postMessage({ type: 'stream_data', data: { chL: subL, chR: subR } }, [subL.buffer, subR.buffer]);
      }
    }

    this.workers.forEach(w => w.postMessage({ type: 'stream_end' }));

    try {
      await Promise.all(resultPromises);
      controller.close();
      this.status = 'FREE';
    } catch (err: unknown) {
      controller.error(err instanceof Error ? err : new Error(String(err)));
      this.status = 'FREE';
    }
  }

  private getMessageType(payload: unknown): string | null {
    if (typeof payload !== 'object' || payload === null) return null;
    if (!('type' in payload)) return null;
    const typeValue = (payload as { type?: unknown }).type;
    return typeof typeValue === 'string' ? typeValue : null;
  }

  private getMessageError(payload: unknown): string | null {
    if (typeof payload !== 'object' || payload === null) return null;
    if (!('error' in payload)) return null;
    const errorValue = (payload as { error?: unknown }).error;
    return typeof errorValue === 'string' ? errorValue : null;
  }

  private getStreamStartedData(payload: unknown): { nfft: number; chunkSize: number; segStep: number } | null {
    if (typeof payload !== 'object' || payload === null) return null;
    if (!('data' in payload)) return null;
    const dataValue = (payload as { data?: unknown }).data;
    if (typeof dataValue !== 'object' || dataValue === null) return null;

    const nfft = (dataValue as { nfft?: unknown }).nfft;
    const chunkSize = (dataValue as { chunkSize?: unknown }).chunkSize;
    const segStep = (dataValue as { segStep?: unknown }).segStep;

    if (typeof nfft !== 'number' || !Number.isFinite(nfft)) return null;
    if (typeof chunkSize !== 'number' || !Number.isFinite(chunkSize)) return null;
    if (typeof segStep !== 'number' || !Number.isFinite(segStep)) return null;
    if (nfft <= 0 || chunkSize <= 0 || segStep <= 0) return null;

    return { nfft, chunkSize, segStep };
  }

  private getStreamResultData(payload: unknown): { chL: Float32Array; chR: Float32Array } | null {
    if (typeof payload !== 'object' || payload === null) return null;
    if (!('data' in payload)) return null;
    const dataValue = (payload as { data?: unknown }).data;
    if (typeof dataValue !== 'object' || dataValue === null) return null;
    const chL = (dataValue as { chL?: unknown }).chL;
    const chR = (dataValue as { chR?: unknown }).chR;
    if (!(chL instanceof Float32Array) || !(chR instanceof Float32Array)) return null;
    return { chL, chR };
  }

  /**
   * 将双声道数据重采样到 44100Hz
   */
  private async resampleChannelsTo44100(
    chL: Float32Array,
    chR: Float32Array,
    sampleRate: number
  ): Promise<{ chL: Float32Array; chR: Float32Array }> {
    if (sampleRate === 44100) return { chL, chR };

    const targetFrames = Math.round((chL.length * 44100) / sampleRate);
    const offlineCtx = new OfflineAudioContext(2, targetFrames, 44100);

    const buffer = offlineCtx.createBuffer(2, chL.length, sampleRate);
    buffer.copyToChannel(chL, 0);
    buffer.copyToChannel(chR, 1);

    const source = offlineCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(offlineCtx.destination);
    source.start();

    const renderedBuffer = await offlineCtx.startRendering();
    return {
      chL: renderedBuffer.getChannelData(0),
      chR: renderedBuffer.getChannelData(1)
    };
  }
}

export default UAR;
