
type Provider = 'wasm' | 'webgpu';

type F32 = Float32Array<ArrayBufferLike>;
type DomF32 = Float32Array<ArrayBuffer>;

export type UVROptions =
  | {
      modelUrl: string;
      provider?: Provider;
      workerCount?: number;
      workerUrl: string;
    }
  | {
      modelUrl: string;
      provider?: Provider;
      workerCount?: number;
      fftWorkerUrl: string;
      ortWorkerUrl: string;
      ifftWorkerUrl: string;
      workerUrl?: string;
    };

interface StreamParams {
  dimF: number;
  dimT: number;
  nfft: number;
  hop: number;
  chunkSize: number;
  segStep: number;
}

interface MainStreamingState extends StreamParams {
  sampleRate: number;
  padSamples: number;
  headFadeSamples: number;
  fadeIn: Float32Array;
  fadeOut: Float32Array;
  inputBufferL: Float32Array;
  inputBufferR: Float32Array;
  inputBufferWritePos: number;
  outputBufferL: Float32Array;
  outputBufferR: Float32Array;
  normBuffer: Float32Array;
  processedPos: number;
  outputPos: number;
  totalReceived: number;
  outputEmitted: number;
}

function createFadeInOut(nfft: number): { fadeIn: Float32Array; fadeOut: Float32Array } {
  const fadeIn = new Float32Array(nfft);
  const fadeOut = new Float32Array(nfft);
  for (let i = 0; i < nfft; i++) {
    const w = 0.5 * (1 - Math.cos((Math.PI * i) / (nfft - 1)));
    fadeIn[i] = w;
    fadeOut[i] = 1 - w;
  }
  return { fadeIn, fadeOut };
}

function toDomFloat32Array(src: F32): DomF32 {
  if (src.buffer instanceof ArrayBuffer) {
    return src as unknown as DomF32;
  }
  return new Float32Array(src) as unknown as DomF32;
}

export class UVR {
  options: UVROptions;

  private status: 'UNINITIALIZED' | 'FREE' | 'BUSY' = 'UNINITIALIZED';
  private workers: Worker[] = [];
  private fftWorkers: Worker[] = [];
  private ifftWorkers: Worker[] = [];
  private ortWorker: Worker | null = null;

  private fftClients: FftWorkerClient[] = [];
  private ifftClients: IfftWorkerClient[] = [];
  private ortClient: OrtWorkerClient | null = null;

  private initPromise: Promise<void> | null = null;
  private audioCtx: AudioContext | null = null;

  public constructor(options: UVROptions) {
    this.options = {
      ...options,
      workerCount: options.workerCount || 1,
    };
  }

  private isPipelineMode(): boolean {
    return (
      typeof (this.options as { fftWorkerUrl?: unknown }).fftWorkerUrl === 'string' &&
      typeof (this.options as { ortWorkerUrl?: unknown }).ortWorkerUrl === 'string' &&
      typeof (this.options as { ifftWorkerUrl?: unknown }).ifftWorkerUrl === 'string'
    );
  }

  private getStreamParams(): StreamParams {
    const dimF = 3072;
    const dimT = 256;
    const nfft = 6144;
    const hop = 1024;
    const chunkSize = hop * (dimT - 1);
    const segStep = chunkSize - nfft;
    return { dimF, dimT, nfft, hop, chunkSize, segStep };
  }

  public async init() {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = (async () => {
      if (this.workers.length > 0 || this.fftWorkers.length > 0 || this.ortWorker || this.ifftWorkers.length > 0) {
        return;
      }

      const count = this.options.workerCount || 1;
      if (this.isPipelineMode()) {
        const opts = this.options as Extract<UVROptions, { fftWorkerUrl: string }>;
        console.log(`[UVR] 正在创建 ${count} 个 FFT Worker...`, opts.fftWorkerUrl);
        console.log('[UVR] 正在创建 ORT Worker...', opts.ortWorkerUrl);
        console.log(`[UVR] 正在创建 ${count} 个 IFFT Worker...`, opts.ifftWorkerUrl);

        try {
          this.fftWorkers = await Promise.all(Array.from({ length: count }, (_, i) => this.createFftWorker(opts.fftWorkerUrl, i)));
          this.ortWorker = await this.createOrtWorker(opts.ortWorkerUrl);
          this.ifftWorkers = await Promise.all(Array.from({ length: count }, (_, i) => this.createIfftWorker(opts.ifftWorkerUrl, i)));

          this.fftClients = this.fftWorkers.map((w, i) => new FftWorkerClient(w, i));
          this.ifftClients = this.ifftWorkers.map((w, i) => new IfftWorkerClient(w, i));
          this.ortClient = new OrtWorkerClient(this.ortWorker);

          await this.ortClient.preload(this.options.modelUrl, this.options.provider);

          const params = this.getStreamParams();
          await Promise.all([
            Promise.all(this.fftClients.map(c => c.init(params))),
            Promise.all(this.ifftClients.map(c => c.init(params))),
          ]);
        } catch (e) {
          this.fftWorkers.forEach(w => w.terminate());
          this.ifftWorkers.forEach(w => w.terminate());
          if (this.ortWorker) this.ortWorker.terminate();
          this.fftWorkers = [];
          this.ifftWorkers = [];
          this.ortWorker = null;
          this.fftClients = [];
          this.ifftClients = [];
          this.ortClient = null;
          this.initPromise = null;
          throw e;
        }
      } else {
        const legacyWorkerUrl = (this.options as { workerUrl?: unknown }).workerUrl;
        if (typeof legacyWorkerUrl !== 'string') {
          throw new Error('未提供 workerUrl（旧模式）或 fftWorkerUrl/ortWorkerUrl/ifftWorkerUrl（新模式）');
        }
        console.log(`[UVR] 正在创建 ${count} 个 Worker...`, legacyWorkerUrl);

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
      }

      // 预创建并启动 AudioContext 以热身（减少首次解码延迟）
      if (!this.audioCtx) {
        console.log('[UVR] 正在预创建 AudioContext (44100Hz)...');
        this.audioCtx = new AudioContext({ sampleRate: 44100 });
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume().catch(e => console.warn('[UVR] AudioContext resume failed during init:', e));
        }
      }

      this.status = 'FREE';
    })();

    return this.initPromise;
  }

  private async createWorker(index: number): Promise<Worker> {
    const workerUrl = (this.options as { workerUrl: string }).workerUrl;
    const worker = new Worker(workerUrl, { type: 'module' });

    await waitForType(worker, 'worker_ready');
    console.log(`[UVR] Worker ${index} 就绪`);
    console.log(`[UVR] Worker ${index} 正在预加载模型...`, this.options.modelUrl);
    worker.postMessage({
      type: 'preload',
      data: {
        modelUrl: this.options.modelUrl,
        provider: this.options.provider,
      },
    });
    await waitForType(worker, 'preloaded');
    return worker;
  }

  private async createFftWorker(workerUrl: string, index: number): Promise<Worker> {
    const worker = new Worker(workerUrl, { type: 'module' });
    await waitForType(worker, 'worker_ready');
    console.log(`[UVR] FFT Worker ${index} 就绪`);
    return worker;
  }

  private async createIfftWorker(workerUrl: string, index: number): Promise<Worker> {
    const worker = new Worker(workerUrl, { type: 'module' });
    await waitForType(worker, 'worker_ready');
    console.log(`[UVR] IFFT Worker ${index} 就绪`);
    return worker;
  }

  private async createOrtWorker(workerUrl: string): Promise<Worker> {
    const worker = new Worker(workerUrl, { type: 'module' });
    await waitForType(worker, 'worker_ready');
    console.log('[UVR] ORT Worker 就绪');
    return worker;
  }

  public destroy() {
    console.log('[UVR] Destroying...');
    this.workers.forEach(w => w.terminate());
    this.workers = [];
    this.fftWorkers.forEach(w => w.terminate());
    this.fftWorkers = [];
    this.ifftWorkers.forEach(w => w.terminate());
    this.ifftWorkers = [];
    if (this.ortWorker) this.ortWorker.terminate();
    this.ortWorker = null;
    this.fftClients = [];
    this.ifftClients = [];
    this.ortClient = null;
    if (this.audioCtx) {
      this.audioCtx.close().catch(console.error);
      this.audioCtx = null;
    }
    this.status = 'UNINITIALIZED';
    this.initPromise = null;
  }

  public process(AudioData: ArrayBuffer): ReadableStream<Float32Array> {
    if (this.status === 'BUSY') {
      throw new Error('UVR is busy');
    }
    console.log('[UVR] 开始处理音频数据, 大小:', AudioData.byteLength);
    this.status = 'BUSY';

    let controller: ReadableStreamDefaultController<Float32Array>;

    return new ReadableStream<Float32Array>({
      start: async (c) => {
        controller = c;
        try {
          await this.init();

          if (!this.audioCtx) {
            this.audioCtx = new AudioContext({ sampleRate: 44100 });
          }
          const audioCtx = this.audioCtx;
          
          if (audioCtx.state === 'suspended') {
            await audioCtx.resume();
          }

          console.log('[UVR] 解码音频数据...');
          const audioBuffer = await audioCtx.decodeAudioData(AudioData.slice(0));
          console.log('[UVR] 音频解码完成, 时长:', audioBuffer.duration, '采样率:', audioBuffer.sampleRate);

          const chL = audioBuffer.getChannelData(0) as unknown as F32;
          const chR = (audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : chL) as unknown as F32;

          // 对齐采样率：如果不是 44100，进行重采样
          let finalChL: F32 = chL;
          let finalChR: F32 = chR;
          if (audioBuffer.sampleRate !== 44100) {
            console.log(`[UVR] 采样率不匹配 (${audioBuffer.sampleRate} -> 44100), 正在重采样...`);
            const resampled = await this.resampleChannelsTo44100(chL, chR, audioBuffer.sampleRate);
            finalChL = resampled.chL;
            finalChR = resampled.chR;
          }

          await this.runParallelStream(finalChL, finalChR, 44100, controller);

        } catch (err: unknown) {
          const error = err instanceof Error ? err : new Error(String(err));
          console.error('[UVR] 处理过程出错:', error);
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
      throw new Error('UVR is busy');
    }
    console.log('[UVR] 开始处理声道数据, 长度:', chL.length, '采样率:', sampleRate);
    this.status = 'BUSY';

    let controller: ReadableStreamDefaultController<Float32Array>;

    return new ReadableStream<Float32Array>({
      start: async (c) => {
        controller = c;
        try {
          await this.init();

          let finalChL = chL;
          let finalChR = chR;
          if (sampleRate !== 44100) {
            console.log(`[UVR] 输入采样率不匹配 (${sampleRate} -> 44100), 正在重采样...`);
            const resampled = await this.resampleChannelsTo44100(chL, chR, sampleRate);
            finalChL = resampled.chL;
            finalChR = resampled.chR;
          }

          await this.runParallelStream(finalChL, finalChR, 44100, controller);

        } catch (err: unknown) {
          const error = err instanceof Error ? err : new Error(String(err));
          console.error('[UVR] 处理过程出错:', error);
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
    if (this.isPipelineMode()) {
      const fftCount = this.fftWorkers.length;
      console.log(`[UVR] 使用流水线模式（FFT Worker: ${fftCount}, ORT Worker: 1, IFFT Worker: ${this.ifftWorkers.length}）`);
      await this.runPipelineStream(chL, chR, sampleRate, controller);
      return;
    }

    const workerCount = this.workers.length;
    console.log(`[UVR] 使用 ${workerCount} 个 Worker 处理`);

    if (workerCount <= 1) {
      await this.runSingleWorkerStream(chL, chR, sampleRate, controller);
      return;
    }

    await this.runMultiWorkerSegmentedStream(chL, chR, sampleRate, controller);
  }

  private initMainStreamingState(sampleRate: number, totalFrames: number): MainStreamingState {
    const params = this.getStreamParams();
    const sr = sampleRate || 44100;

    const padSamples = Math.max(0, Math.floor(0.4 * sr));
    const headFadeSamples = Math.min(params.chunkSize, Math.max(1, Math.floor(0.015 * sr)));
    const { fadeIn, fadeOut } = createFadeInOut(params.nfft);

    const tailPad = params.chunkSize + params.nfft;
    const bufferSize = padSamples + totalFrames + tailPad;
    const padding = padSamples;

    return {
      ...params,
      sampleRate: sr,
      padSamples,
      headFadeSamples,
      fadeIn,
      fadeOut,
      inputBufferL: new Float32Array(bufferSize),
      inputBufferR: new Float32Array(bufferSize),
      inputBufferWritePos: padding,
      outputBufferL: new Float32Array(bufferSize),
      outputBufferR: new Float32Array(bufferSize),
      normBuffer: new Float32Array(bufferSize),
      processedPos: 0,
      outputPos: padding,
      totalReceived: 0,
      outputEmitted: 0,
    };
  }

  private emitStreamResult(state: MainStreamingState, length: number, controller: ReadableStreamDefaultController<Float32Array>) {
    const remainingToEmit = state.totalReceived - state.outputEmitted;
    const actualLen = Math.min(length, remainingToEmit);
    if (actualLen <= 0) return;

    const outL = new Float32Array(actualLen);
    const outR = new Float32Array(actualLen);

    const startIdx = state.outputPos;
    const endIdx = startIdx + actualLen;
    if (endIdx > state.outputBufferL.length) {
      throw new Error('输出缓冲区越界');
    }
    for (let i = 0; i < actualLen; i++) {
      const bufIdx = startIdx + i;
      const d = state.normBuffer[bufIdx] || 1;
      outL[i] = state.outputBufferL[bufIdx] / d;
      outR[i] = state.outputBufferR[bufIdx] / d;
    }

    state.outputBufferL.fill(0, startIdx, endIdx);
    state.outputBufferR.fill(0, startIdx, endIdx);
    state.normBuffer.fill(0, startIdx, endIdx);

    state.outputPos += length;
    state.outputEmitted += actualLen;

    const interleaved = new Float32Array(actualLen * 2);
    for (let i = 0; i < actualLen; i++) {
      interleaved[i * 2] = outL[i];
      interleaved[i * 2 + 1] = outR[i];
    }
    controller.enqueue(interleaved);
  }

  private applySegmentToLinear(
    state: MainStreamingState,
    pos: number,
    segOutL: Float32Array,
    segOutR: Float32Array,
    isLast: boolean,
  ) {
    const { nfft, fadeIn, fadeOut, headFadeSamples, segStep } = state;
    const writeMax = segOutL.length;
    const isFirst = pos === 0;

    const actualHeadFade = isFirst ? headFadeSamples : nfft;
    for (let i = 0; i < actualHeadFade; i++) {
      const bufIdx = pos + i;
      const w = fadeIn[Math.floor((i * (nfft - 1)) / actualHeadFade)];
      state.outputBufferL[bufIdx] += segOutL[i] * w;
      state.outputBufferR[bufIdx] += segOutR[i] * w;
      state.normBuffer[bufIdx] += w;
    }
    for (let i = actualHeadFade; i < Math.min(segStep, writeMax); i++) {
      const bufIdx = pos + i;
      state.outputBufferL[bufIdx] += segOutL[i];
      state.outputBufferR[bufIdx] += segOutR[i];
      state.normBuffer[bufIdx] += 1;
    }

    const tailLen = Math.min(nfft, Math.max(0, writeMax - segStep));
    for (let j = 0; j < tailLen; j++) {
      const i = segStep + j;
      const bufIdx = pos + i;
      const w = isLast ? 1.0 : fadeOut[j];
      state.outputBufferL[bufIdx] += segOutL[i] * w;
      state.outputBufferR[bufIdx] += segOutR[i] * w;
      state.normBuffer[bufIdx] += w;
    }
  }

  private async runPipelineStream(
    chL: Float32Array,
    chR: Float32Array,
    sampleRate: number,
    controller: ReadableStreamDefaultController<Float32Array>,
  ) {
    const ortClient = this.ortClient;
    if (!ortClient || this.fftClients.length === 0 || this.ifftClients.length === 0) {
      throw new Error('流水线 Worker 未初始化');
    }

    const inputLen = chL.length;
    const state = this.initMainStreamingState(sampleRate, inputLen);

    state.inputBufferL.set(chL, state.inputBufferWritePos);
    state.inputBufferR.set(chR, state.inputBufferWritePos);
    state.inputBufferWritePos += inputLen;
    state.totalReceived += inputLen;

    const segExt = state.chunkSize + state.nfft;
    const tailPad = segExt;
    state.inputBufferWritePos += tailPad;

    const positions: number[] = [];
    let cursor = 0;
    while (state.inputBufferWritePos - cursor >= segExt) {
      positions.push(cursor);
      cursor += state.segStep;
    }

    const inflight = Math.max(2, Math.min(4, this.fftClients.length));
    const tasks = new Map<number, Promise<{ pos: number; segOutL: Float32Array; segOutR: Float32Array }>>();

    const startJob = (segmentIndex: number) => {
      const pos = positions[segmentIndex];
      const fftClient = this.fftClients[segmentIndex % this.fftClients.length];
      const ifftClient = this.ifftClients[segmentIndex % this.ifftClients.length];
      const segLCopy = state.inputBufferL.slice(pos, pos + segExt);
      const segRCopy = state.inputBufferR.slice(pos, pos + segExt);
      const promise = (async () => {
        const frames = await fftClient.compute(segmentIndex, segLCopy, segRCopy);
        const spec = await ortClient.run(segmentIndex, frames, state.dimF, state.dimT);
        const out = await ifftClient.compute(segmentIndex, spec);
        return { pos, segOutL: out.segOutL, segOutR: out.segOutR };
      })();
      tasks.set(segmentIndex, promise);
    };

    const initial = Math.min(inflight, positions.length);
    for (let i = 0; i < initial; i++) startJob(i);

    state.processedPos = 0;
    for (let i = 0; i < positions.length; i++) {
      const task = tasks.get(i);
      if (!task) {
        controller.error(new Error('流水线任务调度错误'));
        this.status = 'FREE';
        return;
      }

      try {
        const res = await task;
        tasks.delete(i);
        this.applySegmentToLinear(state, res.pos, res.segOutL, res.segOutR, false);
        state.processedPos += state.segStep;

        const safeOutputLen = state.processedPos - state.outputPos;
        if (safeOutputLen > 0) {
          this.emitStreamResult(state, safeOutputLen, controller);
        }
      } catch (err: unknown) {
        controller.error(err instanceof Error ? err : new Error(String(err)));
        this.status = 'FREE';
        return;
      }

      const next = i + inflight;
      if (next < positions.length) {
        startJob(next);
      }
    }

    const pendingLen = state.inputBufferWritePos - state.outputPos;
    if (pendingLen > 0) {
      this.emitStreamResult(state, pendingLen, controller);
    }

    controller.close();
    this.status = 'FREE';
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
    buffer.copyToChannel(toDomFloat32Array(chL), 0);
    buffer.copyToChannel(toDomFloat32Array(chR), 1);

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

function getMessageType(payload: unknown): string | null {
  if (typeof payload !== 'object' || payload === null) return null;
  if (!('type' in payload)) return null;
  const typeValue = (payload as { type?: unknown }).type;
  return typeof typeValue === 'string' ? typeValue : null;
}

function getMessageError(payload: unknown): string | null {
  if (typeof payload !== 'object' || payload === null) return null;
  if (!('error' in payload)) return null;
  const errorValue = (payload as { error?: unknown }).error;
  return typeof errorValue === 'string' ? errorValue : null;
}

function waitForType(worker: Worker, okType: string): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const handler = (e: MessageEvent) => {
      const payload = e.data as unknown;
      const type = getMessageType(payload);
      if (type === okType) {
        worker.removeEventListener('message', handler);
        resolve();
        return;
      }
      if (type === 'error') {
        worker.removeEventListener('message', handler);
        reject(new Error(getMessageError(payload) || 'Worker error'));
      }
    };
    worker.addEventListener('message', handler);
  });
}

class FftWorkerClient {
  private readonly pending = new Map<number, { resolve: (frames: Float32Array) => void; reject: (err: Error) => void }>();
  public constructor(private readonly worker: Worker, private readonly index: number) {
    this.worker.addEventListener('message', (e: MessageEvent) => {
      const payload = e.data as unknown;
      const type = getMessageType(payload);
      if (type === 'fft_result') {
        const dataValue = (payload as { data?: unknown }).data;
        if (typeof dataValue !== 'object' || dataValue === null) return;
        const jobId = (dataValue as { jobId?: unknown }).jobId;
        const frames = (dataValue as { frames?: unknown }).frames;
        if (typeof jobId !== 'number' || !(frames instanceof Float32Array)) return;
        const pending = this.pending.get(jobId);
        if (!pending) return;
        this.pending.delete(jobId);
        pending.resolve(frames);
        return;
      }
      if (type === 'error') {
        const errText = getMessageError(payload) || `FFT Worker ${this.index} 运行错误`;
        for (const [jobId, p] of this.pending.entries()) {
          this.pending.delete(jobId);
          p.reject(new Error(errText));
        }
      }
    });
  }

  public async init(params: StreamParams): Promise<void> {
    this.worker.postMessage({
      type: 'init',
      data: { dimF: params.dimF, dimT: params.dimT, nfft: params.nfft, hop: params.hop },
    });
    await waitForType(this.worker, 'inited');
  }

  public compute(jobId: number, segL: Float32Array, segR: Float32Array): Promise<Float32Array> {
    return new Promise<Float32Array>((resolve, reject) => {
      this.pending.set(jobId, { resolve, reject });
      this.worker.postMessage(
        { type: 'compute', data: { jobId, segL, segR } },
        [segL.buffer, segR.buffer],
      );
    });
  }
}

class IfftWorkerClient {
  private readonly pending = new Map<
    number,
    { resolve: (out: { segOutL: Float32Array; segOutR: Float32Array }) => void; reject: (err: Error) => void }
  >();
  public constructor(private readonly worker: Worker, private readonly index: number) {
    this.worker.addEventListener('message', (e: MessageEvent) => {
      const payload = e.data as unknown;
      const type = getMessageType(payload);
      if (type === 'ifft_result') {
        const dataValue = (payload as { data?: unknown }).data;
        if (typeof dataValue !== 'object' || dataValue === null) return;
        const jobId = (dataValue as { jobId?: unknown }).jobId;
        const segOutL = (dataValue as { segOutL?: unknown }).segOutL;
        const segOutR = (dataValue as { segOutR?: unknown }).segOutR;
        if (typeof jobId !== 'number' || !(segOutL instanceof Float32Array) || !(segOutR instanceof Float32Array)) return;
        const pending = this.pending.get(jobId);
        if (!pending) return;
        this.pending.delete(jobId);
        pending.resolve({ segOutL, segOutR });
        return;
      }
      if (type === 'error') {
        const errText = getMessageError(payload) || `IFFT Worker ${this.index} 运行错误`;
        for (const [jobId, p] of this.pending.entries()) {
          this.pending.delete(jobId);
          p.reject(new Error(errText));
        }
      }
    });
  }

  public async init(params: StreamParams): Promise<void> {
    this.worker.postMessage({
      type: 'init',
      data: { dimF: params.dimF, dimT: params.dimT, nfft: params.nfft, hop: params.hop },
    });
    await waitForType(this.worker, 'inited');
  }

  public compute(jobId: number, spec: Float32Array): Promise<{ segOutL: Float32Array; segOutR: Float32Array }> {
    return new Promise((resolve, reject) => {
      this.pending.set(jobId, { resolve, reject });
      this.worker.postMessage({ type: 'compute', data: { jobId, spec } }, [spec.buffer]);
    });
  }
}

class OrtWorkerClient {
  private pending: { resolve: (spec: Float32Array) => void; reject: (err: Error) => void } | null = null;
  private queue: Promise<void> = Promise.resolve();

  public constructor(private readonly worker: Worker) {
    this.worker.addEventListener('message', (e: MessageEvent) => {
      const payload = e.data as unknown;
      const type = getMessageType(payload);
      if (type === 'ort_result') {
        const dataValue = (payload as { data?: unknown }).data;
        if (typeof dataValue !== 'object' || dataValue === null) return;
        const spec = (dataValue as { spec?: unknown }).spec;
        if (!(spec instanceof Float32Array)) return;
        const pending = this.pending;
        if (!pending) return;
        this.pending = null;
        pending.resolve(spec);
        return;
      }
      if (type === 'error') {
        const pending = this.pending;
        this.pending = null;
        if (pending) pending.reject(new Error(getMessageError(payload) || 'ORT Worker 运行错误'));
      }
    });
  }

  public async preload(modelUrl: string, provider?: Provider): Promise<void> {
    this.worker.postMessage({ type: 'preload', data: { modelUrl, provider } });
    await waitForType(this.worker, 'preloaded');
  }

  public run(jobId: number, frames: Float32Array, dimF: number, dimT: number): Promise<Float32Array> {
    const runOnce = () =>
      new Promise<Float32Array>((resolve, reject) => {
        if (this.pending) {
          reject(new Error('ORT Worker 队列状态异常：存在未完成任务'));
          return;
        }
        this.pending = { resolve, reject };
        this.worker.postMessage({ type: 'run', data: { jobId, frames, dimF, dimT } }, [frames.buffer]);
      });

    const resultPromise = this.queue.then(runOnce, runOnce);
    this.queue = resultPromise.then(
      () => undefined,
      () => undefined,
    );
    return resultPromise;
  }
}

export default UVR;
