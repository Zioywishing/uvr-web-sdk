import { parseOnnxInputShapes } from '@uvr-web-sdk/onnx-input-shape-parser';

type Provider = 'wasm' | 'webgpu';

type F32 = Float32Array<ArrayBufferLike>;
type DomF32 = Float32Array<ArrayBuffer>;

export interface UVROptions {
  modelUrl: string;
  provider?: Provider;
  workerCount?: number;
  fftWorkerUrl: string;
  ortWorkerUrl: string;
  ifftWorkerUrl: string;
}

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
  private fftWorkers: Worker[] = [];
  private ifftWorkers: Worker[] = [];
  private ortWorker: Worker | null = null;

  private fftClients: FftWorkerClient[] = [];
  private ifftClients: IfftWorkerClient[] = [];
  private ortClient: OrtWorkerClient | null = null;

  private streamParams: StreamParams | null = null;
  private initPromise: Promise<void> | null = null;
  private audioCtx: OfflineAudioContext | null = null;
  private abortController: AbortController | null = null;

  public constructor(options: UVROptions) {
    this.options = {
      ...options,
      workerCount: options.workerCount || 1,
    };
  }

  private async fetchModelAndParseShape(signal?: AbortSignal): Promise<StreamParams> {
    console.log('[UVR] 正在下载模型并解析输入形态...', this.options.modelUrl);
    const response = await fetch(this.options.modelUrl, { signal });
    const buffer = await response.arrayBuffer();
    if (signal?.aborted) throw new Error('操作已中止');

    const inputs = parseOnnxInputShapes(new Uint8Array(buffer));

    if (inputs.length === 0) {
      throw new Error('无法解析 ONNX 模型的输入形态');
    }

    const input = inputs[0];
    const shape = input.shape;
    console.log(`[UVR] 检测到输入: ${input.name}, 形状: [${shape.join(', ')}]`);

    // 期望形状类似 [batch, channels, dimF, dimT]
    // 例如 [1, 4, 3072, 256] 或 ['batch_size', 4, 2048, 256]
    if (shape.length < 4) {
      throw new Error(`模型输入维度不足: ${shape.length} (期望 >= 4)`);
    }

    const dimF = typeof shape[2] === 'number' ? shape[2] : 3072;
    const dimT = typeof shape[3] === 'number' ? shape[3] : 256;

    const nfft = dimF * 2;
    const hop = 1024;
    const chunkSize = hop * (dimT - 1);
    const segStep = chunkSize - nfft;

    return { dimF, dimT, nfft, hop, chunkSize, segStep };
  }

  private getStreamParams(): StreamParams {
    if (!this.streamParams) {
      throw new Error('StreamParams 未初始化，请先调用 init()');
    }
    return this.streamParams;
  }

  public async init() {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.abortController = new AbortController();
    const signal = this.abortController.signal;

    this.initPromise = (async () => {
      if (this.fftWorkers.length > 0 || this.ortWorker || this.ifftWorkers.length > 0) {
        return;
      }

      const count = this.options.workerCount || 1;
      const opts = this.options;
      
      console.log(`[UVR] 正在创建 ${count} 个 FFT Worker...`, opts.fftWorkerUrl);
      console.log('[UVR] 正在创建 ORT Worker...', opts.ortWorkerUrl);
      console.log(`[UVR] 正在创建 ${count} 个 IFFT Worker...`, opts.ifftWorkerUrl);

      let createdFftWorkers: Worker[] = [];
      let createdIfftWorkers: Worker[] = [];
      let createdOrtWorker: Worker | null = null;

      try {
        const indices = Array.from({ length: count }, (_, i) => i);
        
        // 1. 解析模型形状
        const streamParams = await this.fetchModelAndParseShape(signal);
        if (signal.aborted) throw new Error('操作已中止');

        // 2. 并行创建 Workers
        const [fftRes, ortRes, ifftRes] = await Promise.all([
          Promise.all(indices.map((i) => this.createFftWorker(opts.fftWorkerUrl, i, signal))),
          this.createOrtWorker(opts.ortWorkerUrl, signal),
          Promise.all(indices.map((i) => this.createIfftWorker(opts.ifftWorkerUrl, i, signal))),
        ]);

        createdFftWorkers = fftRes;
        createdOrtWorker = ortRes;
        createdIfftWorkers = ifftRes;

        if (signal.aborted) throw new Error('操作已中止');

        // 3. 初始化 Clients
        this.fftWorkers = createdFftWorkers;
        this.ortWorker = createdOrtWorker;
        this.ifftWorkers = createdIfftWorkers;
        this.streamParams = streamParams;

        this.fftClients = this.fftWorkers.map((w, i) => new FftWorkerClient(w, i));
        this.ifftClients = this.ifftWorkers.map((w, i) => new IfftWorkerClient(w, i));
        this.ortClient = new OrtWorkerClient(this.ortWorker);

        // 4. 并行初始化模型和 Worker 状态
        await Promise.all([
          this.ortClient.preload(this.options.modelUrl, this.options.provider, signal),
          Promise.all(this.fftClients.map((c) => c.init(streamParams, signal))),
          Promise.all(this.ifftClients.map((c) => c.init(streamParams, signal))),
        ]);

        if (signal.aborted) throw new Error('操作已中止');

        // 预创建并启动 OfflineAudioContext 以热身
        if (!this.audioCtx) {
          console.log('[UVR] 正在预创建 OfflineAudioContext (44100Hz)...');
          this.audioCtx = new OfflineAudioContext(2, 1, 44100);
        }

        this.status = 'FREE';
      } catch (e) {
        // 清理本次 init 过程中创建的所有资源
        createdFftWorkers.forEach(w => w.terminate());
        createdIfftWorkers.forEach(w => w.terminate());
        if (createdOrtWorker) createdOrtWorker.terminate();
        
        // 如果是当前正在进行的 init 失败，重置状态
        if (this.initPromise) {
          this.fftWorkers = [];
          this.ifftWorkers = [];
          this.ortWorker = null;
          this.fftClients = [];
          this.ifftClients = [];
          this.ortClient = null;
          this.initPromise = null;
          this.streamParams = null;
          this.status = 'UNINITIALIZED';
        }
        throw e;
      }
    })();

    return this.initPromise;
  }

  private async createFftWorker(workerUrl: string, index: number, signal?: AbortSignal): Promise<Worker> {
    const worker = new Worker(workerUrl, { type: 'module' });
    try {
      await waitForType(worker, 'worker_ready', signal);
      console.log(`[UVR] FFT Worker ${index} 就绪`);
      return worker;
    } catch (e) {
      worker.terminate();
      throw e;
    }
  }

  private async createIfftWorker(workerUrl: string, index: number, signal?: AbortSignal): Promise<Worker> {
    const worker = new Worker(workerUrl, { type: 'module' });
    try {
      await waitForType(worker, 'worker_ready', signal);
      console.log(`[UVR] IFFT Worker ${index} 就绪`);
      return worker;
    } catch (e) {
      worker.terminate();
      throw e;
    }
  }

  private async createOrtWorker(workerUrl: string, signal?: AbortSignal): Promise<Worker> {
    const worker = new Worker(workerUrl, { type: 'module' });
    try {
      await waitForType(worker, 'worker_ready', signal);
      console.log('[UVR] ORT Worker 就绪');
      return worker;
    } catch (e) {
      worker.terminate();
      throw e;
    }
  }

  public destroy() {
    console.log('[UVR] Destroying...');
    
    // 1. 立即发出中止信号
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }

    const destroyError = new Error('UVR 实例已销毁');

    // 2. 立即 Reject 所有 Client 中挂起的 Promise
    this.fftClients.forEach(c => c.terminate(destroyError));
    this.ifftClients.forEach(c => c.terminate(destroyError));
    if (this.ortClient) this.ortClient.terminate(destroyError);

    // 3. 严格销毁所有 Worker
    this.fftWorkers.forEach(w => w.terminate());
    this.ifftWorkers.forEach(w => w.terminate());
    if (this.ortWorker) this.ortWorker.terminate();

    // 4. 清空引用
    this.fftWorkers = [];
    this.ifftWorkers = [];
    this.ortWorker = null;
    this.fftClients = [];
    this.ifftClients = [];
    this.ortClient = null;
    this.audioCtx = null;
    this.status = 'UNINITIALIZED';
    this.initPromise = null;
    this.streamParams = null;
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

          if ((this.status as string) === 'UNINITIALIZED') return;

          if (!this.audioCtx) {
            this.audioCtx = new OfflineAudioContext(2, 1, 44100);
          }
          const audioCtx = this.audioCtx;

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

          if ((this.status as string) === 'UNINITIALIZED') return;

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
    const fftCount = this.fftWorkers.length;
    console.log(`[UVR] 使用流水线模式（FFT Worker: ${fftCount}, ORT Worker: 1, IFFT Worker: ${this.ifftWorkers.length}）`);
    await this.runPipelineStream(chL, chR, sampleRate, controller);
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
      throw new Error('流水线 Worker 未初始化或已销毁');
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
      // 检查是否已销毁
      if ((this.status as string) === 'UNINITIALIZED' || this.fftClients.length === 0) return;

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
      // 检查是否已销毁
      if ((this.status as string) === 'UNINITIALIZED') {
        throw new Error('处理中止：实例已销毁');
      }

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
        const error = err instanceof Error ? err : new Error(String(err));
        controller.error(error);
        if ((this.status as string) !== 'UNINITIALIZED') {
          this.status = 'FREE';
        }
        return;
      }

      const next = i + inflight;
      if (next < positions.length) {
        startJob(next);
      }
    }

    const pendingLen = state.inputBufferWritePos - state.outputPos;
    if (pendingLen > 0 && this.status !== 'UNINITIALIZED') {
      this.emitStreamResult(state, pendingLen, controller);
    }

    if (this.status !== 'UNINITIALIZED') {
      controller.close();
      this.status = 'FREE';
    }
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

function waitForType(worker: Worker, okType: string, signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      worker.removeEventListener('message', handler);
      reject(new Error('操作已中止'));
    };

    if (signal?.aborted) {
      return reject(new Error('操作已中止'));
    }
    signal?.addEventListener('abort', onAbort, { once: true });

    const handler = (e: MessageEvent) => {
      const payload = e.data as unknown;
      const type = getMessageType(payload);
      if (type === okType) {
        worker.removeEventListener('message', handler);
        signal?.removeEventListener('abort', onAbort);
        resolve();
        return;
      }
      if (type === 'error') {
        worker.removeEventListener('message', handler);
        signal?.removeEventListener('abort', onAbort);
        reject(new Error(getMessageError(payload) || 'Worker error'));
      }
    };
    worker.addEventListener('message', handler);
  });
}

class FftWorkerClient {
  private pending = new Map<number, { resolve: (frames: Float32Array) => void; reject: (err: Error) => void }>();
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
        this.terminate(new Error(errText));
      }
    });
  }

  public terminate(error: Error) {
    for (const p of this.pending.values()) {
      p.reject(error);
    }
    this.pending.clear();
  }

  public async init(params: StreamParams, signal?: AbortSignal): Promise<void> {
    this.worker.postMessage({
      type: 'init',
      data: { dimF: params.dimF, dimT: params.dimT, nfft: params.nfft, hop: params.hop },
    });
    await waitForType(this.worker, 'inited', signal);
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
  private pending = new Map<
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
        this.terminate(new Error(errText));
      }
    });
  }

  public terminate(error: Error) {
    for (const p of this.pending.values()) {
      p.reject(error);
    }
    this.pending.clear();
  }

  public async init(params: StreamParams, signal?: AbortSignal): Promise<void> {
    this.worker.postMessage({
      type: 'init',
      data: { dimF: params.dimF, dimT: params.dimT, nfft: params.nfft, hop: params.hop },
    });
    await waitForType(this.worker, 'inited', signal);
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
        this.terminate(new Error(getMessageError(payload) || 'ORT Worker 运行错误'));
      }
    });
  }

  public terminate(error: Error) {
    if (this.pending) {
      this.pending.reject(error);
      this.pending = null;
    }
    // 清空队列
    this.queue = Promise.resolve();
  }

  public async preload(modelUrl: string, provider?: Provider, signal?: AbortSignal): Promise<void> {
    this.worker.postMessage({ type: 'preload', data: { modelUrl, provider } });
    await waitForType(this.worker, 'preloaded', signal);
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
