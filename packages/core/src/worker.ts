import { rfft, irfft } from '@uar/fft';
import { ort, type InferenceSession } from '@uar/ort-runtime-webgpu';

// 定义 WebGPU 相关接口，避免使用 any
interface WebGPUAdapter {
    requestDevice(): Promise<unknown>;
}

interface WebGPU {
    requestAdapter(): Promise<WebGPUAdapter | null>;
}

interface NavigatorWithGPU extends Navigator {
    gpu?: WebGPU;
}

console.log('[Worker] 模块加载完成');

// Helpers
function hann(n: number) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    return w
}

interface StreamingState {
    session: InferenceSession;
    inputName: string;
    dimF: number;
    dimT: number;
    nfft: number;
    hop: number;
    chunkSize: number;
    segStep: number;
    win: Float32Array;
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
}

let preloadedSession: InferenceSession | null = null;
let preloadedModelUrl: string | null = null;
let preloadedProvider: string | null = null;
let preloadedStreamingState: StreamingState | null = null;

let streamingState: StreamingState | null = null;
let taskQueue = Promise.resolve();

type WorkerMessage = 
  | { type: 'preload'; data: { modelUrl: string; provider?: string } }
  | { type: 'stream_start'; data: { modelUrl: string; provider?: string; sampleRate: number } }
  | { type: 'stream_data'; data: { chL: Float32Array; chR: Float32Array } }
  | { type: 'stream_end'; data?: never };

// 监听消息
self.addEventListener('message', (e: MessageEvent) => {
    // console.log('[Worker] Raw message received:', e.data?.type);
    taskQueue = taskQueue.then(async () => {
        const msg = e.data as WorkerMessage;
        const { type } = msg;

        try {
            if (type === 'preload') {
                const { modelUrl, provider } = msg.data;
                console.log('[Worker] 收到 preload, 模型:', modelUrl, 'Provider:', provider || 'wasm');
                
                // 如果已经加载了相同的模型和 provider，直接返回
                if (preloadedSession && preloadedModelUrl === modelUrl && preloadedProvider === (provider || 'wasm')) {
                    console.log('[Worker] 模型已预加载，跳过');
                    self.postMessage({ type: 'preloaded' });
                    return;
                }

                preloadedSession = await createSession(modelUrl, provider);
                preloadedModelUrl = modelUrl;
                preloadedProvider = provider || 'wasm';
                
                // 预先初始化流状态（预分配缓冲区和计算窗函数）
                console.log('[Worker] 正在预分配缓冲区和计算窗函数...');
                preloadedStreamingState = await initStreamingWithSession(preloadedSession, { sampleRate: 44100 }); // sampleRate 暂时没用，传个默认值

                self.postMessage({ type: 'preloaded' });
                console.log('[Worker] 预加载完成');
            } else if (type === 'stream_start') {
                console.log('[Worker] 收到 stream_start, 模型:', msg.data.modelUrl, 'Provider:', msg.data.provider || 'wasm');
                
                // 检查是否可以使用预加载的 session 和已分配好的状态
                if (preloadedStreamingState && preloadedModelUrl === msg.data.modelUrl && preloadedProvider === (msg.data.provider || 'wasm')) {
                    console.log('[Worker] 使用预加载的状态并重置指针');
                    streamingState = resetStreamingState(preloadedStreamingState);
                } else {
                    console.log('[Worker] 未命中预加载或配置变更，重新初始化');
                    streamingState = await initStreaming(msg.data);
                }
                
                self.postMessage({
                    type: 'stream_started',
                    data: {
                        nfft: streamingState.nfft,
                        hop: streamingState.hop,
                        chunkSize: streamingState.chunkSize,
                        segStep: streamingState.segStep
                    }
                });
                console.log('[Worker] 初始化完成');
            } else if (type === 'stream_data') {
                if (!streamingState) throw new Error('Streaming not started');
                // console.log('[Worker] 收到音频数据, 长度:', msg.data.chL.length);
                await processStreamChunk(streamingState, msg.data);
            } else if (type === 'stream_end') {
                if (!streamingState) throw new Error('Streaming not started');
                console.log('[Worker] 收到 stream_end, 正在刷新缓冲区...');
                await flushStream(streamingState);
                streamingState = null;
                self.postMessage({ type: 'stream_ended' });
                console.log('[Worker] 处理结束');
            }
        } catch (err: unknown) {
            const error = err instanceof Error ? err.message : String(err);
            console.error('[Worker] Error processing message:', type, error);
            self.postMessage({
                type: 'error',
                error: error
            });
        }
    }).catch(err => {
        console.error('[Worker] TaskQueue error:', err);
    });
});

// 通知主线程 Worker 已就绪
console.log('[Worker] Worker Ready');
self.postMessage({ type: 'worker_ready' });

async function createSession(modelUrl: string, provider?: string): Promise<InferenceSession> {
    let providerUsed = provider || 'wasm';
    
    // WebGPU Check
    if (providerUsed === 'webgpu') {
        const nav = self.navigator as NavigatorWithGPU;
        if (!nav.gpu) {
            throw new Error('WebGPU is not supported by this browser.');
        }
        
        if (!ort.env.webgpu) {
                ort.env.webgpu = { device: null };
        }

        if (!ort.env.webgpu!.device) {
                const adapter = await nav.gpu.requestAdapter();
                if (!adapter) {
                    throw new Error('No WebGPU adapter found.');
                }
                ort.env.webgpu!.device = await adapter.requestDevice();
        }
    }

    return await ort.InferenceSession.create(modelUrl, {
        executionProviders: [providerUsed]
    });
}

async function initStreaming(data: { modelUrl: string, provider?: string, sampleRate: number }): Promise<StreamingState> {
    const session = await createSession(data.modelUrl, data.provider);
    return initStreamingWithSession(session, data);
}

async function initStreamingWithSession(session: InferenceSession, data: { sampleRate: number }): Promise<StreamingState> {
    let dimF = 3072, dimT = 256, nfft = 6144, hop = 1024;
    // Simple heuristic for model params based on name
    // if (modelUrl.includes('9662') || (modelUrl.includes('Inst_3') && !modelUrl.includes('HQ')) || modelUrl.includes('KARA') || modelUrl.includes('Kim_Inst')) {
    //     dimF = 2048;
    //     nfft = 4096;
    // }

    const chunkSize = hop * (dimT - 1);
    const segStep = chunkSize - nfft;
    const win = hann(nfft);
    const fadeIn = new Float32Array(nfft);
    const fadeOut = new Float32Array(nfft);
    for (let i = 0; i < nfft; i++) {
        const w = 0.5 * (1 - Math.cos(Math.PI * i / (nfft - 1)));
        fadeIn[i] = w;
        fadeOut[i] = 1 - w;
    }

    const inputName = session.inputNames ? session.inputNames[0] : 'input';

    const maxBufferSize = 1024 * 1024 * 10; // 10MB buffer roughly
    const padding = nfft; 

    return {
        session,
        inputName,
        dimF, dimT, nfft, hop,
        chunkSize, segStep,
        win, fadeIn, fadeOut,
        
        inputBufferL: new Float32Array(maxBufferSize),
        inputBufferR: new Float32Array(maxBufferSize),
        inputBufferWritePos: padding, // Start with padding
        
        outputBufferL: new Float32Array(maxBufferSize),
        outputBufferR: new Float32Array(maxBufferSize),
        normBuffer: new Float32Array(maxBufferSize),
        
        processedPos: 0,
        outputPos: padding
    };
}

async function processStreamChunk(state: StreamingState, data: { chL: Float32Array, chR: Float32Array }) {
    const { chL, chR } = data;
    const len = chL.length;
    
    // Shift buffer if needed
    if (state.inputBufferWritePos + len > state.inputBufferL.length) {
        const remainingInput = state.inputBufferWritePos - state.processedPos;
        state.inputBufferL.copyWithin(0, state.processedPos, state.inputBufferWritePos);
        state.inputBufferR.copyWithin(0, state.processedPos, state.inputBufferWritePos);
        
        state.outputBufferL.copyWithin(0, state.processedPos, state.outputBufferL.length);
        state.outputBufferR.copyWithin(0, state.processedPos, state.outputBufferR.length);
        state.normBuffer.copyWithin(0, state.processedPos, state.normBuffer.length);

        const validLen = state.outputBufferL.length - state.processedPos;
        state.outputBufferL.fill(0, validLen);
        state.outputBufferR.fill(0, validLen);
        state.normBuffer.fill(0, validLen);

        state.inputBufferWritePos = remainingInput;
        state.outputPos = Math.max(0, state.outputPos - state.processedPos);
        state.processedPos = 0;
    }
    
    state.inputBufferL.set(chL, state.inputBufferWritePos);
    state.inputBufferR.set(chR, state.inputBufferWritePos);
    state.inputBufferWritePos += len;

    const segExt = state.chunkSize + state.nfft;
    while (state.inputBufferWritePos - state.processedPos >= segExt) {
        await processSegment(state, state.processedPos);
        state.processedPos += state.segStep;
        
        const safeOutputLen = state.processedPos - state.outputPos;
        if (safeOutputLen > 0) {
            emitStreamResult(state, safeOutputLen);
        }
    }
}

async function processSegment(state: StreamingState, pos: number) {
    const { session, inputName, dimF, dimT, nfft, hop, win, fadeIn, fadeOut, chunkSize, segStep } = state;
    // console.log('[Worker] 处理片段, 位置:', pos);
    const segExt = chunkSize + nfft;
    
    const segL = state.inputBufferL.subarray(pos, pos + segExt);
    const segR = state.inputBufferR.subarray(pos, pos + segExt);

    const frames = new Float32Array(4 * dimF * dimT);
    for (let t = 0; t < dimT; t++) {
        const start = t * hop;
        const wl = new Float32Array(nfft);
        const wr = new Float32Array(nfft);
        for (let i = 0; i < nfft; i++) {
            wl[i] = (segL[start + i] || 0) * win[i];
            wr[i] = (segR[start + i] || 0) * win[i];
        }
        const outSpecL = rfft(wl);
        const outSpecR = rfft(wr);
        for (let k = 0; k < dimF; k++) {
            const base = k * dimT + t;
            const idx = k * 2;
            frames[base] = outSpecL[idx];
            frames[dimT * dimF + base] = outSpecL[idx + 1];
            frames[2 * dimT * dimF + base] = outSpecR[idx];
            frames[3 * dimT * dimF + base] = outSpecR[idx + 1];
        }
    }
    
    let maxFrames = 0;
    for (let i = 0; i < frames.length; i++) {
        maxFrames = Math.max(maxFrames, Math.abs(frames[i]));
    }

    const inputTensor = new ort.Tensor('float32', frames, [1, 4, dimF, dimT]);
    const outMap = await session.run({ [inputName]: inputTensor });
    const firstKey = Object.keys(outMap)[0];
    const spec = outMap[firstKey].data;

    // 检查模型输出是否为全零
    let maxSpec = 0;
    for (let i = 0; i < spec.length; i++) {
        maxSpec = Math.max(maxSpec, Math.abs(spec[i]));
    }
    if (maxSpec === 0) {
        console.error('[Worker] 模型输出全零! 输入最大值:', maxFrames.toFixed(6));
    }

    const segOutL = new Float32Array(segExt);
    const segOutR = new Float32Array(segExt);
    const segNorm = new Float32Array(segExt);

    for (let t = 0; t < dimT; t++) {
        const start = t * hop;
        const specL = new Float32Array(2 * nfft);
        const specR = new Float32Array(2 * nfft);
        for (let k = 0; k < dimF; k++) {
            const base = k * dimT + t;
            const idx = k * 2;
            specL[idx] = spec[base];
            specL[idx + 1] = spec[dimT * dimF + base];
            specR[idx] = spec[2 * dimT * dimF + base];
            specR[idx + 1] = spec[3 * dimT * dimF + base];
        }
        const xL = irfft(specL);
        const xR = irfft(specR);
        for (let i = 0; i < nfft; i++) {
            const w = win[i];
            segOutL[start + i] += xL[i] * w;
            segOutR[start + i] += xR[i] * w;
            segNorm[start + i] += w * w;
        }
    }

    const dMin = 1e-8;
    for (let i = 0; i < segExt; i++) {
        const d = segNorm[i] || dMin;
        segOutL[i] /= d;
        segOutR[i] /= d;
    }

    const isFirst = pos === 0;
    const writeMax = segExt;
    const headLen = isFirst ? Math.min(nfft, writeMax) : nfft; 
    
    for (let i = 0; i < headLen; i++) {
        const bufIdx = (pos + i) % state.outputBufferL.length;
        const w = fadeIn[i];
        state.outputBufferL[bufIdx] += segOutL[i] * w;
        state.outputBufferR[bufIdx] += segOutR[i] * w;
        state.normBuffer[bufIdx] += w;
    }
    for (let i = headLen; i < Math.min(segStep, writeMax); i++) {
        const bufIdx = (pos + i) % state.outputBufferL.length;
        state.outputBufferL[bufIdx] += segOutL[i];
        state.outputBufferR[bufIdx] += segOutR[i];
        state.normBuffer[bufIdx] += 1;
    }
    const tailLen = Math.min(nfft, Math.max(0, writeMax - segStep));
    for (let j = 0; j < tailLen; j++) {
        const i = segStep + j;
        const bufIdx = (pos + i) % state.outputBufferL.length;
        const w = fadeOut[j];
        state.outputBufferL[bufIdx] += segOutL[i] * w;
        state.outputBufferR[bufIdx] += segOutR[i] * w;
        state.normBuffer[bufIdx] += w;
    }
}

function emitStreamResult(state: StreamingState, length: number) {
    const outL = new Float32Array(length);
    const outR = new Float32Array(length);
    
    for (let i = 0; i < length; i++) {
        const bufIdx = (state.outputPos + i) % state.outputBufferL.length;
        const d = state.normBuffer[bufIdx] || 1;
        outL[i] = state.outputBufferL[bufIdx] / d;
        outR[i] = state.outputBufferR[bufIdx] / d;
        
        state.outputBufferL[bufIdx] = 0;
        state.outputBufferR[bufIdx] = 0;
        state.normBuffer[bufIdx] = 0;
    }
    
    state.outputPos += length;
    self.postMessage({
        type: 'stream_result',
        data: { chL: outL, chR: outR }
    }, [outL.buffer, outR.buffer]);
}

async function flushStream(state: StreamingState) {
    const pendingLen = state.inputBufferWritePos - state.outputPos;
    if (pendingLen > 0) {
        emitStreamResult(state, pendingLen);
    }
}

function resetStreamingState(state: StreamingState): StreamingState {
    const padding = state.nfft;
    state.inputBufferL.fill(0);
    state.inputBufferR.fill(0);
    state.inputBufferWritePos = padding;
    
    state.outputBufferL.fill(0);
    state.outputBufferR.fill(0);
    state.normBuffer.fill(0);
    
    state.processedPos = 0;
    state.outputPos = padding;
    return state;
}
