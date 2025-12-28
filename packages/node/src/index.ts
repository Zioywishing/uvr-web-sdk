import { rfft, irfft } from '@uvr-web-sdk/fft';
import * as ort from 'onnxruntime-node';

// 辅助函数
function hann(n: number) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    return w
}

interface StreamingState {
    session: ort.InferenceSession;
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

export class UVRNode {
    private state: StreamingState | null = null;
    private options: { modelUrl: string; provider?: string } | null = null;

    constructor(options?: { modelUrl: string; provider?: string }) {
        if (options) {
            this.options = options;
        }
    }

    async init(options?: { modelUrl: string; provider?: string }) {
        if (options) this.options = options;
        if (!this.options) throw new Error("Options required");
        
        const { modelUrl, provider } = this.options;
        
        let dimF = 3072, dimT = 256, nfft = 6144, hop = 1024;
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

        const providerUsed = provider === 'webgpu' || provider === 'wasm' ? 'cpu' : (provider || 'cpu');
        
        // 如果是本地路径，直接传给 create
        // onnxruntime-node 支持直接加载模型文件路径
        const session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: [providerUsed]
        });
        
        const inputName = session.inputNames ? session.inputNames[0] : 'input';

        const maxBufferSize = 1024 * 1024 * 10; 
        const padding = nfft; 

        this.state = {
            session,
            inputName,
            dimF, dimT, nfft, hop,
            chunkSize, segStep,
            win, fadeIn, fadeOut,
            
            inputBufferL: new Float32Array(maxBufferSize),
            inputBufferR: new Float32Array(maxBufferSize),
            inputBufferWritePos: padding, 
            
            outputBufferL: new Float32Array(maxBufferSize),
            outputBufferR: new Float32Array(maxBufferSize),
            normBuffer: new Float32Array(maxBufferSize),
            
            processedPos: 0,
            outputPos: padding
        };
    }

    async process(chL: Float32Array, chR: Float32Array): Promise<{ chL: Float32Array, chR: Float32Array }> {
        if (!this.state) await this.init();
        const state = this.state!;
        
        const len = chL.length;
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

        const resultsL: Float32Array[] = [];
        const resultsR: Float32Array[] = [];

        const segExt = state.chunkSize + state.nfft;
        while (state.inputBufferWritePos - state.processedPos >= segExt) {
            await this.processSegment(state, state.processedPos);
            state.processedPos += state.segStep;
            
            const safeOutputLen = state.processedPos - state.outputPos;
            if (safeOutputLen > 0) {
                 const { outL, outR } = this.extractOutput(state, safeOutputLen);
                 resultsL.push(outL);
                 resultsR.push(outR);
            }
        }
        
        const totalLen = resultsL.reduce((acc, cur) => acc + cur.length, 0);
        const mergedL = new Float32Array(totalLen);
        const mergedR = new Float32Array(totalLen);
        let offset = 0;
        for(let i=0; i<resultsL.length; i++) {
            mergedL.set(resultsL[i], offset);
            mergedR.set(resultsR[i], offset);
            offset += resultsL[i].length;
        }
        
        return { chL: mergedL, chR: mergedR };
    }

    async flush(): Promise<{ chL: Float32Array, chR: Float32Array }> {
        if (!this.state) return { chL: new Float32Array(0), chR: new Float32Array(0) };
        const state = this.state;
        const pendingLen = state.inputBufferWritePos - state.outputPos;
        if (pendingLen > 0) {
             const { outL, outR } = this.extractOutput(state, pendingLen);
             return { chL: outL, chR: outR };
        }
        return { chL: new Float32Array(0), chR: new Float32Array(0) };
    }

    private extractOutput(state: StreamingState, length: number) {
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
        return { outL, outR };
    }

    private async processSegment(state: StreamingState, pos: number) {
        const { session, inputName, dimF, dimT, nfft, hop, win, fadeIn, fadeOut, chunkSize, segStep } = state;
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
        
        const inputTensor = new ort.Tensor('float32', frames, [1, 4, dimF, dimT]);
        const outMap = await session.run({ [inputName]: inputTensor });
        const firstKey = Object.keys(outMap)[0];
        const spec = outMap[firstKey].data as Float32Array;

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
}
