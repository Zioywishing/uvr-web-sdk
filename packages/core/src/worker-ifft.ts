import { irfft } from '@uvr-web-sdk/fft';

interface InitMessage {
  type: 'init';
  data: {
    dimF: number;
    dimT: number;
    nfft: number;
    hop: number;
  };
}

interface ComputeMessage {
  type: 'compute';
  data: {
    jobId: number;
    spec: Float32Array;
  };
}

interface ErrorMessage {
  type: 'error';
  error: string;
}

interface IFFTResultMessage {
  type: 'ifft_result';
  data: {
    jobId: number;
    segOutL: Float32Array;
    segOutR: Float32Array;
  };
}

type OutgoingMessage = { type: 'worker_ready' } | { type: 'inited' } | IFFTResultMessage | ErrorMessage;

function hann(n: number): Float32Array {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  return w;
}

function getMessageType(payload: unknown): string | null {
  if (typeof payload !== 'object' || payload === null) return null;
  if (!('type' in payload)) return null;
  const t = (payload as { type?: unknown }).type;
  return typeof t === 'string' ? t : null;
}

let dimF: number | null = null;
let dimT: number | null = null;
let nfft: number | null = null;
let hop: number | null = null;
let win: Float32Array | null = null;

let specL: Float32Array | null = null;
let specR: Float32Array | null = null;

self.addEventListener('message', (e: MessageEvent) => {
  const payload = e.data as unknown;
  const type = getMessageType(payload);

  if (type === 'init') {
    const msg = payload as InitMessage;
    dimF = msg.data.dimF;
    dimT = msg.data.dimT;
    nfft = msg.data.nfft;
    hop = msg.data.hop;
    win = hann(msg.data.nfft);
    specL = new Float32Array(2 * msg.data.nfft);
    specR = new Float32Array(2 * msg.data.nfft);
    const out: OutgoingMessage = { type: 'inited' };
    self.postMessage(out);
    return;
  }

  if (type === 'compute') {
    try {
      if (dimF === null || dimT === null || nfft === null || hop === null || win === null || specL === null || specR === null) {
        throw new Error('IFFT Worker 尚未初始化');
      }

      const msg = payload as ComputeMessage;
      const { jobId, spec } = msg.data;

      const chunkSize = hop * (dimT - 1);
      const segExt = chunkSize + nfft;
      const segOutL = new Float32Array(segExt);
      const segOutR = new Float32Array(segExt);
      const segNorm = new Float32Array(segExt);

      for (let t = 0; t < dimT; t++) {
        const start = t * hop;
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

      const out: IFFTResultMessage = {
        type: 'ifft_result',
        data: { jobId, segOutL, segOutR }
      };
      self.postMessage(out, [segOutL.buffer, segOutR.buffer]);
    } catch (err: unknown) {
      const out: ErrorMessage = {
        type: 'error',
        error: err instanceof Error ? err.message : String(err)
      };
      self.postMessage(out);
    }
  }
});

const ready: OutgoingMessage = { type: 'worker_ready' };
self.postMessage(ready);
