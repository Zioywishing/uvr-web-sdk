import { rfft } from '@uvr-web-sdk/fft';

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
    segL: Float32Array;
    segR: Float32Array;
  };
}

interface ErrorMessage {
  type: 'error';
  error: string;
}

interface FFTResultMessage {
  type: 'fft_result';
  data: {
    jobId: number;
    frames: Float32Array;
  };
}

type OutgoingMessage = { type: 'worker_ready' } | { type: 'inited' } | FFTResultMessage | ErrorMessage;

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

let wl: Float32Array | null = null;
let wr: Float32Array | null = null;

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
    wl = new Float32Array(msg.data.nfft);
    wr = new Float32Array(msg.data.nfft);
    const out: OutgoingMessage = { type: 'inited' };
    self.postMessage(out);
    return;
  }

  if (type === 'compute') {
    try {
      if (dimF === null || dimT === null || nfft === null || hop === null || win === null || wl === null || wr === null) {
        throw new Error('FFT Worker 尚未初始化');
      }

      const msg = payload as ComputeMessage;
      const { jobId, segL, segR } = msg.data;

      const frames = new Float32Array(4 * dimF * dimT);
      for (let t = 0; t < dimT; t++) {
        const start = t * hop;
        for (let i = 0; i < nfft; i++) {
          const li = segL[start + i];
          const ri = segR[start + i];
          const w = win[i];
          wl[i] = (Number.isFinite(li) ? li : 0) * w;
          wr[i] = (Number.isFinite(ri) ? ri : 0) * w;
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

      const out: FFTResultMessage = {
        type: 'fft_result',
        data: { jobId, frames }
      };
      self.postMessage(out, [frames.buffer]);
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
