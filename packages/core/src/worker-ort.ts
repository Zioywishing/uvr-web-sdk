import { ort, type InferenceSession } from '@uar/ort-runtime-webgpu';

interface WebGPUAdapter {
  requestDevice(): Promise<unknown>;
}

interface WebGPU {
  requestAdapter(): Promise<WebGPUAdapter | null>;
}

interface NavigatorWithGPU extends Navigator {
  gpu?: WebGPU;
}

interface PreloadMessage {
  type: 'preload';
  data: {
    modelUrl: string;
    provider?: 'wasm' | 'webgpu';
  };
}

interface RunMessage {
  type: 'run';
  data: {
    jobId: number;
    frames: Float32Array;
    dimF: number;
    dimT: number;
  };
}

interface ErrorMessage {
  type: 'error';
  error: string;
}

interface ORTResultMessage {
  type: 'ort_result';
  data: {
    jobId: number;
    spec: Float32Array;
  };
}

type OutgoingMessage = { type: 'worker_ready' } | { type: 'preloaded' } | ORTResultMessage | ErrorMessage;

function getMessageType(payload: unknown): string | null {
  if (typeof payload !== 'object' || payload === null) return null;
  if (!('type' in payload)) return null;
  const t = (payload as { type?: unknown }).type;
  return typeof t === 'string' ? t : null;
}

let session: InferenceSession | null = null;
let inputName: string | null = null;

async function createSession(modelUrl: string, provider?: 'wasm' | 'webgpu'): Promise<InferenceSession> {
  const providerUsed = provider ?? 'wasm';

  if (providerUsed === 'webgpu') {
    const nav = self.navigator as NavigatorWithGPU;
    if (!nav.gpu) {
      throw new Error('WebGPU is not supported by this browser.');
    }

    if (!ort.env.webgpu) {
      ort.env.webgpu = { device: null };
    }

    if (!ort.env.webgpu.device) {
      const adapter = await nav.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter found.');
      ort.env.webgpu.device = await adapter.requestDevice();
    }
  }

  return ort.InferenceSession.create(modelUrl, {
    executionProviders: [providerUsed]
  });
}

self.addEventListener('message', (e: MessageEvent) => {
  const payload = e.data as unknown;
  const type = getMessageType(payload);

  if (type === 'preload') {
    (async () => {
      try {
        const msg = payload as PreloadMessage;
        session = await createSession(msg.data.modelUrl, msg.data.provider);
        inputName = session.inputNames && session.inputNames.length > 0 ? session.inputNames[0] : 'input';
        const out: OutgoingMessage = { type: 'preloaded' };
        self.postMessage(out);
      } catch (err: unknown) {
        const out: ErrorMessage = { type: 'error', error: err instanceof Error ? err.message : String(err) };
        self.postMessage(out);
      }
    })();
    return;
  }

  if (type === 'run') {
    (async () => {
      try {
        if (!session || !inputName) throw new Error('ORT Worker 尚未 preload');
        const msg = payload as RunMessage;

        const tensor = new ort.Tensor('float32', msg.data.frames, [1, 4, msg.data.dimF, msg.data.dimT]);
        const outMap = await session.run({ [inputName]: tensor });
        const keys = Object.keys(outMap);
        if (keys.length === 0) throw new Error('模型未返回任何输出');
        const firstKey = keys[0];
        const firstTensor = outMap[firstKey];
        const data = (firstTensor as unknown as { data?: unknown }).data;
        if (!(data instanceof Float32Array)) {
          throw new Error('模型输出类型不是 Float32Array');
        }

        const out: ORTResultMessage = {
          type: 'ort_result',
          data: { jobId: msg.data.jobId, spec: data }
        };
        self.postMessage(out, [data.buffer]);
      } catch (err: unknown) {
        const out: ErrorMessage = { type: 'error', error: err instanceof Error ? err.message : String(err) };
        self.postMessage(out);
      }
    })();
  }
});

const ready: OutgoingMessage = { type: 'worker_ready' };
self.postMessage(ready);
