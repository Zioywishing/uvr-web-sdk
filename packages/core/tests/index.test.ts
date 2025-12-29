import { expect, test } from 'vitest'
import { UVR } from '../src'

test('UVR instantiation', () => {
  const uvr = new UVR({
    modelUrl: 'model.onnx',
    fftWorkerUrl: 'worker-fft.js',
    ortWorkerUrl: 'worker-ort.js',
    ifftWorkerUrl: 'worker-ifft.js'
  })
  expect(uvr).toBeDefined();
})

type MessageListener = (event: MessageEvent) => void

interface WorkerLike {
  addEventListener(type: 'message', listener: MessageListener): void
  removeEventListener(type: 'message', listener: MessageListener): void
  postMessage(message: unknown, transfer?: Transferable[]): void
  terminate(): void
}

function concatFloat32Arrays(chunks: readonly Float32Array[]): Float32Array {
  let total = 0
  for (const chunk of chunks) total += chunk.length
  const out = new Float32Array(total)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.length
  }
  return out
}

test('init 并行启动模型解析与 Worker 创建', async () => {
  class FakeInitWorker implements WorkerLike {
    private readonly listeners = new Set<MessageListener>()

    public addEventListener(type: 'message', listener: MessageListener): void {
      if (type !== 'message') return
      this.listeners.add(listener)
    }

    public removeEventListener(type: 'message', listener: MessageListener): void {
      if (type !== 'message') return
      this.listeners.delete(listener)
    }

    public postMessage(message: unknown, _transfer?: Transferable[]): void {
      const payload = message as { type?: unknown }
      const type = payload.type
      if (type === 'init') {
        queueMicrotask(() => this.emit({ type: 'inited' }))
      }
      if (type === 'preload') {
        queueMicrotask(() => this.emit({ type: 'preloaded' }))
      }
    }

    public terminate(): void {
      this.listeners.clear()
    }

    private emit(message: { type: string }): void {
      const event = { data: message } as unknown as MessageEvent
      for (const listener of this.listeners) {
        listener(event)
      }
    }
  }

  class FakeOfflineAudioContext {
    public constructor(_numberOfChannels: number, _length: number, _sampleRate: number) {}
  }
  ;(globalThis as unknown as { OfflineAudioContext?: unknown }).OfflineAudioContext = FakeOfflineAudioContext

  const uvr = new UVR({
    modelUrl: 'model.onnx',
    fftWorkerUrl: 'worker-fft.js',
    ortWorkerUrl: 'worker-ort.js',
    ifftWorkerUrl: 'worker-ifft.js',
    workerCount: 2
  })

  interface StreamParams {
    dimF: number
    dimT: number
    nfft: number
    hop: number
    chunkSize: number
    segStep: number
  }
  interface UVRPrivate {
    fetchModelAndParseShape(): Promise<StreamParams>
    createFftWorker(workerUrl: string, index: number): Promise<Worker>
    createOrtWorker(workerUrl: string): Promise<Worker>
    createIfftWorker(workerUrl: string, index: number): Promise<Worker>
  }

  const started: string[] = []

  let resolveShape!: (p: StreamParams) => void
  const shapePromise = new Promise<StreamParams>((resolve) => {
    resolveShape = resolve
  })

  let releaseFft!: () => void
  const fftGate = new Promise<void>((resolve) => {
    releaseFft = resolve
  })

  let releaseOrt!: () => void
  const ortGate = new Promise<void>((resolve) => {
    releaseOrt = resolve
  })

  let releaseIfft!: () => void
  const ifftGate = new Promise<void>((resolve) => {
    releaseIfft = resolve
  })

  const priv = uvr as unknown as UVRPrivate
  priv.fetchModelAndParseShape = () => {
    started.push('shape')
    return shapePromise
  }
  priv.createFftWorker = async (_workerUrl: string, index: number) => {
    started.push(`fft-${index}`)
    await fftGate
    return new FakeInitWorker() as unknown as Worker
  }
  priv.createOrtWorker = async (_workerUrl: string) => {
    started.push('ort')
    await ortGate
    return new FakeInitWorker() as unknown as Worker
  }
  priv.createIfftWorker = async (_workerUrl: string, index: number) => {
    started.push(`ifft-${index}`)
    await ifftGate
    return new FakeInitWorker() as unknown as Worker
  }

  const initPromise = uvr.init()
  await Promise.resolve()

  expect(started).toContain('shape')
  expect(started).toContain('ort')
  expect(started).toContain('fft-0')
  expect(started).toContain('fft-1')
  expect(started).toContain('ifft-0')
  expect(started).toContain('ifft-1')

  releaseFft()
  releaseOrt()
  releaseIfft()
  resolveShape({ dimF: 1, dimT: 1, nfft: 1, hop: 1, chunkSize: 1, segStep: 1 })

  await initPromise
})

test('流水线模式输出长度与对齐偏移正确', async () => {
  const sampleCount = 20
  const chL = new Float32Array(sampleCount)
  const chR = new Float32Array(sampleCount)
  for (let i = 0; i < sampleCount; i++) {
    chL[i] = i
    chR[i] = 1000 + i
  }

  const uvr = new UVR({
    modelUrl: 'model.onnx',
    fftWorkerUrl: 'worker-fft.js',
    ortWorkerUrl: 'worker-ort.js',
    ifftWorkerUrl: 'worker-ifft.js',
    provider: 'wasm',
    workerCount: 2
  })

  const dimF = 3072
  const dimT = 256
  const nfft = 6144
  const hop = 1024
  const chunkSize = hop * (dimT - 1)
  const segExt = chunkSize + nfft
  const segStep = chunkSize - nfft
  const padSamples = Math.max(0, Math.floor(0.4 * 44100))

  interface FftClientLike {
    compute(jobId: number, segL: Float32Array, segR: Float32Array): Promise<Float32Array>
  }
  interface OrtClientLike {
    run(jobId: number, frames: Float32Array, dimF: number, dimT: number): Promise<Float32Array>
  }
  interface IfftClientLike {
    compute(jobId: number, spec: Float32Array): Promise<{ segOutL: Float32Array; segOutR: Float32Array }>
  }

  const fakeFft: FftClientLike = {
    async compute(_jobId: number, _segL: Float32Array, _segR: Float32Array) {
      return new Float32Array(4 * dimF * dimT)
    }
  }
  const fakeOrt: OrtClientLike = {
    async run(_jobId: number, frames: Float32Array) {
      return frames
    }
  }
  const fakeIfft: IfftClientLike = {
    async compute(_jobId: number, _spec: Float32Array) {
      const segOutL = new Float32Array(segExt)
      const segOutR = new Float32Array(segExt)
      for (let i = 0; i < segExt; i++) {
        segOutL[i] = i
        segOutR[i] = 1000 + i
      }
      return { segOutL, segOutR }
    }
  }

  ;(uvr as unknown as { initPromise: Promise<void> | null }).initPromise = Promise.resolve()
  ;(uvr as unknown as { streamParams: unknown }).streamParams = { dimF, dimT, nfft, hop, chunkSize, segStep }
  ;(uvr as unknown as { fftClients: unknown[] }).fftClients = [fakeFft, fakeFft]
  ;(uvr as unknown as { ifftClients: unknown[] }).ifftClients = [fakeIfft, fakeIfft]
  ;(uvr as unknown as { ortClient: unknown }).ortClient = fakeOrt

  const received: Float32Array[] = []
  const controller = {
    enqueue(chunk: Float32Array) {
      received.push(chunk)
    },
    close() {},
    error(err: unknown) {
      throw err instanceof Error ? err : new Error(String(err))
    }
  }

  type RunParallelStream = (
    left: Float32Array,
    right: Float32Array,
    sampleRate: number,
    controller: ReadableStreamDefaultController<Float32Array>
  ) => Promise<void>

  const runParallelStream = (uvr as unknown as { runParallelStream: RunParallelStream }).runParallelStream
  await runParallelStream.call(
    uvr,
    chL,
    chR,
    44100,
    controller as unknown as ReadableStreamDefaultController<Float32Array>
  )

  const out = concatFloat32Arrays(received)
  expect(out.length).toBe(sampleCount * 2)
  for (let i = 0; i < sampleCount; i++) {
    expect(out[i * 2]).toBe(padSamples + i)
    expect(out[i * 2 + 1]).toBe(1000 + padSamples + i)
  }
})
