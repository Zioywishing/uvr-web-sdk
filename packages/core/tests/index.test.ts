import { expect, test } from 'vitest'
import { UVR } from '../src'

test('UVR instantiation', () => {
  const uvr = new UVR({ modelUrl: 'model.onnx', workerUrl: 'worker.js' });
  expect(uvr).toBeDefined();
})

type MessageListener = (event: MessageEvent) => void

interface WorkerLike {
  addEventListener(type: 'message', listener: MessageListener): void
  removeEventListener(type: 'message', listener: MessageListener): void
  postMessage(message: unknown, transfer?: Transferable[]): void
  terminate(): void
}

type IncomingMessage =
  | { type: 'stream_start'; data: { modelUrl: string; provider?: string; sampleRate: number } }
  | { type: 'stream_data'; data: { chL: Float32Array; chR: Float32Array } }
  | { type: 'stream_end'; data?: never }

type OutgoingMessage =
  | { type: 'stream_started'; data: { nfft: number; hop: number; chunkSize: number; segStep: number } }
  | { type: 'stream_result'; data: { chL: Float32Array; chR: Float32Array } }
  | { type: 'stream_ended' }
  | { type: 'error'; error: string }

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

class FakeWorker implements WorkerLike {
  private readonly listeners = new Set<MessageListener>()
  private readonly receivedL: Float32Array[] = []
  private readonly receivedR: Float32Array[] = []
  private readonly deferResults: boolean

  public constructor(options: { deferResults: boolean }) {
    this.deferResults = options.deferResults
  }

  public addEventListener(type: 'message', listener: MessageListener): void {
    if (type !== 'message') return
    this.listeners.add(listener)
  }

  public removeEventListener(type: 'message', listener: MessageListener): void {
    if (type !== 'message') return
    this.listeners.delete(listener)
  }

  public postMessage(message: unknown, _transfer?: Transferable[]): void {
    const msg = message as IncomingMessage
    if (msg.type === 'stream_start') {
      const started: OutgoingMessage = {
        type: 'stream_started',
        data: { nfft: 1, hop: 1, chunkSize: 4, segStep: 3 }
      }
      this.emit(started)
      return
    }
    if (msg.type === 'stream_data') {
      this.receivedL.push(msg.data.chL)
      this.receivedR.push(msg.data.chR)
      return
    }
    if (msg.type === 'stream_end') {
      const emitAll = () => {
        const allL = concatFloat32Arrays(this.receivedL)
        const allR = concatFloat32Arrays(this.receivedR)

        const cut = Math.min(3, allL.length)
        if (cut > 0) {
          this.emit({ type: 'stream_result', data: { chL: allL.slice(0, cut), chR: allR.slice(0, cut) } })
        }
        if (allL.length > cut) {
          this.emit({ type: 'stream_result', data: { chL: allL.slice(cut), chR: allR.slice(cut) } })
        }
        this.emit({ type: 'stream_ended' })
      }

      if (this.deferResults) {
        queueMicrotask(emitAll)
      } else {
        emitAll()
      }
    }
  }

  public terminate(): void {
    this.listeners.clear()
  }

  private emit(message: OutgoingMessage): void {
    const event = { data: message } as MessageEvent
    for (const listener of this.listeners) {
      listener(event)
    }
  }
}

test('多 worker 分段合并保持顺序与边界', async () => {
  const sampleCount = 20
  const chL = new Float32Array(sampleCount)
  const chR = new Float32Array(sampleCount)
  for (let i = 0; i < sampleCount; i++) {
    chL[i] = i
    chR[i] = 1000 + i
  }

  const worker0 = new FakeWorker({ deferResults: true })
  const worker1 = new FakeWorker({ deferResults: false })

  const uvr = new UVR({ modelUrl: 'model.onnx', workerUrl: 'worker.js', workerCount: 2 })
  ;(uvr as unknown as { workers: Worker[] }).workers = [
    worker0 as unknown as Worker,
    worker1 as unknown as Worker
  ]

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
    expect(out[i * 2]).toBe(chL[i])
    expect(out[i * 2 + 1]).toBe(chR[i])
  }
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
