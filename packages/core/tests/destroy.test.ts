import { expect, test, vi } from 'vitest';
import { UVR } from '../src';

// 模拟 Worker
class MockWorker {
  onmessage: ((ev: MessageEvent) => void) | null = null;
  private listeners = new Set<(ev: MessageEvent) => void>();
  
  constructor(public url: string) {
    // 模拟异步就绪
    setTimeout(() => {
      this.emit({ type: 'worker_ready' });
    }, 10);
  }

  addEventListener(type: string, listener: any) {
    if (type === 'message') this.listeners.add(listener);
  }
  removeEventListener(type: string, listener: any) {
    if (type === 'message') this.listeners.delete(listener);
  }
  postMessage(msg: any) {
    // 模拟处理 init/preload
    if (msg.type === 'init') {
      setTimeout(() => this.emit({ type: 'inited', jobId: msg.jobId }), 10);
    } else if (msg.type === 'preload') {
      setTimeout(() => this.emit({ type: 'preloaded', jobId: msg.jobId }), 10);
    } else if (msg.type === 'compute' || msg.type === 'run') {
        // 模拟计算
        setTimeout(() => {
            if (msg.type === 'compute') {
                this.emit({ type: 'computed', jobId: msg.jobId, data: { segOutL: new Float32Array(10), segOutR: new Float32Array(10) } });
            } else {
                this.emit({ type: 'ran', jobId: msg.jobId, data: new Float32Array(10) });
            }
        }, 20);
    }
  }
  terminate = vi.fn();
  
  private emit(data: any) {
    const ev = { data } as MessageEvent;
    if (this.onmessage) this.onmessage(ev);
    this.listeners.forEach(l => l(ev));
  }
}

// 模拟 OfflineAudioContext
class MockOfflineAudioContext {
  constructor() {}
  decodeAudioData = vi.fn().mockResolvedValue({
    duration: 1,
    sampleRate: 44100,
    numberOfChannels: 2,
    getChannelData: () => new Float32Array(44100)
  });
}

// 模拟全局环境
global.Worker = MockWorker as any;
global.OfflineAudioContext = MockOfflineAudioContext as any;
global.fetch = vi.fn().mockImplementation(() => 
  Promise.resolve({
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(100))
  })
);

// 模拟 parseOnnxInputShapes
vi.mock('@uvr-web-sdk/onnx-input-shape-parser', () => ({
  parseOnnxInputShapes: () => [{ name: 'input', shape: [1, 4, 3072, 256] }]
}));

test('destroy() 应该立即终止 init() 过程', async () => {
  const uvr = new UVR({
    modelUrl: 'http://test.com/model.onnx',
    fftWorkerUrl: 'fft.js',
    ortWorkerUrl: 'ort.js',
    ifftWorkerUrl: 'ifft.js',
    workerCount: 1
  });

  const initPromise = uvr.init();
  
  // 立即销毁
  uvr.destroy();
  
  // initPromise 应该被 reject
  await expect(initPromise).rejects.toThrow('操作已中止');
  
  // 状态应该重置为 UNINITIALIZED
  expect((uvr as any).status).toBe('UNINITIALIZED');
});

test('destroy() 应该清理所有正在运行的 Worker', async () => {
  const uvr = new UVR({
    modelUrl: 'http://test.com/model.onnx',
    fftWorkerUrl: 'fft.js',
    ortWorkerUrl: 'ort.js',
    ifftWorkerUrl: 'ifft.js',
    workerCount: 2
  });

  // 等待初始化完成一半时销毁
  const initPromise = uvr.init();
  
  // 模拟一些延迟
  await new Promise(r => setTimeout(r, 5));
  
  uvr.destroy();
  
  await expect(initPromise).rejects.toThrow();
  
  // 检查 terminate 是否被调用
  // 由于 init 是异步的，我们需要检查在销毁瞬间已经创建出来的 worker
  const fftWorkers = (uvr as any).fftWorkers;
  expect(fftWorkers.length).toBe(0); // 销毁后引用应被清空
});

test('在 process 运行期间 destroy() 应该停止处理', async () => {
    const uvr = new UVR({
        modelUrl: 'http://test.com/model.onnx',
        fftWorkerUrl: 'fft.js',
        ortWorkerUrl: 'ort.js',
        ifftWorkerUrl: 'ifft.js',
        workerCount: 1
    });

    // 注入一些模拟数据
    const audioData = new ArrayBuffer(1024);
    const stream = uvr.process(audioData);
    const reader = stream.getReader();

    // 启动处理
    const readPromise = reader.read();
    
    // 稍微等待一下让它开始
    await new Promise(r => setTimeout(r, 50));
    
    // 销毁
    uvr.destroy();
    
    // readPromise 可能会因为 controller.error 而被 reject，或者停止
    try {
        await readPromise;
    } catch (e) {
        expect((e as Error).message).toMatch(/已销毁|Abort|中止/);
    }

    expect((uvr as any).status).toBe('UNINITIALIZED');
});

test('destroy() 是幂等的，多次调用不会出错', () => {
    const uvr = new UVR({
        modelUrl: 'http://test.com/model.onnx',
        fftWorkerUrl: 'fft.js',
        ortWorkerUrl: 'ort.js',
        ifftWorkerUrl: 'ifft.js'
    });

    expect(() => {
        uvr.destroy();
        uvr.destroy();
        uvr.destroy();
    }).not.toThrow();
});
