export class UAR {
  options: {
    modelUrl: string;
    workerUrl: string;
    provider?: 'wasm' | 'webgpu';
  };

  private status: 'UNINITIALIZED' | 'FREE' | 'BUSY' = 'UNINITIALIZED';
  private worker: Worker | null = null;
  private initPromise: Promise<void> | null = null;

  private resolveStreamStart: (() => void) | null = null;
  private rejectStreamStart: ((err: Error) => void) | null = null;

  public constructor(options: {
    modelUrl: string;
    workerUrl: string;
    provider?: 'wasm' | 'webgpu';
  }) {
    this.options = options;
  }

  private async init() {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = (async () => {
      if (this.worker) {
        return;
      }

      console.log('[UAR] 正在创建 Worker...', this.options.workerUrl);
      this.worker = new Worker(this.options.workerUrl, { type: 'module' });

      // 等待 Worker 就绪
      await new Promise<void>((resolve, reject) => {
        const initHandler = (e: MessageEvent) => {
          const { type, error } = e.data;
          // console.log('[UAR] 收到 Worker 初始消息:', type);
          if (type === 'worker_ready') {
            console.log('[UAR] Worker 就绪');
            // 移除临时的 initHandler，后面会设置通用的 handler
            this.worker?.removeEventListener('message', initHandler);
            resolve();
          } else if (type === 'error') {
             this.worker?.removeEventListener('message', initHandler);
             reject(new Error(error || 'Worker init error'));
          }
        };
        this.worker!.addEventListener('message', initHandler);
      });

      // 设置通用的握手消息处理器
      this.worker.addEventListener('message', (e) => {
        const { type, error } = e.data;
        if (type === 'stream_started') {
          if (this.resolveStreamStart) {
            this.resolveStreamStart();
            this.resolveStreamStart = null;
            this.rejectStreamStart = null;
          }
        } else if (type === 'error') {
          const err = new Error(error || 'Unknown worker error');
          if (this.rejectStreamStart) {
            this.rejectStreamStart(err);
            this.resolveStreamStart = null;
            this.rejectStreamStart = null;
          } else {
            console.error('[UAR] Worker 错误:', err);
          }
        }
      });

      this.status = 'FREE';
    })();

    return this.initPromise;
  }

  public process(AudioData: ArrayBuffer): ReadableStream<Float32Array> {
    if (this.status === 'BUSY') {
      throw new Error('UAR is busy');
    }
    console.log('[UAR] 开始处理音频数据, 大小:', AudioData.byteLength);
    this.status = 'BUSY';

    let controller: ReadableStreamDefaultController<Float32Array>;
    let audioCtx: AudioContext | null = null;

    return new ReadableStream<Float32Array>({
      start: async (c) => {
        controller = c;
        try {
          if (!this.worker) {
            await this.init();
          }

          audioCtx = new AudioContext();
          console.log('[UAR] 解码音频数据...');
          const audioBuffer = await audioCtx.decodeAudioData(AudioData.slice(0));
          console.log('[UAR] 音频解码完成, 时长:', audioBuffer.duration, '采样率:', audioBuffer.sampleRate);

          const chL = audioBuffer.getChannelData(0);
          const chR = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : chL;

          // Handshake
          console.log('[UAR] 发送 stream_start 到 Worker, provider:', this.options.provider || 'wasm');
          await new Promise<void>((resolve, reject) => {
            this.resolveStreamStart = resolve;
            this.rejectStreamStart = reject;
            this.worker!.postMessage({
              type: 'stream_start',
              data: {
                modelUrl: this.options.modelUrl,
                provider: this.options.provider,
                sampleRate: audioBuffer.sampleRate
              }
            });
          });

          // 设置流式数据的消息处理器
          const streamHandler = (e: MessageEvent) => {
            const { type, data, error } = e.data;
            if (type === 'stream_result') {
              const { chL, chR } = data;
              // Interleave to stereo Float32Array
              const len = chL.length;
              const interleaved = new Float32Array(len * 2);
              for (let i = 0; i < len; i++) {
                interleaved[i * 2] = chL[i];
                interleaved[i * 2 + 1] = chR[i];
              }
              controller.enqueue(interleaved);
            } else if (type === 'stream_ended') {
              console.log('[UAR] 流处理结束');
              this.worker?.removeEventListener('message', streamHandler);
              controller.close();
              this.status = 'FREE';
              if (audioCtx && audioCtx.state !== 'closed') audioCtx.close();
            } else if (type === 'error') {
              console.error('[UAR] Worker 运行时错误:', error);
              this.worker?.removeEventListener('message', streamHandler);
              controller.error(new Error(error));
              this.status = 'FREE';
              if (audioCtx && audioCtx.state !== 'closed') audioCtx.close();
            }
          };
          this.worker!.addEventListener('message', streamHandler);

          // Send data in chunks
          const chunkSize = 16384; // 较小的 chunk size 以便更流畅地流式处理
          let offset = 0;
          console.log('[UAR] 开始发送音频 chunk 到 Worker...');
          while (offset < chL.length) {
            const end = Math.min(offset + chunkSize, chL.length);
            const subL = chL.slice(offset, end);
            const subR = chR.slice(offset, end);

            this.worker!.postMessage({
              type: 'stream_data',
              data: { chL: subL, chR: subR }
            }, [subL.buffer, subR.buffer]);

            offset = end;
          }

          console.log('[UAR] 所有数据已发送, 发送 stream_end');
          this.worker!.postMessage({ type: 'stream_end' });
        } catch (err: unknown) {
          const error = err instanceof Error ? err : new Error(String(err));
          console.error('[UAR] 处理过程出错:', error);
          controller.error(error);
          this.status = 'FREE';
          if (audioCtx && audioCtx.state !== 'closed') audioCtx.close();
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
      throw new Error('UAR is busy');
    }
    console.log('[UAR] 开始处理声道数据, 长度:', chL.length, '采样率:', sampleRate);
    this.status = 'BUSY';

    let controller: ReadableStreamDefaultController<Float32Array>;

    return new ReadableStream<Float32Array>({
      start: async (c) => {
        controller = c;
        try {
          if (!this.worker) {
            await this.init();
          }

          // Handshake
          console.log('[UAR] 发送 stream_start 到 Worker, provider:', this.options.provider || 'wasm');
          await new Promise<void>((resolve, reject) => {
            this.resolveStreamStart = resolve;
            this.rejectStreamStart = reject;
            this.worker!.postMessage({
              type: 'stream_start',
              data: {
                modelUrl: this.options.modelUrl,
                provider: this.options.provider,
                sampleRate: sampleRate
              }
            });
          });

          // Override onmessage for streaming data
          this.worker!.onmessage = (e) => {
            const { type, data, error } = e.data;
            if (type === 'stream_result') {
              const { chL: resL, chR: resR } = data;
              const len = resL.length;
              
              // 检查数据是否有振幅
              let maxAmp = 0;
              for (let i = 0; i < len; i++) {
                  maxAmp = Math.max(maxAmp, Math.abs(resL[i]), Math.abs(resR[i]));
              }
              if (maxAmp === 0) {
                  // console.warn('[UAR] 收到全零音频块');
              } else {
                  // console.log('[UAR] 收到有效音频块, 最大振幅:', maxAmp.toFixed(6));
              }

              const interleaved = new Float32Array(len * 2);
              for (let i = 0; i < len; i++) {
                interleaved[i * 2] = resL[i];
                interleaved[i * 2 + 1] = resR[i];
              }
              controller.enqueue(interleaved);
            } else if (type === 'stream_ended') {
              console.log('[UAR] 流处理结束');
              controller.close();
              this.status = 'FREE';
            } else if (type === 'error') {
              console.error('[UAR] Worker 运行时错误:', error);
              controller.error(new Error(error));
              this.status = 'FREE';
            }
          };

          // Send data in chunks
          const chunkSize = 16384;
          let offset = 0;
          while (offset < chL.length) {
            const end = Math.min(offset + chunkSize, chL.length);
            const subL = chL.slice(offset, end);
            const subR = chR.slice(offset, end);

            this.worker!.postMessage({
              type: 'stream_data',
              data: { chL: subL, chR: subR }
            }, [subL.buffer, subR.buffer]);

            offset = end;
          }

          console.log('[UAR] 所有数据已发送, 发送 stream_end');
          this.worker!.postMessage({ type: 'stream_end' });
        } catch (err: unknown) {
          const error = err instanceof Error ? err : new Error(String(err));
          console.error('[UAR] 处理过程出错:', error);
          controller.error(error);
          this.status = 'FREE';
        }
      }
    });
  }
}

export default UAR;
