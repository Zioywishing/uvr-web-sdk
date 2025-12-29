<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { UVR } from '@uvr-web-sdk/core';
import fftWorkerUrl from '@uvr-web-sdk/core/worker-fft?worker&url';
import ortWorkerUrl from '@uvr-web-sdk/core/worker-ort?worker&url';
import ifftWorkerUrl from '@uvr-web-sdk/core/worker-ifft?worker&url';

// 内置模型列表
const availableModels = [
  'UVR-MDX-NET-Inst_HQ_3.onnx',
  'UVR-MDX-NET-Inst_HQ_3_int8.onnx',
  'UVR-MDX-NET-Inst_HQ_3_int4.onnx',
  'UVR-MDX-NET-Inst_HQ_3_fp16.onnx',
  'UVR-MDX-NET-Inst_3.onnx',
  'UVR_MDXNET_KARA_2.onnx',
  'UVR_MDXNET_KARA.onnx',
  'UVR_MDXNET_3_9662.onnx',
  'Kim_Inst.onnx'
];

const selectedModel = ref(availableModels[0]);
const selectedProvider = ref<'wasm' | 'webgpu'>('webgpu');
const workerCount = ref(3);
const customModelUrl = ref('');
const useCustomUrl = ref(false);

const modelUrl = computed(() => {
  if (useCustomUrl.value) {
    return customModelUrl.value;
  }
  return `/models/${selectedModel.value}`;
});

const status = ref('就绪');
const audioFile = ref<File | null>(null);
const isProcessing = ref(false);
const progress = ref(0);
const metrics = ref({
  initTime: 0,
  firstFrameLatency: 0,
  totalProcessingTime: 0
});
const processedChunks = ref<Float32Array[]>([]);
const canDownload = ref(false);

let uvr: UVR | null = null;
let audioCtx: AudioContext | null = null;

onMounted(async () => {
  console.log('[Demo] 页面挂载，开始自动初始化...');
  const start = performance.now();
  try {
    uvr = new UVR({
      modelUrl: modelUrl.value,
      fftWorkerUrl,
      ortWorkerUrl,
      ifftWorkerUrl,
      provider: selectedProvider.value,
      workerCount: workerCount.value
    });
    status.value = '正在预初始化...';
    await uvr.init();
    metrics.value.initTime = performance.now() - start;
    status.value = '就绪 (已初始化)';
    console.log('[Demo] 自动初始化完成, 耗时:', metrics.value.initTime.toFixed(2), 'ms');
  } catch (err: unknown) {
    console.error('[Demo] 自动初始化失败:', err);
    const errorMessage = err instanceof Error ? err.message : String(err);
    status.value = `初始化失败 - ${errorMessage}`;
  }
});

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (target.files && target.files.length > 0) {
    audioFile.value = target.files[0];
    canDownload.value = false;
  }
};

const useTestAudio = async () => {
  try {
    status.value = '正在加载测试音频...';
    const response = await fetch('/test.mp3');
    const blob = await response.blob();
    const file = new File([blob], 'test.mp3', { type: 'audio/mpeg' });
    audioFile.value = file;
    status.value = '测试音频加载完成';
    canDownload.value = false;
  } catch (err) {
    console.error('加载测试音频失败:', err);
    status.value = '加载测试音频失败';
  }
};

const startProcessing = async () => {
  if (!audioFile.value) {
    alert('请选择音频文件');
    return;
  }
  if (!modelUrl.value) {
    alert('请输入模型 URL');
    return;
  }

  // 验证 workerCount
  if (workerCount.value < 1) {
    alert('Worker 数量至少为 1');
    return;
  }

  isProcessing.value = true;
  status.value = '处理中...';
  progress.value = 0;
  processedChunks.value = [];
  canDownload.value = false;
  metrics.value.firstFrameLatency = 0;
  metrics.value.totalProcessingTime = 0;

  const processStart = performance.now();

  try {
    // 如果 uvr 已经初始化且模型 URL 没变，可以直接使用
    // 这里为了简单，如果实例不存在则创建，如果存在则检查配置
    if (!uvr) {
      const initStart = performance.now();
      uvr = new UVR({
        modelUrl: modelUrl.value,
        fftWorkerUrl,
        ortWorkerUrl,
        ifftWorkerUrl,
        provider: selectedProvider.value,
        workerCount: workerCount.value
      });
      await uvr.init();
      metrics.value.initTime = performance.now() - initStart;
    } else {
      // 检查配置是否变化
      if (
        uvr.options.modelUrl !== modelUrl.value || 
        uvr.options.provider !== selectedProvider.value ||
        uvr.options.workerCount !== workerCount.value
      ) {
        console.log('[Demo] 检测到配置变化，重新创建实例');
        uvr.destroy(); // 销毁旧实例
        const initStart = performance.now();
        uvr = new UVR({
          modelUrl: modelUrl.value,
          fftWorkerUrl,
          ortWorkerUrl,
          ifftWorkerUrl,
          provider: selectedProvider.value,
          workerCount: workerCount.value
        });
        await uvr.init();
        metrics.value.initTime = performance.now() - initStart;
      }
    }

    status.value = '读取音频...';
    const arrayBuffer = await audioFile.value.arrayBuffer();

    status.value = '分离处理中...';
    const stream = uvr.process(arrayBuffer);

    // 估算总长度以便计算进度
    if (!audioCtx) {
      audioCtx = new AudioContext({ sampleRate: 44100 });
    }
    const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
    const totalSamples = decoded.length;

    await playStream(stream, totalSamples, processStart);

    metrics.value.totalProcessingTime = performance.now() - processStart;
    status.value = '完成';
    canDownload.value = true;
  } catch (err: unknown) {
    console.error(err);
    const errorMessage = err instanceof Error ? err.message : String(err);
    status.value = `出错 - ${errorMessage}`;
  } finally {
    isProcessing.value = false;
  }
};

const playStream = async (stream: ReadableStream<Float32Array>, totalSamples: number, processStartTime: number) => {
  if (!audioCtx) {
    audioCtx = new AudioContext({
      sampleRate: 44100
    });
  }
  if (audioCtx.state === 'suspended') {
    await audioCtx.resume();
  }

  let nextStartTime = audioCtx.currentTime;
  const reader = stream.getReader();
  let receivedSamples = 0;

  try {
    let chunkCount = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log('[Demo] 流读取完成, 总 chunk 数:', chunkCount);
        progress.value = 100;
        break;
      }

      if (value) {
        if (chunkCount === 0) {
          metrics.value.firstFrameLatency = performance.now() - processStartTime;
        }
        chunkCount++;
        processedChunks.value.push(new Float32Array(value));
        
        // 假设输出是立体声 (interleaved)
        const frameCount = value.length / 2;
        receivedSamples += frameCount;
        progress.value = Math.min(99, Math.floor((receivedSamples / totalSamples) * 100));

        const buffer = audioCtx.createBuffer(2, frameCount, audioCtx.sampleRate);
        const chL = buffer.getChannelData(0);
        const chR = buffer.getChannelData(1);

        for (let i = 0; i < frameCount; i++) {
          chL[i] = value[i * 2];
          chR[i] = value[i * 2 + 1];
        }

        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(audioCtx.destination);

        if (nextStartTime < audioCtx.currentTime) {
          nextStartTime = audioCtx.currentTime + 0.05;
        }
        source.start(nextStartTime);
        nextStartTime += buffer.duration;
      }
    }
  } finally {
    reader.releaseLock();
  }
};

const downloadResult = () => {
  if (processedChunks.value.length === 0) return;

  const totalLength = processedChunks.value.reduce((acc, chunk) => acc + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of processedChunks.value) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  const wavBlob = encodeWAV(merged, 44100);
  const url = URL.createObjectURL(wavBlob);
  const a = document.createElement('a');
  const originalName = audioFile.value?.name.split('.')[0] || 'audio';
  a.href = url;
  a.download = `${originalName}_sep.wav`;
  a.click();
  URL.revokeObjectURL(url);
};

function encodeWAV(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (view: DataView, offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

const handleDestroy = () => {
  if (uvr) {
    uvr.destroy();
    uvr = null;
    status.value = '实例已销毁';
    console.log('[Demo] 实例已销毁');
  }
};

const handleReinit = async () => {
  handleDestroy();
  const start = performance.now();
  try {
    uvr = new UVR({
      modelUrl: modelUrl.value,
      fftWorkerUrl,
      ortWorkerUrl,
      ifftWorkerUrl,
      provider: selectedProvider.value,
      workerCount: workerCount.value
    });
    status.value = '正在重新初始化...';
    await uvr.init();
    metrics.value.initTime = performance.now() - start;
    status.value = '就绪 (已重新初始化)';
    console.log('[Demo] 重新初始化完成, 耗时:', metrics.value.initTime.toFixed(2), 'ms');
  } catch (err: unknown) {
    console.error('[Demo] 重新初始化失败:', err);
    const errorMessage = err instanceof Error ? err.message : String(err);
    status.value = `重新初始化失败 - ${errorMessage}`;
  }
};
</script>

<template>
  <div class="container">
    <h1>UVR Web SDK Demo (Vue)</h1>
    
    <div class="card">
      <div class="form-group">
        <label>选择模型:</label>
        <select v-model="selectedModel" :disabled="useCustomUrl">
          <option v-for="model in availableModels" :key="model" :value="model">
            {{ model }}
          </option>
        </select>
      </div>

      <div class="form-group">
        <label>推理后端:</label>
        <select v-model="selectedProvider">
          <option value="wasm">WASM</option>
          <option value="webgpu">WebGPU</option>
        </select>
      </div>

      <div class="form-group">
        <label>Worker 数量:</label>
        <input type="number" v-model.number="workerCount" min="1" max="16" />
      </div>

      <div class="form-group">
        <label>
          <input type="checkbox" v-model="useCustomUrl" /> 使用自定义模型 URL
        </label>
      </div>

      <div class="form-group" v-if="useCustomUrl">
        <input type="text" v-model="customModelUrl" placeholder="输入 .onnx 模型地址" />
      </div>
      
      <div class="form-group">
        <label>选择音频:</label>
        <div class="file-input-wrapper">
          <input type="file" @change="handleFileChange" accept="audio/*" />
          <button @click="useTestAudio" :disabled="isProcessing" class="secondary-btn">
            使用测试音频 (test.mp3)
          </button>
        </div>
        <p v-if="audioFile" class="file-info">当前音频: {{ audioFile.name }}</p>
      </div>

      <div class="actions">
        <button @click="startProcessing" :disabled="isProcessing" class="primary-btn">
          {{ isProcessing ? '处理中...' : '开始处理' }}
        </button>
        <button @click="downloadResult" :disabled="!canDownload || isProcessing" class="download-btn">
          下载分离音频
        </button>
        <button @click="handleReinit" :disabled="isProcessing" class="reinit-btn">
          重新初始化
        </button>
        <button @click="handleDestroy" :disabled="isProcessing" class="destroy-btn">
          销毁实例
        </button>
      </div>

      <div v-if="isProcessing || progress > 0" class="progress-container">
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: progress + '%' }"></div>
        </div>
        <div class="progress-text">{{ progress }}%</div>
      </div>
    </div>

    <div class="card metrics-card" v-if="metrics.initTime > 0">
      <h3>性能统计</h3>
      <div class="metrics-grid">
        <div class="metric-item">
          <span class="label">初始化耗时:</span>
          <span class="value">{{ metrics.initTime.toFixed(2) }} ms</span>
        </div>
        <div class="metric-item">
          <span class="label">首帧播放延迟:</span>
          <span class="value">{{ metrics.firstFrameLatency > 0 ? metrics.firstFrameLatency.toFixed(2) + ' ms' : '-' }}</span>
        </div>
        <div class="metric-item">
          <span class="label">分离完成总耗时:</span>
          <span class="value">{{ metrics.totalProcessingTime > 0 ? metrics.totalProcessingTime.toFixed(2) + ' ms' : '-' }}</span>
        </div>
      </div>
    </div>

    <div class="card status-card">
      <p>当前状态: {{ status }}</p>
    </div>
  </div>
</template>

<style scoped>
.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
  text-align: center;
}

.card {
  padding: 2em;
  margin: 1.5em 0;
  border: none;
  border-radius: 16px;
  background-color: #ffffff;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: transform 0.2s;
}

.card:hover {
  transform: translateY(-2px);
}

.status-card {
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-left: 5px solid #646cff;
}

.metrics-card {
  background-color: #f8f9fa;
  text-align: left;
}

.metrics-card h3 {
  margin-top: 0;
  color: #333;
  font-size: 1.2rem;
  border-bottom: 2px solid #eee;
  padding-bottom: 0.5rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-top: 1rem;
}

.metric-item {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.metric-item .label {
  font-size: 0.9rem;
  color: #666;
}

.metric-item .value {
  font-size: 1.1rem;
  font-weight: 600;
  color: #646cff;
}

.form-group {
  margin-bottom: 1.5rem;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.5rem;
  max-width: 500px;
  margin-left: auto;
  margin-right: auto;
}

.file-input-wrapper {
  display: flex;
  gap: 1rem;
  width: 100%;
  align-items: center;
}

select, input[type="text"], input[type="number"] {
  padding: 0.6rem 1rem;
  width: 100%;
  border: 1px solid #ddd;
  border-radius: 8px;
  font-size: 1rem;
  transition: border-color 0.2s;
}

select:focus, input:focus {
  outline: none;
  border-color: #646cff;
}

.actions {
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
  margin-top: 2rem;
}

button {
  border-radius: 10px;
  border: 1px solid transparent;
  padding: 0.8rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.primary-btn {
  background-color: #646cff;
  color: white;
}

.primary-btn:hover:not(:disabled) {
  background-color: #535bf2;
  box-shadow: 0 4px 12px rgba(100, 108, 255, 0.3);
}

.secondary-btn {
  background-color: #f1f3f5;
  color: #495057;
  border: 1px solid #dee2e6;
}

.secondary-btn:hover:not(:disabled) {
  background-color: #e9ecef;
}

.download-btn {
  background-color: #28a745;
  color: white;
}

.download-btn:hover:not(:disabled) {
  background-color: #218838;
  box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
}

.reinit-btn {
  background-color: #ffc107;
  color: #212529;
}

.reinit-btn:hover:not(:disabled) {
  background-color: #e0a800;
  box-shadow: 0 4px 12px rgba(255, 193, 7, 0.3);
}

.destroy-btn {
  background-color: #fff;
  color: #dc3545;
  border: 1px solid #dc3545;
}

.destroy-btn:hover:not(:disabled) {
  background-color: #dc3545;
  color: white;
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none !important;
}

.progress-container {
  margin-top: 2rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.progress-bar {
  flex: 1;
  height: 10px;
  background-color: #e9ecef;
  border-radius: 5px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #646cff, #a2a7ff);
  transition: width 0.3s ease;
}

.progress-text {
  font-weight: 600;
  color: #646cff;
  min-width: 3rem;
}

.file-info {
  margin-top: 0.5rem;
  font-size: 0.9rem;
  color: #666;
  font-style: italic;
}
</style>
