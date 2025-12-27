<script setup lang="ts">
import { ref, computed } from 'vue';
import { UAR } from 'uar-web-sdk';
import workerUrl from 'uar-web-sdk/worker?worker&url';

// 内置模型列表
const availableModels = [
  'UVR-MDX-NET-Inst_HQ_3.onnx',
  'UVR-MDX-NET-Inst_3.onnx',
  'UVR_MDXNET_KARA_2.onnx',
  'UVR_MDXNET_KARA.onnx',
  'UVR_MDXNET_3_9662.onnx',
  'Kim_Inst.onnx'
];

const selectedModel = ref(availableModels[0]);
const selectedProvider = ref<'wasm' | 'webgpu'>('wasm');
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

let uar: UAR | null = null;
let audioCtx: AudioContext | null = null;

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (target.files && target.files.length > 0) {
    audioFile.value = target.files[0];
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

  isProcessing.value = true;
  status.value = '初始化中...';

  try {
    // 每次开始前重新初始化 UAR 以确保状态清洁，或者复用实例但需要处理状态
    // 这里为了演示简单，如果已存在则复用，但在实际应用中可能需要更复杂的生命周期管理
    if (!uar) {
      uar = new UAR({
        modelUrl: modelUrl.value,
        workerUrl: workerUrl,
        provider: selectedProvider.value
      });
    } else {
        // 如果支持更换模型，可能需要重新创建 UAR 实例或调用特定方法
        // 目前 UAR 实现似乎不支持动态更换模型，所以我们这里重新创建
        // 注意：实际生产中应正确销毁旧实例（终止 worker 等）
        // 由于 UAR 类目前没有 destroy 方法，我们暂且假设用户刷新页面或我们重新 new 一个
        // 为了安全起见，这里我们简单的覆盖旧实例
        uar = new UAR({
            modelUrl: modelUrl.value,
            workerUrl: workerUrl,
            provider: selectedProvider.value
        });
    }

    status.value = '读取音频...';
    const arrayBuffer = await audioFile.value.arrayBuffer();

    status.value = '处理中...';
    const stream = uar.process(arrayBuffer);

    status.value = '正在播放...';
    await playStream(stream);

    status.value = '完成';
  } catch (err: unknown) {
    console.error(err);
    const errorMessage = err instanceof Error ? err.message : String(err);
    status.value = `出错 - ${errorMessage}`;
  } finally {
    isProcessing.value = false;
  }
};

const playStream = async (stream: ReadableStream<Float32Array>) => {
  if (!audioCtx) {
    audioCtx = new AudioContext();
  }
  if (audioCtx.state === 'suspended') {
    await audioCtx.resume();
  }

  let nextStartTime = audioCtx.currentTime;
  const reader = stream.getReader();

  try {
    let chunkCount = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log('[Demo] 流读取完成, 总 chunk 数:', chunkCount);
        break;
      }

      if (value) {
        chunkCount++;
        // console.log('[Demo] 收到音频数据 chunk, 长度:', value.length);
        // 假设输出是立体声 (interleaved)
        const frameCount = value.length / 2;
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
</script>

<template>
  <div class="container">
    <h1>UAR Web SDK Demo (Vue)</h1>
    
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
        <label>
          <input type="checkbox" v-model="useCustomUrl" /> 使用自定义模型 URL
        </label>
      </div>

      <div class="form-group" v-if="useCustomUrl">
        <input type="text" v-model="customModelUrl" placeholder="输入 .onnx 模型地址" />
      </div>
      
      <div class="form-group">
        <label>选择音频:</label>
        <input type="file" @change="handleFileChange" accept="audio/*" />
      </div>

      <div class="actions">
        <button @click="startProcessing" :disabled="isProcessing">
          {{ isProcessing ? '处理中...' : '开始处理' }}
        </button>
      </div>
    </div>

    <div class="card status-card">
      <p>状态: {{ status }}</p>
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
  margin: 1em 0;
  border: 1px solid #eee;
  border-radius: 8px;
  background-color: #f9f9f9;
}

.status-card {
  background-color: #eef;
  font-weight: bold;
}

.form-group {
  margin-bottom: 1em;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5em;
}

select, input[type="text"] {
  padding: 0.5em;
  width: 300px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

button {
  border-radius: 8px;
  border: 1px solid transparent;
  padding: 0.6em 1.2em;
  font-size: 1em;
  font-weight: 500;
  font-family: inherit;
  background-color: #1a1a1a;
  color: white;
  cursor: pointer;
  transition: border-color 0.25s;
}

button:hover:not(:disabled) {
  border-color: #646cff;
  background-color: #2a2a2a;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
