<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { UAR } from 'uar-web-sdk';
import fftWorkerUrl from 'uar-web-sdk/worker-fft?worker&url';
import ortWorkerUrl from 'uar-web-sdk/worker-ort?worker&url';
import ifftWorkerUrl from 'uar-web-sdk/worker-ifft?worker&url';

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

let uar: UAR | null = null;
let audioCtx: AudioContext | null = null;

onMounted(async () => {
  console.log('[Demo] 页面挂载，开始自动初始化...');
  try {
    uar = new UAR({
      modelUrl: modelUrl.value,
      fftWorkerUrl,
      ortWorkerUrl,
      ifftWorkerUrl,
      provider: selectedProvider.value,
      workerCount: workerCount.value
    });
    status.value = '正在预初始化...';
    await uar.init();
    status.value = '就绪 (已初始化)';
    console.log('[Demo] 自动初始化完成');
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

  try {
    // 如果 uar 已经初始化且模型 URL 没变，可以直接使用
    // 这里为了简单，如果实例不存在则创建，如果存在则检查配置
    if (!uar) {
      uar = new UAR({
        modelUrl: modelUrl.value,
        fftWorkerUrl,
        ortWorkerUrl,
        ifftWorkerUrl,
        provider: selectedProvider.value,
        workerCount: workerCount.value
      });
      await uar.init();
    } else {
      // 检查配置是否变化
      if (
        uar.options.modelUrl !== modelUrl.value || 
        uar.options.provider !== selectedProvider.value ||
        uar.options.workerCount !== workerCount.value
      ) {
        console.log('[Demo] 检测到配置变化，重新创建实例');
        uar.destroy(); // 销毁旧实例
        uar = new UAR({
          modelUrl: modelUrl.value,
          fftWorkerUrl,
          ortWorkerUrl,
          ifftWorkerUrl,
          provider: selectedProvider.value,
          workerCount: workerCount.value
        });
        await uar.init();
      }
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
    audioCtx = new AudioContext({
      sampleRate: 44100
    });
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

const handleDestroy = () => {
  if (uar) {
    uar.destroy();
    uar = null;
    status.value = '实例已销毁';
    console.log('[Demo] 实例已销毁');
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
        <input type="file" @change="handleFileChange" accept="audio/*" />
      </div>

      <div class="actions">
        <button @click="startProcessing" :disabled="isProcessing">
          {{ isProcessing ? '处理中...' : '开始处理' }}
        </button>
        <button @click="handleDestroy" :disabled="isProcessing" class="destroy-btn">
          销毁实例
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

select, input[type="text"], input[type="number"] {
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

.destroy-btn {
  background-color: #d32f2f;
  margin-left: 10px;
}

.destroy-btn:hover:not(:disabled) {
  background-color: #b71c1c;
  border-color: #ef5350;
}
</style>
