import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    worker: 'src/worker.ts',
    'worker-fft': 'src/worker-fft.ts',
    'worker-ort': 'src/worker-ort.ts',
    'worker-ifft': 'src/worker-ifft.ts',
  },
  format: ['esm'],
  target: 'esnext',
  platform: 'browser',
  dts: true,
  // Ensure we bundle our internal packages
  noExternal: ['@uar/fft', '@uar/ort-runtime-webgpu'],
})
