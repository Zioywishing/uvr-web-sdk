import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    worker: 'src/worker.ts',
  },
  format: ['esm'],
  target: 'esnext',
  platform: 'browser',
  dts: true,
  // Ensure we bundle our internal packages
  noExternal: ['@uar/fft', '@uar/ort-runtime-webgpu'],
})
