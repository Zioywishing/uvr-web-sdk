import { defineConfig } from 'tsdown'
import fs from 'node:fs'

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm'],
  target: 'esnext',
  platform: 'browser',
  dts: true,
  plugins: [
    {
      name: 'wasm-base64',
      load(id) {
        if (id.endsWith('.wasm')) {
          const buffer = fs.readFileSync(id)
          const base64 = buffer.toString('base64')
          return `export default "${base64}"`
        }
      }
    }
  ]
})
