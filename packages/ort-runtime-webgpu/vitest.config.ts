import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
  },
  plugins: [
    {
      name: 'wasm-import',
      enforce: 'pre',
      resolveId(id) {
        if (id.endsWith('.wasm')) {
          return id;
        }
      },
      load(id) {
        if (id.endsWith('.wasm')) {
          return 'export default "base64mock"';
        }
      }
    }
  ]
});
