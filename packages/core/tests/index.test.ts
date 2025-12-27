import { expect, test } from 'vitest'
import { UAR } from '../src'

test('UAR instantiation', () => {
  const uar = new UAR({ modelUrl: 'model.onnx', workerUrl: 'worker.js' });
  expect(uar).toBeDefined();
})
