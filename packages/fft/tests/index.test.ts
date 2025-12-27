import { expect, test } from 'vitest'
import { rfft, irfft } from '../src'

test('FFT/IFFT exports', () => {
  expect(rfft).toBeDefined();
  expect(irfft).toBeDefined();
})
