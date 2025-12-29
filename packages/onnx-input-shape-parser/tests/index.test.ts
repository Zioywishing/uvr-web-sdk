import { describe, it, expect } from 'bun:test'
import { parseOnnxInputShapes } from '../src'
import { join } from 'node:path'
import { readdirSync, readFileSync } from 'node:fs'

const MODELS_DIR = join(import.meta.dir, '../../../demo-html/public/models')

describe('ONNX Input Shape Parser', () => {
  it('should parse all models in demo-html/public/models', () => {
    const files = readdirSync(MODELS_DIR).filter(f => f.endsWith('.onnx'))

    expect(files.length).toBeGreaterThan(0)

    for (const file of files) {
      const filePath = join(MODELS_DIR, file)
      const buffer = readFileSync(filePath)
      const inputs = parseOnnxInputShapes(new Uint8Array(buffer))

      console.log(`\n模型: ${file}`)
      expect(inputs.length).toBeGreaterThan(0)
      
      inputs.forEach(input => {
        console.log(`  输入名称: ${input.name}`)
        console.log(`  输入形状: [${input.shape.join(', ')}]`)
        expect(input.name).toBeDefined()
        expect(input.shape).toBeInstanceOf(Array)
      })
    }
  })
})
