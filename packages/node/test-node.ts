import { UARNode } from './src/index';
import { join } from 'path';

async function runTest() {
  console.log('--- 开始 Node 版本自测 ---');

  // 使用 core 包里的测试模型
  const modelPath = join(process.cwd(), '../core/tests/UVR-MDX-NET-Inst_HQ_3.onnx');
  console.log('模型路径:', modelPath);

  const uar = new UARNode({
    modelUrl: modelPath,
    provider: 'wasm' // 或者 cpu, 取决于环境支持
  });

  // 生成正弦波
  const duration = 1; // 1秒
  const sampleRate = 44100;
  const frequency = 440; 
  const numSamples = duration * sampleRate;
  const chL = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    chL[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 0.5;
  }
  const chR = new Float32Array(chL); 

  console.log('音频生成完成, 样本数:', numSamples);

  try {
    console.log('初始化并开始处理...');
    const startTime = performance.now();
    
    // 分块处理模拟流式
    const chunkSize = 16384;
    let offset = 0;
    let totalOutSamples = 0;

    while (offset < numSamples) {
        const end = Math.min(offset + chunkSize, numSamples);
        const subL = chL.slice(offset, end);
        const subR = chR.slice(offset, end);
        
        const result = await uar.process(subL, subR);
        totalOutSamples += result.chL.length;
        
        if (result.chL.length > 0) {
            // console.log(`处理进度: ${offset}/${numSamples}, 输出: ${result.chL.length}`);
        }
        
        offset = end;
    }

    const flushResult = await uar.flush();
    totalOutSamples += flushResult.chL.length;
    
    const endTime = performance.now();
    console.log(`处理完成, 耗时: ${(endTime - startTime).toFixed(2)}ms`);
    console.log('总输入样本:', numSamples);
    console.log('总输出样本:', totalOutSamples);
    
    if (totalOutSamples > 0) {
        console.log('✅ 测试通过: 成功获得推理结果');
    } else {
        console.error('❌ 测试失败: 未获得输出样本');
        process.exit(1);
    }

  } catch (err) {
    console.error('❌ 测试出错:', err);
    process.exit(1);
  }
}

runTest();
