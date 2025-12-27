/// <reference path="./types.d.ts" />
import * as ortModule from './ort.mjs';
import ortWasmBase64 from './ort-wasm-simd-threaded.wasm';
import ortWasmJsepBase64 from './ort-wasm-simd-threaded.jsep.wasm';
import ortWasmJsepMjsContent from './ort-wasm-simd-threaded.jsep.mjs.js';

function base64ToBlob(base64: string, type: string): Blob {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return new Blob([bytes], { type });
}

// Create URLs
const ortWasmUrl = URL.createObjectURL(base64ToBlob(ortWasmBase64, 'application/wasm'));
const ortWasmJsepUrl = URL.createObjectURL(base64ToBlob(ortWasmJsepBase64, 'application/wasm'));
const ortWasmJsepMjsUrl = URL.createObjectURL(new Blob([ortWasmJsepMjsContent], { type: 'application/javascript' }));

// Configure
if (ortModule.env && ortModule.env.wasm) {
    ortModule.env.wasm.wasmPaths = {
        'mjs': ortWasmJsepMjsUrl,
        'wasm': ortWasmJsepUrl,
        'ort-wasm-simd-threaded.wasm': ortWasmUrl,
        'ort-wasm-simd-threaded.jsep.wasm': ortWasmJsepUrl,
        'ort-wasm-simd-threaded.jsep.mjs': ortWasmJsepMjsUrl
    };
    ortModule.env.wasm.numThreads = 1;
}

export interface Tensor {
    data: Float32Array;
    dims: number[];
    type: string;
}

export interface InferenceSession {
    inputNames: string[];
    run(inputs: Record<string, Tensor>): Promise<Record<string, Tensor>>;
}

// Type definitions for the ort module structure
interface OrtModule {
    env: {
        wasm: {
            wasmPaths: Record<string, string>;
            numThreads: number;
        };
        webgpu: {
            device: any;
        } | null;
    };
    InferenceSession: {
        create(path: string | Uint8Array, options?: any): Promise<InferenceSession>;
    };
    Tensor: new (type: string, data: Float32Array, dims: number[]) => Tensor;
}

// Cast the imported module to the typed interface
export const ort = ortModule as unknown as OrtModule;
export default ort;
