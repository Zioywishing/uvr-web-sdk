declare module '*/ort-wasm-simd-threaded.wasm' {
    const content: string;
    export default content;
}

declare module '*/ort-wasm-simd-threaded.jsep.wasm' {
    const content: string;
    export default content;
}

declare module '*.mjs.js' {
    const content: string;
    export default content;
}

declare module '*.mjs' {
    const content: any;
    export default content;
    export const env: any;
    export const InferenceSession: any;
    export const Tensor: any;
}
