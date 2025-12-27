declare module '*/kissfft.wasm' {
    const content: string;
    export default content;
}

declare module '*/kissfft.mjs' {
    const factory: (moduleOverrides?: any) => Promise<any>;
    export default factory;
}

declare module '*/api.js' {
    export function fft(input: Float32Array): Float32Array;
    export function ifft(input: Float32Array): Float32Array;
    export function fft2d(input: Float32Array, n: number, m: number): Float32Array;
    export function fftnd(input: Float32Array, dims: number | number[]): Float32Array;
    export function ifft2d(input: Float32Array, n: number, m: number): Float32Array;
    export function ifftnd(input: Float32Array, dims: number | number[]): Float32Array;
    export function rfft(input: Float32Array): Float32Array;
    export function irfft(input: Float32Array): Float32Array;
    export function rfft2d(input: Float32Array, n: number, m: number): Float32Array;
    export function rfftnd(input: Float32Array, dims: number | number[]): Float32Array;
    export function irfft2d(input: Float32Array, n: number, m: number): Float32Array;
    export function irfftnd(input: Float32Array, dims: number | number[]): Float32Array;
}
