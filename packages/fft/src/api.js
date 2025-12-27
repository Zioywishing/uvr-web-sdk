import { ComplexArray, RealArray } from "./types.js";
import { FFTConfig } from "./fft.js";
import { FFTNDConfig } from "./fftnd.js";
import { RealFFTConfig, InverseRealFFTConfig } from "./rfft.js";
import { RealFFTNDConfig, InverseRealFFTNDConfig } from "./rfftnd.js";
function fftInner(config, inputType, outputType, input) {
    const inputArray = inputType.fromDataArray(input);
    const outputArray = new outputType(inputArray.length);
    config.work(inputArray, outputArray);
    const output = outputArray.toFloat32Array();
    for (const v of [config, inputArray, outputArray]) {
        v.free();
    }
    return output;
}
function makeDims(dims, args) {
    return typeof dims === "number" ? Array.from({
        length: args.length - 1
    }).map((_, i) => args[i + 1]) : dims;
}
// Forward 1D FFT
export function fft(input) {
    return fftInner(new FFTConfig(input.length / 2, false), ComplexArray, ComplexArray, input);
}
// Inverse 1D FFT
export function ifft(input) {
    return fftInner(new FFTConfig(input.length / 2, true), ComplexArray, ComplexArray, input);
}
// Forward 2D FFT
export function fft2d(input, n, m) {
    return fftnd(input, n, m);
}
export function fftnd(input, dims) {
    const intDims = makeDims(dims, arguments);
    return fftInner(new FFTNDConfig(intDims, false), ComplexArray, ComplexArray, input);
}
// Inverse 2D FFT
export function ifft2d(input, n, m) {
    return ifftnd(input, n, m);
}
export function ifftnd(input, dims) {
    const intDims = makeDims(dims, arguments);
    return fftInner(new FFTNDConfig(intDims, true), ComplexArray, ComplexArray, input);
}
// Forward Real 1D FFT
export function rfft(input) {
    return fftInner(new RealFFTConfig(input.length), RealArray, ComplexArray, input);
}
// Inverse Real 1D FFT
export function irfft(input) {
    return fftInner(new InverseRealFFTConfig(input.length / 2), ComplexArray, RealArray, input);
}
// Forward Real 2D FFT
export function rfft2d(input, n, m) {
    return rfftnd(input, n, m);
}
export function rfftnd(input, dims) {
    const intDims = makeDims(dims, arguments);
    return fftInner(new RealFFTNDConfig(intDims), RealArray, ComplexArray, input);
}
// Inverse Real 2D FFT
export function irfft2d(input, n, m) {
    return fftInner(new InverseRealFFTNDConfig([n, m]), ComplexArray, RealArray, input);
}
export function irfftnd(input, dims) {
    const intDims = makeDims(dims, arguments);
    return fftInner(new InverseRealFFTNDConfig(intDims), ComplexArray, RealArray, input);
}
//# sourceMappingURL=api.js.map