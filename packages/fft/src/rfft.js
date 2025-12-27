import { KissFFTConfig } from "./types.js";
import { wasm } from "./wasm.js";
export function checkRealFFT(nfft) {
    if (nfft % 2 === 1) {
        throw new Error("Real FFT optimization must be even.");
    }
}
class AbstractRealFFTConfig extends KissFFTConfig {
    ptr = 0;
    constructor(nfft, inverse) {
        super(nfft, inverse);
        checkRealFFT(nfft);
        this.ptr = wasm._kiss_fftr_alloc(nfft, inverse, 0, 0);
    }
    get pointer() {
        return this.ptr;
    }
    free() {
        wasm._free(this.ptr);
        this.ptr = 0;
    }
}
export class RealFFTConfig extends AbstractRealFFTConfig {
    constructor(nfft) {
        super(nfft, false);
    }
    work(input, output) {
        this.check(input, output);
        wasm._kiss_fftr(this.ptr, input.pointer, output.pointer);
    }
}
export class InverseRealFFTConfig extends AbstractRealFFTConfig {
    constructor(nfft) {
        super(nfft, true);
    }
    work(input, output) {
        this.check(input, output);
        wasm._kiss_fftri(this.ptr, input.pointer, output.pointer);
        wasm._scale(output.pointer, this.nfft, (1.0 / this.nfft));
    }
}
//# sourceMappingURL=rfft.js.map