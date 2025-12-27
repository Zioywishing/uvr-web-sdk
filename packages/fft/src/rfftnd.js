import { KissFFTConfig } from "./types.js";
import { checkRealFFT } from "./rfft.js";
import { wasm } from "./wasm.js";
class AbstractRealFFTNDConfig extends KissFFTConfig {
    dims;
    ptr = 0;
    constructor(dims, inverse) {
        super(dims.reduce((x, y) => (x * y)), inverse);
        this.dims = dims;
        const ndims = dims.length;
        checkRealFFT(dims[ndims - 1]);
        const dimsPtr = wasm._malloc((Int32Array.BYTES_PER_ELEMENT * ndims));
        dims.forEach((x, i) => {
            wasm.HEAP32[dimsPtr / Int32Array.BYTES_PER_ELEMENT + i] = x;
        });
        this.ptr = wasm._kiss_fftndr_alloc(dimsPtr, ndims, inverse, 0, 0);
        wasm._free(dimsPtr);
    }
    get pointer() {
        return this.ptr;
    }
    free() {
        wasm._free(this.ptr);
        this.ptr = 0;
    }
}
export class RealFFTNDConfig extends AbstractRealFFTNDConfig {
    constructor(dims) {
        super(dims, false);
    }
    work(input, output) {
        this.check(input, output);
        wasm._kiss_fftndr(this.ptr, input.pointer, output.pointer);
    }
}
export class InverseRealFFTNDConfig extends AbstractRealFFTNDConfig {
    constructor(dims) {
        super(dims, true);
    }
    work(input, output) {
        this.check(input, output);
        wasm._kiss_fftndri(this.ptr, input.pointer, output.pointer);
        wasm._scale(output.pointer, this.nfft, (1.0 / this.nfft));
    }
}
//# sourceMappingURL=rfftnd.js.map