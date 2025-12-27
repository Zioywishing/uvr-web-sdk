import { KissFFTConfig } from "./types.js";
import { wasm } from "./wasm.js";
export class FFTNDConfig extends KissFFTConfig {
    dims;
    ptr = 0;
    constructor(dims, inverse) {
        super(dims.reduce((x, y) => (x * y)), inverse);
        this.dims = dims;
        const ndims = dims.length;
        const dimsPtr = wasm._malloc((Int32Array.BYTES_PER_ELEMENT * ndims));
        dims.forEach((x, i) => {
            wasm.HEAP32[dimsPtr / Int32Array.BYTES_PER_ELEMENT + i] = x;
        });
        this.ptr = wasm._kiss_fftnd_alloc(dimsPtr, ndims, inverse, 0, 0);
        wasm._free(dimsPtr);
    }
    get pointer() {
        return this.ptr;
    }
    free() {
        wasm._free(this.ptr);
        this.ptr = 0;
    }
    work(input, output) {
        this.check(input, output);
        wasm._kiss_fftnd(this.ptr, input.pointer, output.pointer);
        if (this.inverse) {
            wasm._scale(output.pointer, (this.nfft * 2), (1.0 / this.nfft));
        }
    }
}
//# sourceMappingURL=fftnd.js.map