import { KissFFTConfig } from "./types.js";
import { wasm } from "./wasm.js";
export class FFTConfig extends KissFFTConfig {
    ptr = 0;
    constructor(nfft, inverse) {
        super(nfft, inverse);
        this.ptr = wasm._kiss_fft_alloc(nfft, inverse, 0, 0);
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
        wasm._kiss_fft(this.ptr, input.pointer, output.pointer);
        if (this.inverse) {
            wasm._scale(output.pointer, (this.nfft * 2), (1.0 / this.nfft));
        }
    }
}
//# sourceMappingURL=fft.js.map