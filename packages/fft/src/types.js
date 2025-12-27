import { wasm } from "./wasm.js";
const NULLPTR_F32 = 0;
const BYTES_PER_ELEMENT = Float32Array.BYTES_PER_ELEMENT; // 4
export class KissFFTArray {
    dataPointer = NULLPTR_F32;
    dataLength = 0;
    get pointer() {
        return this.dataPointer;
    }
    // The instance will become invalid and cannot be used after `free()` is called.
    // Note that all the methods won't check this for performance.
    free() {
        wasm._free(this.dataPointer);
        this.dataPointer = NULLPTR_F32;
    }
    get valid() {
        return this.dataPointer !== 0;
    }
    asFloat32Array() {
        const f32p = this.dataPointer / BYTES_PER_ELEMENT;
        return wasm.HEAPF32.subarray(f32p, f32p + this.dataLength);
    }
    // The data is copied.
    toFloat32Array() {
        return new Float32Array(this.asFloat32Array());
    }
}
export class RealArray extends KissFFTArray {
    constructor(nOrArray) {
        super();
        if (typeof nOrArray === "number") {
            this.dataLength = nOrArray;
            this.dataPointer = wasm._allocate(this.dataLength);
        }
        else {
            this.dataLength = nOrArray.dataLength;
            this.dataPointer = wasm._copy(nOrArray.pointer, this.dataLength);
        }
    }
    get length() {
        return this.dataLength;
    }
    static fromDataArray(arr) {
        return RealArray.fromArray(arr);
    }
    static fromArray(arr) {
        const real = new RealArray(arr.length);
        wasm.HEAPF32.set(arr, real.dataPointer / BYTES_PER_ELEMENT);
        return real;
    }
}
export class ComplexArray extends KissFFTArray {
    constructor(nOrArray) {
        super();
        if (typeof nOrArray === "number") {
            this.dataLength = 2 * nOrArray;
            this.dataPointer = wasm._allocate(this.dataLength);
        }
        else {
            this.dataLength = nOrArray.dataLength;
            this.dataPointer = wasm._copy(nOrArray.pointer, this.dataLength);
        }
    }
    get length() {
        return this.dataLength / 2;
    }
    realAt(i) {
        return wasm._get_value(this.dataPointer, i * 2);
    }
    imagAt(i) {
        return wasm._get_value(this.dataPointer, i * 2 + 1);
    }
    valueAt(i) {
        return {
            real: this.realAt(i),
            imag: this.imagAt(i)
        };
    }
    // The structure of memory:
    // (arr[0], arr[1]) are the real and imaginary part of the first number, respectively.
    static fromDataArray(arr) {
        if (arr.length % 2 === 1) {
            throw new Error("Array length must be even.");
        }
        const complex = new ComplexArray(arr.length / 2);
        wasm.HEAPF32.set(arr, complex.dataPointer / BYTES_PER_ELEMENT);
        return complex;
    }
    static fromArray(real, imag) {
        const n = real.length;
        if (imag !== undefined && n !== imag.length) {
            throw new Error(`Inconsistent length of arguments: real=${n} - imag=${imag.length}`);
        }
        const arr = new ComplexArray(n);
        for (let i = 0; i < n; ++i) {
            wasm._set_value(arr.dataPointer, i * 2, real[i]);
        }
        if (imag !== undefined) {
            for (let i = 0; i < n; ++i) {
                wasm._set_value(arr.dataPointer, i * 2 + 1, imag[i]);
            }
        }
        return arr;
    }
    toRealArray() {
        return Float32Array.from({ length: this.length }).map((_, i) => this.realAt(i));
    }
    toImagArray() {
        return Float32Array.from({ length: this.length }).map((_, i) => this.imagAt(i));
    }
}
export class KissFFTConfig {
    nfft;
    inverse;
    constructor(nfft, inverse) {
        this.nfft = nfft;
        this.inverse = inverse;
    }
    get valid() {
        return this.pointer !== 0;
    }
    check(input, output) {
        if (input.length !== this.nfft || output.length !== this.nfft) {
            throw new Error("Input or Output length is inconsistent to Config length.");
        }
    }
}
//# sourceMappingURL=types.js.map