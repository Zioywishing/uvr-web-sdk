/**
 * 简单的 Protobuf 读取器，用于解析 ONNX 模型中的输入形状
 */
class ProtobufReader {
  private offset = 0;

  constructor(private buffer: Uint8Array) {}

  get finished(): boolean {
    return this.offset >= this.buffer.length;
  }

  readVarint(): number {
    let result = 0n;
    let shift = 0n;
    while (true) {
      const byte = BigInt(this.buffer[this.offset++]);
      result |= (byte & 0x7fn) << shift;
      if (!(byte & 0x80n)) break;
      shift += 7n;
    }
    return Number(result);
  }

  // 处理大整数的 varint，虽然在 ONNX 形状中不常用，但为了健壮性
  readVarintBigInt(): bigint {
    let result = BigInt(0);
    let shift = BigInt(0);
    while (true) {
      const byte = BigInt(this.buffer[this.offset++]);
      result |= (byte & 0x7fn) << shift;
      if (!(byte & 0x80n)) break;
      shift += 7n;
    }
    return result;
  }

  readTag(): { field: number; wireType: number } {
    const tag = this.readVarint();
    return {
      field: tag >>> 3,
      wireType: tag & 0x07,
    };
  }

  readString(): string {
    const length = this.readVarint();
    const bytes = this.buffer.subarray(this.offset, this.offset + length);
    this.offset += length;
    return new TextDecoder().decode(bytes);
  }

  readBytes(): Uint8Array {
    const length = this.readVarint();
    const bytes = this.buffer.subarray(this.offset, this.offset + length);
    this.offset += length;
    return bytes;
  }

  skip(wireType: number): void {
    switch (wireType) {
      case 0: // Varint
        this.readVarint();
        break;
      case 1: // 64-bit
        this.offset += 8;
        break;
      case 2: // Length-delimited
        const length = this.readVarint();
        this.offset += length;
        break;
      case 5: // 32-bit
        this.offset += 4;
        break;
      default:
        throw new Error(`未知 wire type: ${wireType}`);
    }
  }

  subarray(length: number): ProtobufReader {
    const sub = new ProtobufReader(this.buffer.subarray(this.offset, this.offset + length));
    this.offset += length;
    return sub;
  }
}

export interface OnnxInput {
  name: string;
  shape: (number | string)[];
}

/**
 * 解析 ONNX 模型文件以获取输入张量形态
 * @param buffer 模型文件的二进制内容
 */
export function parseOnnxInputShapes(buffer: Uint8Array): OnnxInput[] {
  const reader = new ProtobufReader(buffer);
  const inputs: OnnxInput[] = [];

  while (!reader.finished) {
    const { field, wireType } = reader.readTag();
    if ((field === 3 || field === 7) && wireType === 2) { // graph: GraphProto (可能在 3 或 7)
      const length = reader.readVarint();
      const graphReader = reader.subarray(length);
      parseGraph(graphReader, inputs);
      if (inputs.length > 0) break; 
    } else {
      reader.skip(wireType);
    }
  }

  return inputs;
}

function parseGraph(reader: ProtobufReader, inputs: OnnxInput[]): void {
  while (!reader.finished) {
    const { field, wireType } = reader.readTag();
    if (field === 11 && wireType === 2) { // input: repeated ValueInfoProto
      const length = reader.readVarint();
      const inputReader = reader.subarray(length);
      const input = parseValueInfo(inputReader);
      if (input) inputs.push(input);
    } else {
      reader.skip(wireType);
    }
  }
}

function parseValueInfo(reader: ProtobufReader): OnnxInput | null {
  let name = '';
  let shape: (number | string)[] = [];

  while (!reader.finished) {
    const { field, wireType } = reader.readTag();
    if (field === 1 && wireType === 2) {
      name = reader.readString();
    } else if (field === 2 && wireType === 2) { // type: TypeProto
      const typeReader = reader.subarray(reader.readVarint());
      shape = parseType(typeReader);
    } else {
      reader.skip(wireType);
    }
  }

  return name ? { name, shape } : null;
}

function parseType(reader: ProtobufReader): (number | string)[] {
  while (!reader.finished) {
    const { field, wireType } = reader.readTag();
    if (field === 1 && wireType === 2) { // tensor_type: Tensor
      const tensorReader = reader.subarray(reader.readVarint());
      return parseTensorType(tensorReader);
    } else {
      reader.skip(wireType);
    }
  }
  return [];
}

function parseTensorType(reader: ProtobufReader): (number | string)[] {
  while (!reader.finished) {
    const { field, wireType } = reader.readTag();
    if (field === 2 && wireType === 2) { // shape: TensorShapeProto
      const shapeReader = reader.subarray(reader.readVarint());
      return parseTensorShape(shapeReader);
    } else {
      reader.skip(wireType);
    }
  }
  return [];
}

function parseTensorShape(reader: ProtobufReader): (number | string)[] {
  const dims: (number | string)[] = [];
  while (!reader.finished) {
    const { field, wireType } = reader.readTag();
    if (field === 1 && wireType === 2) { // dim: repeated Dimension
      const dimReader = reader.subarray(reader.readVarint());
      dims.push(parseDimension(dimReader));
    } else {
      reader.skip(wireType);
    }
  }
  return dims;
}

function parseDimension(reader: ProtobufReader): number | string {
  while (!reader.finished) {
    const { field, wireType } = reader.readTag();
    if (field === 1 && wireType === 0) { // dim_value: int64
      return Number(reader.readVarintBigInt());
    } else if (field === 2 && wireType === 2) { // dim_param: string
      return reader.readString();
    } else {
      reader.skip(wireType);
    }
  }
  return 0;
}
