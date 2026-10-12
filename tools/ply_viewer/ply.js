// Point-cloud reader: ASCII and binary PLY, vertex element first; faces ignored.
export function parsePLY(buffer) {
  const bytes = new Uint8Array(buffer);
  const prefix = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 65536)));
  const end = /^end_header\r?\n/m.exec(prefix);
  if (!prefix.startsWith('ply\n') && !prefix.startsWith('ply\r\n')) throw Error('PLY 파일이 아닙니다.');
  if (!end) throw Error('PLY header가 없거나 너무 큽니다.');
  const header = prefix.slice(0, end.index + end[0].length);
  const offset = new TextEncoder().encode(header).length;
  const types = {
    char: ['getInt8', 1], int8: ['getInt8', 1], uchar: ['getUint8', 1], uint8: ['getUint8', 1],
    short: ['getInt16', 2], int16: ['getInt16', 2], ushort: ['getUint16', 2], uint16: ['getUint16', 2],
    int: ['getInt32', 4], int32: ['getInt32', 4], uint: ['getUint32', 4], uint32: ['getUint32', 4],
    float: ['getFloat32', 4], float32: ['getFloat32', 4], double: ['getFloat64', 8], float64: ['getFloat64', 8],
  };
  let format, count, element, stride = 0;
  const props = [];
  for (const line of header.split(/\r?\n/)) {
    const t = line.trim().split(/\s+/);
    if (t[0] === 'format') format = t[1];
    if (t[0] === 'element') {
      if (count === undefined && t[1] !== 'vertex') throw Error('vertex가 첫 번째 element인 PLY만 지원합니다.');
      element = t[1];
      if (element === 'vertex') count = Number(t[2]);
    }
    if (t[0] === 'property' && element === 'vertex') {
      if (!types[t[1]]) throw Error(`지원하지 않는 vertex property: ${t[1]}`);
      props.push({name: t[2], type: t[1], offset: stride, index: props.length});
      stride += types[t[1]][1];
    }
  }
  if (!Number.isInteger(count) || count < 0 || count > 20000000) throw Error('유효하지 않거나 너무 많은 vertex 수입니다.');
  if (!['ascii', 'binary_little_endian', 'binary_big_endian'].includes(format)) throw Error('지원하지 않는 PLY format입니다.');
  const xyz = ['x', 'y', 'z'].map(name => props.find(p => p.name === name));
  if (xyz.some(p => !p)) throw Error('PLY에 x, y, z가 필요합니다.');
  const rgb = ['red', 'green', 'blue'].map(name => props.find(p => p.name === name));
  const step = Math.max(1, Math.ceil(count / 1000000));
  const data = new Float32Array(Math.ceil(count / step) * 6);
  const view = new DataView(buffer);
  let lines, lineIndex = 0;
  if (format === 'ascii') lines = new TextDecoder().decode(bytes.subarray(offset)).trim().split(/\r?\n/);
  else if (offset + count * stride > buffer.byteLength) throw Error('PLY binary 데이터가 잘렸습니다.');
  let used = 0, omitted = 0;
  for (let i = 0; i < count; i++) {
    let tokens;
    if (lines) {
      while (lineIndex < lines.length && !lines[lineIndex].trim()) lineIndex++;
      if (lineIndex >= lines.length) throw Error('PLY ASCII 데이터가 잘렸습니다.');
      tokens = lines[lineIndex++].trim().split(/\s+/);
      if (tokens.length < props.length) throw Error('PLY vertex property가 부족합니다.');
    }
    if (i % step) continue;
    const read = p => lines ? Number(tokens[p.index]) : view[types[p.type][0]](offset + i * stride + p.offset, format === 'binary_little_endian');
    const point = xyz.map(read);
    if (!point.every(v => Number.isFinite(v) && Math.abs(v) <= 3e38)) { omitted++; continue; }
    data.set(point, used * 6);
    for (let c = 0; c < 3; c++) {
      const p = rgb[c];
      let value = p ? read(p) : 190;
      if (!p || !['float', 'float32', 'double', 'float64'].includes(p.type)) value /= 255;
      data[used * 6 + 3 + c] = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0.75;
    }
    used++;
  }
  return {data: data.subarray(0, used * 6), count: used, originalCount: count, omitted, step};
}
