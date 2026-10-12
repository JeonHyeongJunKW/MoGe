import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source = await readFile(new URL('../tools/ply_viewer/ply.js', import.meta.url), 'utf8');
const {parsePLY} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const header = format => `ply\nformat ${format} 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`;
const ascii = new TextEncoder().encode(header('ascii')+'1 2 3 255 0 0\nNaN 2 3 0 255 0\n');
const cloud = parsePLY(ascii.buffer);
assert.equal(cloud.count, 1); assert.equal(cloud.omitted, 1);
assert.deepEqual([...cloud.data], [1,2,3,1,0,0]);
for (const little of [true,false]) {
  const h = new TextEncoder().encode(header(little?'binary_little_endian':'binary_big_endian'));
  const b = new Uint8Array(h.length+30); b.set(h);
  const v = new DataView(b.buffer);
  for(let i=0;i<2;i++) {
    [1,2,3].forEach((x,k)=>v.setFloat32(h.length+i*15+k*4,x+i,little));
    b[h.length+i*15+12]=255;
  }
  const c = parsePLY(b.buffer);
  assert.deepEqual([...c.data], [1,2,3,1,0,0,2,3,4,1,0,0]);
  assert.throws(()=>parsePLY(b.buffer.slice(0,-1)), /잘렸/);
}
assert.throws(()=>parsePLY(new TextEncoder().encode('invalid').buffer), /PLY/);
assert.throws(()=>parsePLY(new TextEncoder().encode(header('ascii')+'1 2 3 0 0 0\n').buffer), /잘렸/);
console.log('PLY parser: ASCII, both binary byte orders, invalid points and truncated input passed.');
