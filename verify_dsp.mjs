/* Compute log-mel for one recording with the browser code, dump as JSON. */
import fs from 'fs';
import * as dsp from './web/dsp.js';

const path = process.argv[2];
const buf = fs.readFileSync(path);
// minimal PCM16 WAV reader
let off = 12, fmt = null, dataOff = 0, dataLen = 0;
while (off < buf.length - 8) {
  const id = buf.toString('ascii', off, off + 4);
  const sz = buf.readUInt32LE(off + 4);
  if (id === 'fmt ') fmt = { channels: buf.readUInt16LE(off + 10), rate: buf.readUInt32LE(off + 12), bits: buf.readUInt16LE(off + 22) };
  if (id === 'data') { dataOff = off + 8; dataLen = sz; break; }
  off += 8 + sz + (sz % 2);
}
const n = Math.floor(dataLen / 2 / fmt.channels);
let x = new Float32Array(n);
for (let i = 0; i < n; i++) {
  let acc = 0;
  for (let c = 0; c < fmt.channels; c++) acc += buf.readInt16LE(dataOff + (i * fmt.channels + c) * 2);
  x[i] = acc / fmt.channels;
}
x = dsp.normaliseAudio(dsp.resample(x, fmt.rate, dsp.SR));
const starts = dsp.segmentIndices(x.length);
const segs = starts.map((s) => {
  let seg = x.subarray(s, s + dsp.SEG_LEN);
  if (seg.length < dsp.SEG_LEN) { const p = new Float32Array(dsp.SEG_LEN); p.set(seg); seg = p; }
  return dsp.logMel(seg);
});
fs.writeFileSync(process.argv[3], JSON.stringify({
  rate: fmt.rate, samples: n, starts, nMels: segs[0].nMels, frames: segs[0].frames,
  segments: segs.map((s) => Array.from(s.data)),
}));
console.log(`js: ${segs.length} segments, ${segs[0].nMels}x${segs[0].frames}`);
