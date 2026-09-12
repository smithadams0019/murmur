/* Feature extraction, ported from features.py.
 *
 * This must match the Python exactly. The model was measured on features
 * Python produced; if the browser computes something different, the held-out
 * numbers describe a system nobody is running. verify_dsp.py checks the two
 * against each other and the check is part of the build, not a courtesy.
 */
export const SR = 4000;
export const SEG_SECONDS = 4.0;
export const SEG_LEN = SR * SEG_SECONDS;
export const HOP_SECONDS = 2.0;
export const N_MELS = 64;
export const N_FFT = 256;
export const HOP = 128;
export const FMIN = 25.0;
export const FMAX = 1000.0;
export const MAX_SEGMENTS = 8;

const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
const melToHz = (m) => 700 * (10 ** (m / 2595) - 1);

export function melFilterbank(sr = SR, nFft = N_FFT, nMels = N_MELS, fmin = FMIN, fmax = FMAX) {
  const lo = hzToMel(fmin), hi = hzToMel(fmax);
  const edges = Array.from({ length: nMels + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (nMels + 1)));
  const bins = nFft / 2 + 1;
  const freqs = Array.from({ length: bins }, (_, i) => (i * sr) / 2 / (bins - 1));
  const fb = [];
  for (let m = 0; m < nMels; m++) {
    const [a, b, c] = [edges[m], edges[m + 1], edges[m + 2]];
    const row = new Float32Array(bins);
    for (let k = 0; k < bins; k++) {
      const left = (freqs[k] - a) / Math.max(b - a, 1e-9);
      const right = (c - freqs[k]) / Math.max(c - b, 1e-9);
      row[k] = Math.max(Math.min(left, right), 0);
    }
    fb.push(row);
  }
  return fb;
}
const FB = melFilterbank();

/* scipy.signal.stft applies a Hann window scaled so the transform is unitary in
 * amplitude. The scale factor matters: it moves every log-mel value by a
 * constant, which per-segment normalisation would hide -- but only if the
 * normalisation is applied, so it is reproduced rather than assumed away. */
function hann(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}
const WIN = hann(N_FFT);
const WIN_SUM = WIN.reduce((a, b) => a + b, 0);

function dft(re, im) {
  /* N_FFT is 256 and segments are few, so a direct radix-2 FFT is plenty. */
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

export function logMel(seg) {
  const bins = N_FFT / 2 + 1;
  const frames = Math.floor((seg.length - N_FFT) / HOP) + 1;
  const out = new Float32Array(N_MELS * frames);
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
  const power = new Float64Array(bins);
  for (let t = 0; t < frames; t++) {
    const off = t * HOP;
    for (let i = 0; i < N_FFT; i++) { re[i] = seg[off + i] * WIN[i]; im[i] = 0; }
    dft(re, im);
    for (let k = 0; k < bins; k++) {
      const mag = Math.hypot(re[k], im[k]) / WIN_SUM;
      /* scipy's one-sided spectrum doubles every bin except DC and Nyquist. */
      const amp = (k === 0 || k === bins - 1) ? mag : mag * 2;
      power[k] = amp * amp;
    }
    for (let m = 0; m < N_MELS; m++) {
      const row = FB[m];
      let acc = 0;
      for (let k = 0; k < bins; k++) acc += row[k] * power[k];
      out[m * frames + t] = Math.log(acc + 1e-8);
    }
  }
  let mean = 0;
  for (let i = 0; i < out.length; i++) mean += out[i];
  mean /= out.length;
  let v = 0;
  for (let i = 0; i < out.length; i++) v += (out[i] - mean) ** 2;
  const std = Math.sqrt(v / out.length);
  for (let i = 0; i < out.length; i++) out[i] = (out[i] - mean) / (std + 1e-6);
  return { data: out, nMels: N_MELS, frames };
}

export function normaliseAudio(x) {
  let peak = 0;
  for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
  if (peak === 0) return x;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] / peak;
  return out;
}

export function resample(x, from, to) {
  if (from === to) return x;
  const n = Math.round((x.length * to) / from);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const pos = (i * (x.length - 1)) / (n - 1);
    const a = Math.floor(pos), b = Math.min(a + 1, x.length - 1);
    out[i] = x[a] + (x[b] - x[a]) * (pos - a);
  }
  return out;
}

export function segmentIndices(len) {
  const hop = SR * HOP_SECONDS;
  let starts = [];
  for (let s = 0; s <= Math.max(0, len - SEG_LEN); s += hop) starts.push(s);
  if (starts.length === 0) starts = [0];
  if (starts.length > MAX_SEGMENTS) {
    const pick = [];
    for (let i = 0; i < MAX_SEGMENTS; i++) {
      pick.push(starts[Math.round((i * (starts.length - 1)) / (MAX_SEGMENTS - 1))]);
    }
    starts = [...new Set(pick)];
  }
  return starts;
}
