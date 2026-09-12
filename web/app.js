/* Murmur -- browser screening aid.
 *
 * The feature pipeline here must stay identical to verify_dsp.mjs, which is the
 * script that was checked cell by cell against the Python the model was measured
 * on. Anything that drifts from it produces a number the held-out metrics do not
 * describe, so this file imports dsp.js rather than reimplementing any of it.
 */

import * as dsp from './dsp.js';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs';

const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
ort.env.wasm.wasmPaths = ORT_DIST;
ort.env.wasm.numThreads = 1;   /* no SharedArrayBuffer, so no cross-origin isolation needed */
ort.env.logLevel = 'error';

const MIC_MAX_SECONDS = 20;
const MIC_MIN_SECONDS = 4;

const LOCATIONS = {
  AV: 'aortic valve',
  MV: 'mitral valve',
  PV: 'pulmonary valve',
  TV: 'tricuspid valve',
  Phc: 'other site',
};

const state = {
  meta: null,
  manifest: [],
  session: null,
  inputName: 'segment',
  outputName: 'logit',
  result: null,
  selected: 0,
  busy: false,
  ctx: null,
  source: null,
  recording: null,
};

/* Exposed so a headless run can read what the page actually computed rather
 * than scraping it back out of the rendered text. */
const probe = {
  ready: false,
  error: null,
  result: null,
  play: { playing: false, segment: null, plays: 0 },
  consoleErrors: [],
};
window.__murmur = probe;

const $ = (sel) => document.querySelector(sel);

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
const WAVE_PATH = 'M2 9h2.6l1.9-5 3 10 1.9-5H16';

/* Built rather than pasted as innerHTML: the icon is decoration, and nothing
 * that renders a patient id should be going through an HTML parser. */
function icon(d) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 18 18');
  svg.setAttribute('width', '16');
  svg.setAttribute('height', '16');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.5');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  const path = document.createElementNS(SVG_NS, 'path');
  path.setAttribute('d', d);
  svg.appendChild(path);
  return svg;
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const fix = (v, n) => Number(v).toFixed(n);

/* Rounding a 0.9995 up to "1.000" would show a certainty the model never
 * expressed, so scores that would round to either end gain a digit instead. */
function scoreText(v) {
  const t = Number(v).toFixed(3);
  if ((t === '1.000' && v < 1) || (t === '0.000' && v > 0)) return Number(v).toFixed(4);
  return t;
}

function clock(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function tick() {
  return new Promise((done) => requestAnimationFrame(() => setTimeout(done, 0)));
}

function setStatus(text) {
  const box = $('#status');
  if (!text) { box.hidden = true; box.textContent = ''; return; }
  box.hidden = false;
  box.textContent = text;
}

function setError(text) {
  const box = $('#error');
  if (!text) { box.hidden = true; box.textContent = ''; return; }
  box.hidden = false;
  box.textContent = text;
  probe.error = text;
}

/* ---- audio decoding ---------------------------------------------------- */

/* Mirrors the WAV reader in verify_dsp.mjs. Kept rather than handing the bytes
 * to decodeAudioData because the browser's decoder resamples to the context
 * rate with a filter of its own, and the verified path resamples with
 * dsp.resample. Same bytes in, same features out. */
function parseWav(buffer) {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  const tag = (off) => String.fromCharCode(bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]);
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') return null;

  let off = 12;
  let fmt = null;
  let dataOff = 0;
  let dataLen = 0;
  while (off < view.byteLength - 8) {
    const id = tag(off);
    const size = view.getUint32(off + 4, true);
    if (id === 'fmt ') {
      fmt = {
        format: view.getUint16(off + 8, true),
        channels: view.getUint16(off + 10, true),
        rate: view.getUint32(off + 12, true),
        bits: view.getUint16(off + 22, true),
      };
    }
    if (id === 'data') { dataOff = off + 8; dataLen = Math.min(size, view.byteLength - dataOff); break; }
    off += 8 + size + (size % 2);
  }
  if (!fmt || !dataOff) return null;

  const ch = fmt.channels;
  if (fmt.format === 1 && fmt.bits === 16) {
    const n = Math.floor(dataLen / 2 / ch);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let c = 0; c < ch; c++) acc += view.getInt16(dataOff + (i * ch + c) * 2, true);
      out[i] = acc / ch;
    }
    return { data: out, rate: fmt.rate };
  }
  if (fmt.format === 3 && fmt.bits === 32) {
    const n = Math.floor(dataLen / 4 / ch);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let c = 0; c < ch; c++) acc += view.getFloat32(dataOff + (i * ch + c) * 4, true);
      out[i] = acc / ch;
    }
    return { data: out, rate: fmt.rate };
  }
  return null;
}

async function decodeAny(buffer) {
  const wav = parseWav(buffer);
  if (wav) return wav;
  const ctx = audioCtx();
  const buf = await ctx.decodeAudioData(buffer.slice(0));
  const n = buf.length;
  const out = new Float32Array(n);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < n; i++) out[i] += ch[i] / buf.numberOfChannels;
  }
  return { data: out, rate: buf.sampleRate };
}

function audioCtx() {
  if (!state.ctx) state.ctx = new (window.AudioContext || window.webkitAudioContext)();
  return state.ctx;
}

/* ---- the pipeline ------------------------------------------------------ */

async function analyse(mono, rate, source) {
  if (state.busy) return;
  state.busy = true;
  setError('');
  stopPlayback();
  lockInputs(true);

  try {
    setStatus(`Resampling ${rate} Hz to ${dsp.SR} Hz...`);
    await tick();
    const audio = dsp.normaliseAudio(dsp.resample(mono, rate, dsp.SR));
    const starts = dsp.segmentIndices(audio.length);

    setStatus(`Computing log-mel features for ${starts.length} segment${starts.length === 1 ? '' : 's'}...`);
    await tick();

    const width = state.meta.width;
    const nMels = dsp.N_MELS;
    const feats = new Float32Array(starts.length * nMels * width);
    for (let i = 0; i < starts.length; i++) {
      let seg = audio.subarray(starts[i], starts[i] + dsp.SEG_LEN);
      if (seg.length < dsp.SEG_LEN) {
        const padded = new Float32Array(dsp.SEG_LEN);
        padded.set(seg);
        seg = padded;
      }
      const { data, frames } = dsp.logMel(seg);
      const w = Math.min(frames, width);
      const base = i * nMels * width;
      for (let m = 0; m < nMels; m++) {
        for (let t = 0; t < w; t++) feats[base + m * width + t] = data[m * frames + t];
      }
    }

    setStatus('Running the model...');
    await tick();

    const tensor = new ort.Tensor('float32', feats, [starts.length, 1, nMels, width]);
    const out = await state.session.run({ [state.inputName]: tensor });
    const logits = out[state.outputName].data;
    const scores = Array.from(logits, sigmoid);

    let top = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[top]) top = i;

    state.result = {
      source,
      audio,
      starts,
      scores,
      top,
      score: scores[top],
      seconds: audio.length / dsp.SR,
      refer: scores[top] >= state.meta.threshold,
    };
    state.selected = top;

    probe.result = {
      source: source.label,
      sample: source.sample ? source.sample.file : null,
      score: state.result.score,
      scores: scores.slice(),
      top,
      starts: starts.slice(),
      threshold: state.meta.threshold,
      refer: state.result.refer,
    };

    setStatus('');
    renderResult();
    $('#result').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  } catch (err) {
    setStatus('');
    setError(`Could not analyse this recording: ${err && err.message ? err.message : err}`);
  } finally {
    state.busy = false;
    lockInputs(false);
  }
}

/* ---- rendering the result ---------------------------------------------- */

function renderResult() {
  const r = state.result;
  const thr = state.meta.threshold;
  const box = $('#result');
  box.hidden = false;
  box.textContent = '';

  /* verdict */
  const head = el('div', 'result-head');
  head.appendChild(el('p', 'result-src', r.source.label));
  const v = el('h3', `verdict${r.refer ? ' is-refer' : ''}`, r.refer ? 'Refer for echocardiography' : 'No referral indicated');
  v.id = 'verdict';
  head.appendChild(v);
  head.appendChild(el('p', 'verdict-sub', r.refer
    ? 'A murmur is audible enough in this recording to be worth imaging. This is a prompt to arrange an echocardiogram, not a finding about the heart.'
    : 'No murmur was audible enough in this recording to prompt imaging. This does not rule anything out; a clinician who suspects disease should refer regardless.'));
  box.appendChild(head);

  box.appendChild(scaleBlock(r, thr));
  box.appendChild(segmentBlock(r, thr));
  if (r.source.sample) box.appendChild(truthBlock(r.source.sample, r.refer));
}

function scaleBlock(r, thr) {
  const block = el('section', 'block');
  block.appendChild(el('h4', 'block-h', 'Model score against the decision threshold'));

  const scale = el('div', 'scale');

  const above = el('div', 'scale-top');
  const tagScore = el('div', 'scale-score');
  tagScore.appendChild(el('b', null, scoreText(r.score)));
  tagScore.appendChild(el('span', null, 'this recording'));
  place(tagScore, r.score);
  above.appendChild(tagScore);
  scale.appendChild(above);

  const mark = el('div', 'scale-mark');
  const tri = el('span', 'scale-tri');
  tri.style.left = `${r.score * 100}%`;
  mark.appendChild(tri);
  scale.appendChild(mark);

  const track = el('div', 'scale-track');
  const over = el('span', 'scale-over');
  over.style.left = `${thr * 100}%`;
  track.appendChild(over);

  const thrLine = el('span', 'scale-thr');
  thrLine.style.left = `${thr * 100}%`;
  track.appendChild(thrLine);

  const pin = el('span', 'scale-pin');
  pin.style.left = `${r.score * 100}%`;
  track.appendChild(pin);
  scale.appendChild(track);

  const ends = el('div', 'scale-ends');
  ends.appendChild(el('span', null, '0.000'));
  ends.appendChild(el('span', null, '1.000'));
  scale.appendChild(ends);

  const labels = el('div', 'scale-labels');
  const tagThr = el('div', 'scale-tag tag-thr');
  tagThr.appendChild(el('b', null, `threshold ${fix(thr, 3)}`));
  tagThr.appendChild(el('span', null, r.refer ? 'refer above this' : 'no referral below this'));
  place(tagThr, thr);
  labels.appendChild(tagThr);
  scale.appendChild(labels);

  block.appendChild(scale);

  const gap = Math.abs(r.score - thr);
  const note = el('p', 'scale-note');
  note.appendChild(document.createTextNode('This recording sits '));
  note.appendChild(el('span', 'scale-gap', fix(gap, 3)));
  note.appendChild(document.createTextNode(r.score >= thr
    ? ' above the threshold. '
    : ' below the threshold. '));
  note.appendChild(document.createTextNode(gap < 0.1
    ? 'That is a close call: a slightly different microphone, room, or moment in the recording could have moved it across. '
    : ''));
  note.appendChild(document.createTextNode('The score is the model\'s output for this recording. It is not a probability that this child has rheumatic heart disease, and it should not be read as one.'));
  block.appendChild(note);

  return block;
}

/* Labels near either end would otherwise run off the track, so they flip to the
 * inside once they get close to an edge. */
function place(node, pos) {
  const pct = pos * 100;
  node.style.left = `${pct}%`;
  if (pct > 60) { node.style.transform = 'translateX(-100%)'; node.style.textAlign = 'right'; }
  else if (pct < 14) node.style.transform = 'none';
  else node.style.transform = 'translateX(-50%)';
}

function segmentBlock(r, thr) {
  const block = el('section', 'block');
  block.appendChild(el('h4', 'block-h', 'The evidence'));

  const lede = el('p', 'note', `This recording is ${fix(r.seconds, 1)} seconds long. It was cut into ${r.starts.length} four-second segment${r.starts.length === 1 ? '' : 's'} spanning the whole of it, and the model scored each one on its own. The score above is the highest of them, so that one segment is the evidence the decision rests on. It is here to listen to.`);
  block.appendChild(lede);

  const chart = el('div', 'segs');
  const line = el('div', 'seg-thr');
  line.style.bottom = `${thr * 100}%`;
  line.appendChild(el('span', 'seg-thr-tag', 'threshold'));
  chart.appendChild(line);

  const axis = el('div', 'seg-axis');

  r.scores.forEach((s, i) => {
    const b = el('button', `seg${i === r.top ? ' is-top' : ''}`);
    b.type = 'button';
    b.setAttribute('aria-pressed', String(i === state.selected));
    const startSec = r.starts[i] / dsp.SR;
    b.title = `Segment ${i + 1}: ${clock(startSec)} to ${clock(startSec + dsp.SEG_SECONDS)}, score ${scoreText(s)}`;
    b.setAttribute('aria-label', b.title);
    const bar = el('span', 'seg-bar');
    bar.style.height = `${Math.max(2, s * 100)}%`;
    b.appendChild(bar);
    b.addEventListener('click', () => selectSegment(i));
    chart.appendChild(b);
    axis.appendChild(el('span', null, clock(startSec)));
  });

  block.appendChild(chart);
  block.appendChild(axis);

  const pick = el('div', 'seg-pick');
  const facts = el('div', 'seg-facts');
  facts.id = 'seg-facts';
  pick.appendChild(facts);
  const play = el('button', 'btn btn-primary');
  play.type = 'button';
  play.id = 'play-btn';
  play.addEventListener('click', togglePlay);
  pick.appendChild(play);
  block.appendChild(pick);

  const sel = axis.children[state.selected];
  if (sel) sel.classList.add('is-sel');

  window.setTimeout(() => describeSelection(), 0);
  return block;
}

function selectSegment(i) {
  state.selected = i;
  document.querySelectorAll('.seg').forEach((n, k) => n.setAttribute('aria-pressed', String(k === i)));
  document.querySelectorAll('.seg-axis span').forEach((n, k) => n.classList.toggle('is-sel', k === i));
  stopPlayback();
  describeSelection();
}

function describeSelection() {
  const r = state.result;
  if (!r) return;
  const i = state.selected;
  const facts = $('#seg-facts');
  const play = $('#play-btn');
  if (!facts || !play) return;
  const startSec = r.starts[i] / dsp.SR;
  facts.textContent = '';
  facts.appendChild(el('b', null, `Segment ${i + 1} of ${r.scores.length}   ${clock(startSec)} to ${clock(startSec + dsp.SEG_SECONDS)}   score ${scoreText(r.scores[i])}`));
  facts.appendChild(document.createTextNode(i === r.top
    ? 'Highest-scoring segment. This is what set the result.'
    : 'Selected segment. The result was set by segment ' + (r.top + 1) + '.'));
  play.textContent = state.source ? 'Stop' : 'Play this segment';
  play.classList.toggle('btn-rec', Boolean(state.source));
}

function truthBlock(sample, refer) {
  const block = el('section', 'block');
  block.appendChild(el('h4', 'block-h', 'Reference annotation'));

  const row = el('dl', 'truth-row');
  const cell = (term, value) => {
    const d = el('div', 'truth-cell');
    d.appendChild(el('dt', null, term));
    d.appendChild(el('dd', null, value));
    row.appendChild(d);
  };
  cell('Murmur', sample.truth);
  cell('Outcome', sample.outcome);
  cell('Age group', sample.age);
  cell('Site', LOCATIONS[sample.location] || sample.location);
  cell('Test set', sample.held_out ? 'held out' : 'not held out');
  block.appendChild(row);

  const call = el('p', 'truth-call');
  let kind = 'call-unknown';
  let verdict = '';
  let rest = '';
  if (sample.truth === 'Unknown') {
    verdict = 'The annotators could not tell either.';
    rest = ' Human experts listened to this recording and marked the murmur label Unknown. Recordings like it were excluded from training, and the model has no way to return that answer, so it returns a number regardless. This one was not part of the held-out test set.';
  } else if (sample.truth === 'Present' && refer) {
    kind = 'call-hit';
    verdict = 'Correct referral.';
    rest = ' A murmur was annotated as present, and the model scored this recording above the threshold. This child would have been sent for an echocardiogram.';
  } else if (sample.truth === 'Present' && !refer) {
    kind = 'call-miss';
    verdict = 'A miss.';
    rest = ' A murmur was annotated as present, and the model scored this recording below the threshold. This child would have been sent home. Misses are the cost that matters most here, and the threshold is set where it is to keep them rare rather than to make the tool look accurate.';
  } else if (sample.truth === 'Absent' && refer) {
    kind = 'call-over';
    verdict = 'An over-referral.';
    rest = ' No murmur was annotated, and the model scored this recording above the threshold. This child would have been sent for an echocardiogram that was not needed, which costs a scarce appointment and frightens a family.';
  } else {
    kind = 'call-hit';
    verdict = 'Correct.';
    rest = ' No murmur was annotated, and the model scored this recording below the threshold.';
  }
  call.classList.add(kind);
  call.appendChild(el('b', null, verdict));
  call.appendChild(document.createTextNode(rest));
  block.appendChild(call);

  if (probe.result) probe.result.call = kind;
  return block;
}

/* ---- playback ---------------------------------------------------------- */

function togglePlay() {
  if (state.source) { stopPlayback(); return; }
  playSegment(state.selected);
}

function playSegment(i) {
  const r = state.result;
  if (!r) return;
  const ctx = audioCtx();
  if (ctx.state === 'suspended') ctx.resume();

  const start = r.starts[i];
  const buf = ctx.createBuffer(1, dsp.SEG_LEN, dsp.SR);
  const chunk = new Float32Array(dsp.SEG_LEN);
  chunk.set(r.audio.subarray(start, Math.min(start + dsp.SEG_LEN, r.audio.length)));
  buf.copyToChannel(chunk, 0);

  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.onended = () => { if (state.source === src) stopPlayback(); };
  src.start();

  state.source = src;
  probe.play = { playing: true, segment: i, plays: probe.play.plays + 1 };
  describeSelection();
}

function stopPlayback() {
  if (state.source) {
    try { state.source.onended = null; state.source.stop(); } catch (err) { /* already ended */ }
    state.source = null;
  }
  probe.play.playing = false;
  describeSelection();
}

/* ---- sources ----------------------------------------------------------- */

function lockInputs(on) {
  document.querySelectorAll('.sample').forEach((b) => { b.disabled = on; });
  const mic = $('#mic-btn');
  if (!state.recording) mic.disabled = on;
}

async function runSample(i) {
  const sample = state.manifest[i];
  document.querySelectorAll('.sample').forEach((b, k) => b.setAttribute('aria-pressed', String(k === i)));
  try {
    setStatus('Loading recording...');
    const res = await fetch(`samples/${sample.file}`);
    if (!res.ok) throw new Error(`sample ${sample.file} returned ${res.status}`);
    const { data, rate } = await decodeAny(await res.arrayBuffer());
    await analyse(data, rate, {
      label: `Sample recording   patient ${patientId(sample.file)}`,
      sample,
    });
  } catch (err) {
    setStatus('');
    setError(`Could not load that recording: ${err && err.message ? err.message : err}`);
  }
}

function patientId(file) {
  const m = file.match(/(\d+)\.wav$/);
  return m ? m[1] : file;
}

/* The card shows the patient id, age and site and nothing else. The manifest
 * also carries the kind -- caught, missed, false-alarm -- and putting that on
 * the card would tell the reader the answer before the model has said a word. */
function renderSamples() {
  const list = $('#samples');
  list.textContent = '';
  state.manifest.forEach((s, i) => {
    const b = el('button', 'card sample');
    b.type = 'button';
    b.disabled = true;
    b.setAttribute('aria-pressed', 'false');
    b.dataset.index = String(i);

    const head = el('div', 'card-head');
    const tile = el('span', 'tile');
    tile.setAttribute('aria-hidden', 'true');
    tile.appendChild(icon(WAVE_PATH));
    head.appendChild(tile);

    const text = el('div');
    text.appendChild(el('div', 'card-t', `Patient ${patientId(s.file)}`));
    text.appendChild(el('div', 'card-s', `${s.age}, ${LOCATIONS[s.location] || s.location}`));
    head.appendChild(text);

    b.appendChild(head);
    b.addEventListener('click', () => runSample(i));
    list.appendChild(b);
  });
}

/* ---- microphone -------------------------------------------------------- */

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) this.port.postMessage(new Float32Array(ch));
    return true;
  }
}
registerProcessor('murmur-tap', Tap);
`;

async function startRecording() {
  setError('');
  let stream;
  try {
    /* Echo cancellation, noise suppression and auto gain are all tuned for
     * speech and would eat a heart sound alive, so they are turned off. */
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
    });
  } catch (err) {
    setError(`No microphone available: ${err && err.message ? err.message : err}. The sample recordings work without one.`);
    return;
  }

  const ctx = audioCtx();
  await ctx.resume();
  const src = ctx.createMediaStreamSource(stream);
  const sink = ctx.createGain();
  sink.gain.value = 0;
  sink.connect(ctx.destination);

  const chunks = [];
  const onChunk = (block) => {
    chunks.push(block);
    let peak = 0;
    for (let i = 0; i < block.length; i++) peak = Math.max(peak, Math.abs(block[i]));
    const fill = $('#mic-level');
    if (fill) fill.style.width = `${Math.min(100, peak * 140)}%`;
  };

  let node = null;
  if (ctx.audioWorklet) {
    try {
      const url = URL.createObjectURL(new Blob([WORKLET], { type: 'text/javascript' }));
      await ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      node = new AudioWorkletNode(ctx, 'murmur-tap');
      node.port.onmessage = (e) => onChunk(e.data);
    } catch (err) {
      node = null;
    }
  }
  if (!node) {
    node = ctx.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (e) => onChunk(new Float32Array(e.inputBuffer.getChannelData(0)));
  }
  src.connect(node);
  node.connect(sink);

  const started = performance.now();
  const timer = window.setInterval(() => {
    const secs = (performance.now() - started) / 1000;
    $('#mic-state').textContent = `Recording  ${clock(secs)}  (stops at ${clock(MIC_MAX_SECONDS)})`;
    if (secs >= MIC_MAX_SECONDS) stopRecording();
  }, 100);

  state.recording = { stream, src, node, sink, chunks, timer, started, rate: ctx.sampleRate };

  const btn = $('#mic-btn');
  btn.textContent = 'Stop and analyse';
  btn.classList.add('btn-rec');
  $('#mic-meter').hidden = false;
  $('#mic-state').textContent = 'Recording  0:00';
  document.querySelectorAll('.sample').forEach((b) => { b.disabled = true; });
}

async function stopRecording() {
  const rec = state.recording;
  if (!rec) return;
  state.recording = null;
  window.clearInterval(rec.timer);

  try { rec.src.disconnect(); rec.node.disconnect(); rec.sink.disconnect(); } catch (err) { /* already down */ }
  if (rec.node.port) rec.node.port.onmessage = null;
  rec.node.onaudioprocess = null;
  rec.stream.getTracks().forEach((t) => t.stop());

  const btn = $('#mic-btn');
  btn.textContent = 'Record from microphone';
  btn.classList.remove('btn-rec');
  $('#mic-meter').hidden = true;
  $('#mic-level').style.width = '0';
  document.querySelectorAll('.sample').forEach((b) => { b.disabled = false; });

  let total = 0;
  rec.chunks.forEach((c) => { total += c.length; });
  const seconds = total / rec.rate;
  if (seconds < MIC_MIN_SECONDS) {
    $('#mic-state').textContent = '';
    setError(`That recording is only ${fix(seconds, 1)} seconds. The model needs at least ${MIC_MIN_SECONDS} seconds of audio.`);
    return;
  }

  const mono = new Float32Array(total);
  let at = 0;
  rec.chunks.forEach((c) => { mono.set(c, at); at += c.length; });

  $('#mic-state').textContent = `Captured ${fix(seconds, 1)} s at ${rec.rate} Hz`;
  document.querySelectorAll('.sample').forEach((b) => b.setAttribute('aria-pressed', 'false'));
  await analyse(mono, rec.rate, { label: `Microphone   ${fix(seconds, 1)} seconds`, sample: null });
}

/* ---- how it works ------------------------------------------------------ */

/* metrics.test is reported at whatever threshold training settled on. The
 * threshold this tool ships with is the chosen operating point, which is a
 * different place on the same curve -- so the panel quotes that point's
 * sensitivity and specificity, not the ones next to them in the file. Quoting
 * the wrong pair would describe a tool nobody is running. */
function heldOut(m) {
  const t = m.metrics.test;
  const murmurs = t.tp + t.fn;
  const clean = t.tn + t.fp;
  const op = m.chosen_operating_point;
  if (!op) return { n: t.n, auc: t.auc, murmurs, clean, tp: t.tp, fn: t.fn, tn: t.tn, fp: t.fp,
    sensitivity: t.sensitivity, specificity: t.specificity };
  return {
    n: t.n, auc: t.auc, murmurs, clean,
    fn: op.missed, fp: op.over_referred,
    tp: murmurs - op.missed, tn: clean - op.over_referred,
    sensitivity: op.test_sensitivity, specificity: op.test_specificity,
  };
}

function renderAbout() {
  const m = state.meta;
  const t = heldOut(m);

  const stages = [
    ['Resample and normalise',
      'The recording is mixed to mono, resampled to a low rate, and divided by its own peak so that a quiet recording and a loud one arrive at the model on the same footing.',
      `${m.sample_rate} Hz, peak normalised`],
    ['Log-mel spectrogram',
      'Each segment becomes a picture of how sound energy is spread across frequency over time, on a mel scale bounded to the band where heart sounds and murmurs live. Every picture is standardised to zero mean and unit variance.',
      `${m.n_mels} mel bands, ${m.fmin}-${m.fmax} Hz, ${m.n_fft}-point FFT, hop ${m.hop}`],
    ['A small CNN scores each segment',
      'A four-block convolutional network reads one spectrogram at a time and returns a single number for that four seconds of audio. It sees no more of the recording than that. Long recordings are sampled down to eight segments spread across their length rather than scored window by window.',
      `${m.seg_seconds} s segments every ${m.hop_seconds} s, at most 8 per recording, ${m.n_mels} by ${m.width} input`],
    ['Take the maximum across segments',
      'A murmur need only be audible somewhere, so the recording takes the highest segment score rather than the average. Averaging would let a few clean seconds bury a murmur that was plainly there.',
      `max over segments, refer at score >= ${fix(m.threshold, 4)}`],
  ];

  const ol = $('#stages');
  ol.textContent = '';
  stages.forEach((s, i) => {
    const li = el('li', 'stage');
    li.appendChild(el('div', 'stage-n', String(i + 1).padStart(2, '0')));
    const body = el('div');
    body.appendChild(el('div', 'stage-t', s[0]));
    body.appendChild(el('div', 'stage-d', s[1]));
    body.appendChild(el('div', 'stage-k', s[2]));
    li.appendChild(body);
    ol.appendChild(li);
  });

  const figures = [
    ['Held-out patients', String(t.n), 'never seen in training'],
    ['AUC', fix(t.auc, 3), 'ranking, before any threshold'],
    ['Sensitivity', fix(t.sensitivity, 3), `${t.tp} of ${t.murmurs} murmurs caught`],
    ['Specificity', fix(t.specificity, 3), `${t.tn} of ${t.clean} cleared`],
  ];
  const dl = $('#figures');
  dl.textContent = '';
  figures.forEach(([term, value, sub]) => {
    const d = el('div');
    d.appendChild(el('dt', null, term));
    const dd = el('dd', null, value);
    dd.appendChild(el('span', 'fig-sub', sub));
    d.appendChild(dd);
    dl.appendChild(d);
  });

  const miss = t.fn === 1 ? '1 murmur' : `${t.fn} murmurs`;
  const over = t.fp === 1 ? '1 child' : `${t.fp} children`;
  $('#limits-prose').textContent =
    `On ${t.n} held-out patients the model reached an AUC of ${fix(t.auc, 3)}. At the threshold this tool uses, ${fix(m.threshold, 4)}, it caught ${t.tp} of the ${t.murmurs} patients annotated with a murmur and cleared ${t.tn} of the ${t.clean} without one. That means it missed ${miss}, and would have sent ${over} for an echocardiogram they did not need. Both of those are the price of this operating point, and neither is a rounding error.`;
}

/* ---- views ------------------------------------------------------------- */

const VIEWS = ['screen', 'evidence', 'about', 'limits'];

function showView(name) {
  document.querySelectorAll('.nav-item').forEach((t) => {
    const on = t.dataset.view === name;
    t.classList.toggle('is-on', on);
    if (on) t.setAttribute('aria-current', 'page');
    else t.removeAttribute('aria-current');
  });
  VIEWS.forEach((v) => {
    const node = $(`#view-${v}`);
    if (node) node.hidden = v !== name;
  });
  window.scrollTo({ top: 0 });
}

/* ---- boot -------------------------------------------------------------- */

async function boot() {
  /* Both the rail nav and the primary button carry data-view. */
  document.querySelectorAll('[data-view]').forEach((t) => {
    t.addEventListener('click', () => showView(t.dataset.view));
  });
  $('#mic-btn').addEventListener('click', () => {
    if (state.recording) stopRecording();
    else startRecording();
  });

  try {
    const [meta, manifest] = await Promise.all([
      fetch('model_meta.json').then((r) => r.json()),
      fetch('samples/manifest.json').then((r) => r.json()),
    ]);
    state.meta = meta;
    state.manifest = manifest;
    probe.threshold = meta.threshold;

    renderSamples();
    renderAbout();

    $('#boot').textContent = 'Loading model (about 600 KB, once)...';
    state.session = await ort.InferenceSession.create('murmur.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    state.inputName = state.session.inputNames[0];
    state.outputName = state.session.outputNames[0];

    $('#boot').hidden = true;
    document.querySelectorAll('.sample').forEach((b) => { b.disabled = false; });
    probe.ready = true;
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    $('#boot').textContent = `The model could not be loaded: ${msg}`;
    probe.error = msg;
  }
}

window.addEventListener('error', (e) => probe.consoleErrors.push(String(e.message)));
window.addEventListener('unhandledrejection', (e) => probe.consoleErrors.push(String(e.reason)));

boot();
