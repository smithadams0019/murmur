# Murmur

Decides which children need an echocardiogram, and says out loud what it missed.

Screening schoolchildren in Cambodia and Mozambique, doctors examined them the usual way —
stethoscope, trained ears — and found rheumatic heart disease in 2.2 and 2.3 children per
thousand. Echocardiography on the same children found 21.5 and 30.4 per thousand. Roughly ten
times as many. Listening had found one case in ten.

Those children were not failed by a lack of medicine. Rheumatic heart disease follows an
untreated strep throat, and further damage is prevented with benzathine penicillin G — an
injection every three or four weeks, cheap for decades. The problem is that the disease is
silent until a valve is wrecked, the only reliable way to see it early is an echo, and echo is
scarce exactly where the disease is common.

Murmur does not try to be the echo. It tries to be the thing that decides who gets one.

## What the numbers are

Trained and evaluated on [CirCor DigiScope](https://physionet.org/content/circor-heart-sound/)
(942 patients, PhysioNet, ODC-By), split **by patient** so no child appears in both training and
test.

| | |
|---|---|
| Held-out test patients | 174 |
| Test AUC | **0.957** |
| Sensitivity at the shipped threshold | **94.3%** |
| Specificity | 77.0% |
| Murmurs missed | **2 of 35** |
| Children over-referred | 32 |

The threshold (0.9455) is not the one that maximises accuracy. A higher one would make the
table look better and send two more children home. Missing a murmur and raising a false alarm
are not the same mistake: one costs a child a valve, the other costs an ultrasound appointment.
The operating point is set accordingly, and the two missed murmurs are printed next to the AUC
rather than left in a confusion matrix nobody opens.

Reproduce with `python evaluate.py`.

## How it works

A heart-sound recording is cut into four-second segments, each turned into a log-mel
spectrogram, and a small four-block convolutional network scores them in the browser. The
patient score is the **loudest segment, not the average** — a murmur audible at one auscultation
position and not another is still a murmur.

The model is 575 KB and runs on the device. No audio of a child's chest is uploaded anywhere.

## The same arithmetic in both languages

`features.py` produced the numbers above. `web/dsp.js` runs in the browser. If they drifted, the
published accuracy would describe software nobody runs. `verify_dsp.py` / `verify_dsp.mjs`
compare the feature extraction across languages, and `verify_predictions.py` checks the scores
themselves.

There is no bandpass filter in this pipeline, deliberately. The Python originally had a
third-order Butterworth that could not be reproduced bit-for-bit in JavaScript. A model that
sees different features in the browser than it saw in training is worse than a slightly less
accurate model, because you no longer know what it is doing. The filter was removed and the
model retrained without it.

## What this is not

Not a medical device. **It does not diagnose rheumatic heart disease, or anything else.** A
murmur is a sound, not a disease, and plenty of children have innocent ones. The output is a
referral decision, and the decision to act on it belongs to a clinician.

Never deployed or trialled. The screening evidence above says echocardiography finds what
listening misses. It says nothing about this software.

Not characterised outside its corpus. CirCor was recorded in one screening programme with
particular equipment. A noisy clinic, a different stethoscope, a different population — none of
that is measured here.

## Layout

```
features.py        log-mel extraction, scipy only
model.py           the network
prepare.py         patient-level manifest
train.py           training, patient-level split
evaluate.py        the numbers above
export.py          ONNX export, with an equivalence assertion
web/dsp.js         the JavaScript port of features.py
verify_dsp.*       proof the two agree
web/               the screening console
docs/              sources, one-page summary, deck, testing instructions
```

## Data

CirCor DigiScope, Oliveira et al., *IEEE J Biomed Health Inform.* 2022;26(6):2524-2535, via
PhysioNet under ODC-By 1.0. The corpus is **not** redistributed here; `prepare.py` expects it
under `data/`. The five demo recordings in `web/samples/` are from the corpus, are all held-out
test patients, and are included under that licence with attribution.

Full citations, including two figures corrected after checking them against the source, are in
[`docs/sources.md`](docs/sources.md).
