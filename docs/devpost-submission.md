# Murmur — Devpost submission

## Form fields

**Project name:** Murmur
**Tagline:** Decides which children need an echocardiogram, and says out loud what it missed.
**Live demo:** https://d2ajaodgsqzl9.cloudfront.net
**Repository:** https://github.com/smithadams0019/murmur
**One-page PDF:** docs/onepager.pdf
**Video:** (under 3:00)
**Built with:** PyTorch, ONNX Runtime Web, scipy, S3, CloudFront

---

## Elevator pitch

Rheumatic heart disease is preventable with an antibiotic that has been cheap for seventy
years. What is missing is not the cure and not the knowledge — it is the echocardiogram that
tells you which child needs it. Murmur listens to a heart-sound recording in the browser and
decides who to send. It is tuned to miss as few murmurs as possible, and it reports the ones it
missed on the same screen as the score.

---

## The story

Screening schoolchildren in Cambodia and Mozambique, doctors examined them the usual way —
stethoscope, trained ears — and found rheumatic heart disease in 2.2 and 2.3 children per
thousand. Then they scanned the same children with echocardiography and found 21.5 and 30.4 per
thousand.

Roughly ten times as many. Listening had found one case in ten.

Those are not children who were failed by a lack of medicine. Rheumatic heart disease follows an
untreated strep throat, and further damage is prevented with benzathine penicillin G — an
injection every three or four weeks. The drug is not the problem. The problem is that the disease
is silent until a valve is wrecked, the only reliable way to see it early is an echo, and echo is
scarce exactly where the disease is common.

An echo machine costs what it costs. A phone is already in the room.

So Murmur does not try to be the echo. It tries to be the thing that decides who gets one.

## What it actually does

It takes a heart-sound recording, cuts it into four-second segments, turns each into a log-mel
spectrogram, and runs a small four-block convolutional network over them in the browser. The
per-patient score is the loudest segment, not the average — a murmur present in one position and
not another is still a murmur.

Then it says one of two things: **refer for echocardiography**, or **no murmur detected**. It
never says "rheumatic heart disease", because it cannot know that, and neither can a murmur.

## The numbers, including the bad one

Trained and evaluated on the CirCor DigiScope dataset (942 patients, PhysioNet, ODC-By), split
**by patient** so that no child appears in both training and test.

| | |
|---|---|
| Held-out test patients | 174 |
| Test AUC | **0.957** |
| Sensitivity at the shipped threshold | **94.3%** |
| Specificity | 77.0% |
| Murmurs missed | **2 of 35** |
| Children over-referred | 32 |

The threshold is not the one that maximises accuracy. A higher one would have made the table
look better and sent two more children home. Missing a murmur and raising a false alarm are not
the same mistake: one costs a child a valve, the other costs an ultrasound appointment. The
operating point is set accordingly, and the two missed murmurs are printed next to the AUC
rather than left in a confusion matrix nobody opens.

## Three things in the demo worth clicking

**A referral.** The score, the threshold it cleared, and the four-second segment that drove it —
so a clinician can listen to the same audio the model reacted to and disagree.

**A miss.** One of the two murmurs the model did not catch, shown deliberately. If a screening
tool only demonstrates its successes, the demo is a sales pitch.

**The same recording, scored twice.** Every sample ships the score Python produced during
evaluation. The page shows it beside the score your browser just computed. They match, which is
the only reason to believe the published accuracy describes the thing you are running.

## Why the browser

The model is 575 KB and runs on the device. No audio of a child's chest is uploaded anywhere,
there is no server to pay for, and the page works on a phone over a bad connection. Deployed as a
static site on S3 behind CloudFront — HTTPS is needed for microphone access anyway.

## Challenges

**A bandpass filter I had to delete.** The Python pipeline used a Butterworth filter I could not
reproduce bit-for-bit in JavaScript. A model that sees slightly different features in the browser
than it saw in training is worse than a slightly less accurate model, because you no longer know
what it is doing. I removed the filter and retrained without it.

**A stale model that nearly shipped.** I retrained and forgot to re-export the ONNX. The browser
was running the previous model, against features it was not trained on, at a threshold from a
different run. It was caught because the shipped sample scores no longer matched Python's. That
check now exists precisely because it caught this.

**A metrics file that quietly disagreed with itself.** `model_meta.json` carries a `metrics.test`
block computed at one threshold and a `chosen_operating_point` at the one actually shipped. Read
the wrong field and the page reports 88.6% sensitivity while the model runs at 94.3%. Both are
true of different thresholds and only one describes what the user is using.

**Demo samples from the training set.** The first set of examples was picked for sounding clear.
Some were patients the model had trained on. They were re-picked from held-out test patients only.

## What it does not do

It does not diagnose rheumatic heart disease. A murmur is a sound, not a disease, and plenty of
children have innocent ones.

It has never been deployed or trialled. The screening evidence above says echo finds what
listening misses; it says nothing about this software.

It is not characterised outside its corpus. CirCor was recorded in one screening programme with
particular equipment. A noisy clinic, a different stethoscope, a different population — none of
that is measured here, and claiming otherwise would be the easiest lie in the project.

## What I learned

That the honest version of a screening tool is mostly an argument about which mistake to make.
Once you accept that the two errors are not interchangeable, the threshold stops being a tuning
knob and becomes the actual design decision — and then you have to be willing to print the cost.

## What's next

Recording quality assessment before scoring, so the tool can refuse a recording rather than score
noise. Position-aware aggregation, since the corpus records four auscultation sites and currently
they are pooled. And validation on a corpus that is not CirCor, which is the only thing that would
justify any claim about the field.

## Try it

https://d2ajaodgsqzl9.cloudfront.net — no sign-up, nothing to install. Full testing instructions
in `docs/testing-instructions.md`. Every source in `docs/sources.md`.
