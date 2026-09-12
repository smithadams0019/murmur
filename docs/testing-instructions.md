No sign-up. No credentials. Nothing is uploaded — the model runs in your browser.

Open https://d2ajaodgsqzl9.cloudfront.net

**The samples are all held-out test patients.** None of them was seen during
training. Each one ships the score Python produced during evaluation, and the
page shows it beside the score your browser just computed. If those two ever
disagreed, the published accuracy would be describing different software.

**A referral.** Open the first sample. You get the score, the threshold it
cleared, and the four-second segment that drove the decision — so you can listen
to the same audio the model reacted to and disagree with it.

**A missed murmur.** One of the two murmurs the model does not catch is in the
sample list deliberately. It is the most useful thing on the site. A screening
tool that only demonstrates its successes is a sales pitch.

**The operating point.** The threshold is 0.9455, chosen for sensitivity rather
than accuracy: 94.3% sensitivity, 77.0% specificity, 2 murmurs missed of 35, 32
children referred who did not need it. A higher threshold would make the numbers
look better and send two more children home.

To reproduce the metrics: `python evaluate.py` in the repository root. Needs the CirCor
DigiScope dataset from PhysioNet under `data/` — it is not redistributed here.

Murmur does not diagnose anything. It decides who should be sent for an
echocardiogram.
