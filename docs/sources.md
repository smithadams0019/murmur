# Murmur — sources

Every factual claim, with where it came from. Each was read from the primary source or the
publisher's own abstract, not from secondary reporting. Figures were checked against the source
rather than written from memory — two were wrong on the first pass and are noted at the bottom.

## The disease

> "Rheumatic heart disease is the most commonly acquired heart disease in people under age 25."
>
> "Rheumatic heart disease affects an estimated 55 million people worldwide and claims
> approximately 360 000 lives each year – the large majority in low- or middle-income
> countries."

WHO, *Rheumatic heart disease* fact sheet.
https://www.who.int/news-room/fact-sheets/detail/rheumatic-heart-disease

The same fact sheet sets out why early detection matters: the damage comes from repeated
episodes of rheumatic fever following group A streptococcal infection, and further episodes are
prevented with benzathine penicillin G "given by intramuscular injection every 3–4 weeks over
many years". The treatment is cheap and old. Finding the child is the hard part.

## Listening misses most of it

Marijon E, Ou P, Celermajer DS, et al. **Prevalence of rheumatic heart disease detected by
echocardiographic screening.** *N Engl J Med.* 2007 Aug 2;357(5):470-6.
doi:10.1056/NEJMoa065085. PMID 17671255.

Schoolchildren aged 6–17 in Cambodia and Mozambique, screened both clinically and by
echocardiography:

| | Clinical examination | Echocardiography |
|---|---|---|
| Cambodia (n=3,677) | 2.2 per 1,000 (95% CI 0.7–3.7) | **21.5 per 1,000** (95% CI 16.8–26.2) |
| Mozambique (n=2,170) | 2.3 per 1,000 (95% CI 0.3–4.3) | **30.4 per 1,000** (95% CI 23.2–37.6) |

The authors' own conclusion: echocardiographic screening "reveals a much higher prevalence of
rheumatic heart disease (approximately 10 times as great)", and "secondary prevention may be
effective after accurate identification of early cases".

This is the whole argument for Murmur. A stethoscope in trained hands found roughly one case in
ten. The knowledge of what to do next is not missing; the child never reaches the echo.

## The data this was trained on

Oliveira J, Renna F, Costa PD, et al. **The CirCor DigiScope Dataset: From Murmur Detection to
Murmur Classification.** *IEEE J Biomed Health Inform.* 2022 Jun;26(6):2524-2535.
doi:10.1109/JBHI.2021.3137048. PMID 34932490.

Distributed via PhysioNet under the Open Data Commons Attribution Licence (ODC-By 1.0).
942 patients, heart sounds recorded at multiple auscultation sites, with murmur annotations
made by an expert annotator. Not redistributed in this repository.

## What Murmur itself measures

Produced by `train.py` and `evaluate.py` in this repository; reproduce with `python evaluate.py`.

- Split by **patient**, never by recording, so no patient contributes to both training and test.
- Held-out test set: **174 patients**, none seen in training or validation.
- **Test AUC 0.957.**
- At the shipped operating point (threshold 0.9455, chosen for sensitivity):
  **sensitivity 94.3%, specificity 77.0%** — 2 murmurs missed of 35, and 32 children referred
  who did not need it.
- Patients annotated "Unknown" in the corpus are excluded from training and evaluation rather
  than folded into either class.

The threshold is deliberately not the one that maximises accuracy. A missed murmur sends a
child home; a false alarm sends a child for an echo. Those are not the same mistake and the
operating point says so.

## What is deliberately not claimed

- **Murmur does not diagnose rheumatic heart disease, or anything else.** It decides who should
  be sent for an echocardiogram. The 55 million and the 360,000 above describe the disease, not
  this model's effect on it.
- **No claim that Murmur improves any outcome.** It has never been deployed or trialled. The
  Marijon trial establishes that screening finds cases clinical examination misses; it says
  nothing about this software.
- **No claim about performance outside the corpus.** CirCor was recorded in one screening
  programme in Brazil with particular equipment. Nothing here characterises behaviour on other
  hardware, other populations, or in a noisy clinic.
- The 2 missed murmurs are reported as 2 missed murmurs, on the same page as the AUC.

## Corrections made while writing this

- An earlier draft carried "40 million people living with RHD and 300,000 deaths a year" from
  memory. The WHO fact sheet says **55 million** and **approximately 360,000**. Corrected.
- The Marijon paper was first looked up under the wrong PMID, which returned an unrelated
  paper on HPV genotyping. Found by title and author search instead, and the numbers above are
  from the correct record (PMID 17671255).
