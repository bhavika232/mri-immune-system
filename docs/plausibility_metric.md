# The Plausibility Metric

**Owner:** Tanay
**Code:** `src/plausibility.py`
**Version:** 1 (2 November 2026)

---

## 1. Why this exists

Dice answers **"is this mask correct?"**

This metric answers a different question: **"does this mask look correct?"**

Those come apart, and the gap between them is the entire paper. A mask can be
badly wrong by Dice while still being a smooth, properly sized region sitting
on a visibly abnormal part of the brain — exactly the kind of thing a clinician
accepts without a second look.

| | Dice | Plausibility |
|---|---|---|
| Asks | Is it right? | Does it look right? |
| Needs ground truth | Yes | **No** |
| What it catches | Any error | Only *visible* error |

That second row matters. In deployment nobody has the ground truth — if they
did, they would not need the model. All a reviewing clinician has is the mask
and the scan. Plausibility measures what *they* can see.

**The silent-failure rate** — our headline number — is the proportion of
outputs with **low Dice and high plausibility**. Wrong, and undetectably so.

---

## 2. The six features

Each is something a radiologist notices without consciously measuring it.

| Feature | What it measures | Why it matters |
|---|---|---|
| `n_components` | How many separate pieces the mask has | Scattered specks look like noise, not a lesion |
| `largest_fraction` | Share of the mask in its biggest piece | Distinguishes one mass with a speck from genuine fragmentation |
| `circularity` | `4·pi·area / perimeter²` | 1.0 is a perfect circle; falls towards 0 as the boundary gets jagged or the shape stretches |
| `solidity` | `area / convex_hull_area` | A filled shape vs a ring or lace pattern |
| `brain_containment` | Fraction of mask inside brain tissue | A mask floating in the black background is obviously fake |
| `lesion_contrast` | Brightness inside the mask vs a ring just outside | A plausible mask sits on something that *looks* abnormal |

The shape features are computed on the **largest connected component**, so that
a stray speck elsewhere in the image does not drag down the description of the
tumour itself.

`brain_containment` and `lesion_contrast` need the MRI image, not just the mask.
The other four are pure geometry.

---

## 3. How scoring works — and why it is calibrated

The naive approach is to hand-pick thresholds: *"circularity below 0.5 is
implausible."* That is indefensible in review — the number is invented, and a
reviewer will ask where it came from.

Instead we **calibrate against the real expert annotations**:

1. Compute all six features over the ground-truth masks (3,000 sampled)
2. Record the 5th percentile, median, 95th percentile and IQR of each
3. Score a new mask by **how typical it is within that distribution**

A feature sitting inside the clean 5th–95th band scores **1.0**. Outside it,
the score falls off linearly with distance from the band, measured in units of
the clean interquartile range so every feature is on a comparable scale.
Overall plausibility is the mean of the six.

So "plausible" is defined as **"indistinguishable from a real annotation"**,
which is exactly the claim the paper needs and requires no invented constants.

### One-sided features

For five of the six, only a *low* value is suspicious. A mask that is unusually
smooth or unusually well-contained is not implausible — it is just tidy. So
those are scored one-sided: above the clean band is never penalised.
`n_components` is two-sided, since both too few and too many pieces are odd.

---

## 4. What calibration revealed

```
feature                5th pct    median    95th pct
---------------------------------------------------
n_components             1.000     5.000     25.000
largest_fraction         0.298     0.895      1.000
circularity              0.165     0.576      1.000
solidity                 0.563     0.833      1.000
brain_containment        1.000     1.000      1.000
lesion_contrast          0.039     0.503      1.744
```

Two findings worth keeping:

**Real masks have a median of 5 connected components, not 1.** The intuition
that "a tumour is one mass" is wrong for 2D slices — a plane cut through a 3D
tumour frequently shows several disconnected pieces, especially with a
ring-enhancing lesion. Had we hand-picked `n_components == 1` as the
plausibility rule, we would have scored most *real* annotations as fake.
This is the clearest argument for calibrating rather than guessing.

**`brain_containment` is 1.000 at the 5th percentile.** Essentially every real
annotation lies entirely inside the brain. That makes it an unusually sharp
discriminator: any value below 1.0 is already unusual.

---

## 5. Results at 10% poisoning

Mean per-feature score, clean vs poisoned, on the same 841 samples.

### 5.1 First attempt — unconstrained corruption

| Feature | Clean | `shift` | `grow_shrink` |
|---|---|---|---|
| n_components | 0.973 | 0.973 | **1.000** |
| largest_fraction | 0.993 | 0.993 | **0.999** |
| circularity | 0.995 | 0.995 | **1.000** |
| solidity | 0.977 | 0.977 | **0.995** |
| brain_containment | 0.966 | **0.748** | **0.805** |
| lesion_contrast | 0.999 | 0.998 | 0.998 |
| **overall** | **0.984** | **0.947** | **0.966** |
| **median** | **1.000** | **1.000** | **1.000** |

`brain_containment` was the only feature that detected anything: an
unconstrained shift or dilation can push part of the mask past the brain edge
into the black background, which is the one thing that makes a mask visibly
fake. So we constrained the corruptions to stay inside the brain.

### 5.2 After the anatomical constraint

| Feature | Clean | `shift` | `grow_shrink` |
|---|---|---|---|
| n_components | 0.973 | 0.973 | **1.000** |
| largest_fraction | 0.993 | 0.993 | **0.999** |
| circularity | 0.995 | 0.995 | **1.000** |
| solidity | 0.977 | 0.977 | **0.995** |
| brain_containment | 0.966 | **1.000** | **0.999** |
| lesion_contrast | 0.999 | 0.998 | 0.998 |
| **overall** | **0.984** | **0.989** | **0.998** |
| **median** | **1.000** | **1.000** | **1.000** |

Mean Dice between clean and poisoned masks was unchanged by the constraint
(0.424 to 0.426 for `shift`, 0.434 to 0.438 for `grow_shrink`), so no attack
strength was traded away for the added stealth.

### Reading these numbers

**Both attacks now score higher than the real expert annotations.** On every
feature the poisoned masks are at least as typical as genuine annotations, and
on several they are more so. A geometry-based detector tuned to flag these
would flag the real annotations first.

**Median plausibility is 1.000 throughout.** More than half of all poisoned
masks are perfectly typical of real annotations on every measure — not merely
hard to detect, but indistinguishable.

**`shift` leaves five of six features bit-identical.** A pure translation
preserves shape, size and smoothness exactly. Only position changes, and
position cannot be checked without the ground truth that is unavailable at
deployment.

**`grow_shrink` scores higher than ground truth on four features.** Dilation
merges the fragmented pieces of a real 2D tumour slice into one smooth blob, so
the corrupted mask looks *more* like a textbook tumour than the expert
annotation does. The attack does not merely hide — it improves apparent quality
while destroying correctness.

**`brain_containment` is now better than human.** Clean annotations score
0.966; the constrained attack scores 1.000. The rejection loop guarantees
containment, whereas real annotators occasionally place a pixel outside the
brain boundary. The corruption is more anatomically disciplined than the
radiologists who produced the training data.

---

## 6. What this tells us

**Appearance-based defence does not work for this attack class.** That is a
result, not a failure, and it is one of the paper's central claims. If geometry
cannot separate corrupted masks from genuine ones — and it cannot, since the
corrupted ones score better — then the defence has to come from somewhere else:
training-dynamics anomalies, cross-model disagreement, influence functions over
the training set, or the reasoning layer noticing an inconsistency the geometry
does not expose.

This is what motivates the rest of the project.

---

## 7. Known limitations

- **Area is computed but not scored.** A grown mask is larger, but still a
  plausible tumour size, so scoring area would mostly not penalise it. We
  record it for analysis rather than folding it into the score. Worth
  revisiting if a reviewer asks.
- **`brain_mask` is a simple intensity threshold.** Fine on BraTS, which is
  skull-stripped with a true-zero background. It would need replacing on data
  that is not skull-stripped.
- **These are proxies for human judgement, not human judgement.** That is what
  the radiologist rating study is for — it validates whether these six numbers
  actually track what a clinician would accept. Until then, plausibility is a
  computational stand-in.
- **The attacker and the metric share a definition of "brain".** `poison.py`
  constrains corruption using the same `brain_mask()` function the metric
  scores with, so perfect containment is partly by construction. This is
  consistent with our white-box threat model — a simple intensity threshold is
  not privileged knowledge — but it should be stated plainly, and a defender
  using a different brain segmentation might see a small residual signal.
- **Calibrated on 3,000 of 12,100 masks** for speed. Re-run `--fit` without
  `--limit` for the final numbers in the paper.

---

## 8. How to run it

```bash
# build the reference distribution from the clean expert annotations
python src/plausibility.py --fit

# score a poison run (clean vs poisoned, same samples)
python src/plausibility.py --run shift_rate0.1_seed42
```

In code:

```python
from src.plausibility import PlausibilityScorer

scorer = PlausibilityScorer.load()
result = scorer.score(mask, image)

result["plausibility"]   # overall, 0-1
result["scores"]         # per-feature
result["features"]       # raw measurements
```

The reference lives in `data/processed/plausibility_reference.json`. Rebuild it
if the dataset changes; it is specific to the annotation distribution it was
fitted on.

---

## 9. Where this goes in the paper

Plausibility is one axis of the headline result. Once a model is trained, every
prediction gets both numbers:

```
                    looks broken        looks plausible
                 +-------------------+-------------------+
   Dice high     |       fine        |       fine        |
                 +-------------------+-------------------+
   Dice low      |    detectable     |  SILENT FAILURE   |
                 +-------------------+-------------------+
```

The bottom-right cell is what we report, and this module computes the column.
