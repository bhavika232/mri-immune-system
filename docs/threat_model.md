# Threat Model

**Owner:** Tanay
**Version:** 1 (2 November 2026)
**Scope:** Phase 1 (A1) — poisoning the segmentation model.
Phase 2 attacks (A2) and the coordinated cascade (A3) are specified separately.

---

## 1. Scenario

Segmentation training data is only as trustworthy as its annotations. In
practice tumour masks are produced by contracted annotators, junior clinicians,
semi-automated tools, or crowdsourced labelling pipelines — rarely by the team
that trains the model. An adversary who compromises any part of that annotation
chain can alter labels without ever touching a scan, a server, or the training
code.

We model that adversary.

## 2. Attacker goal

Degrade the deployed model's segmentation accuracy **while keeping its outputs
clinically plausible**, so that the corruption is not caught by the human who
reviews the mask before treatment planning.

Stated as an objective: maximise Dice degradation subject to the plausibility
score remaining within the range of clean model outputs.

An attack that makes the model output obvious garbage is a failure by this
definition — it would be caught immediately. **Stealth is the objective, not
destruction.** This is what distinguishes our threat model from standard
poisoning work.

## 3. Attacker capabilities

### CAN

- Modify **segmentation masks** in the training set
- Select which training samples to modify, up to a poison budget
- Choose the corruption type and magnitude per sample
- Know the model architecture and training procedure (white-box)

### CANNOT

- Modify training **images** — only the annotations
- Touch the **validation or test sets** in any way
- Modify model architecture, loss function, optimiser or hyperparameters
- Access or alter the trained weights directly
- Interfere at inference time
- Modify the corruption ledger or any detection machinery

Rationale: the attacker owns the annotation pipeline and nothing else. This is
the narrowest capability set that still produces a realistic and interesting
attack, which makes the result stronger — the damage is achieved with minimal
access.

## 4. Poison budget

Fraction of training samples whose masks are corrupted. Swept, not fixed:

```
poison_rate in {0.01, 0.05, 0.10, 0.20}
```

Reporting the full sweep matters more than any single rate. The interesting
question is where the curves cross — the point at which Dice has fallen
meaningfully while plausibility has not.

## 5. Poisoning methods

Two stealthy attacks, plus one control.

### 5.1 Boundary shift — `shift` (primary)

Translate the mask by a small offset in a random direction.

```
offset ~ uniform(3, 10) pixels, random direction
```

The mask stays a single smooth connected region of the correct size, sitting on
or near a real lesion. Visually it looks like a competent segmentation. The
boundary is simply in the wrong place.

**Clinical meaning:** a resection or radiation field planned around the wrong
margin.

### 5.2 Dilate / erode — `grow_shrink` (primary)

Expand or contract the mask by a morphological radius.

```
radius ~ uniform(2, 6) pixels, dilate or erode chosen at random
```

Produces a perfect-looking blob in exactly the right place that systematically
over- or under-segments.

**Clinical meaning:** over-segmentation removes healthy tissue;
under-segmentation leaves tumour behind.

### 5.3 Blank mask — `blank` (control, not a proposed attack)

Set the mask to all background.

Included **only as a contrast condition.** It is crude and trivially detectable,
and exists to demonstrate what an obvious attack looks like so the subtle ones
can be measured against it. We do not claim it as a contribution.

### Shared constraint

Every poisoned mask must remain **binary** and **non-empty** (except `blank`).
A poisoned mask containing interpolated or fractional values would be detectable
by a trivial type check and would not represent a realistic annotation error.

## 6. Attack success criteria

An attack succeeds when **both** hold:

1. **Effective** — model Dice on the clean test set drops measurably
2. **Stealthy** — the poisoned model's outputs score within the clean model's
   plausibility range, and are accepted by radiologists at a comparable rate

Condition 2 is what makes this our paper rather than a standard poisoning study.

## 7. Defender assumptions

The defender:

- Does **not** know which samples were poisoned
- Does **not** know the poison rate or the corruption type
- **Does** have access to the full training set and the trained model
- Must detect, localise and repair **without human intervention**

Recovery time is measured from the moment corrupted behaviour becomes
detectable, on fixed hardware, and compared against full retraining from clean
data.

## 8. Out of scope

- Test-time evasion attacks (FGSM, PGD) — the attack here is training-time
- Image-space perturbation — the attacker cannot touch scans
- Backdoor / trigger-based attacks — no trigger pattern is implanted; the
  corruption is indiscriminate rather than input-conditional
- Attacks on the reasoning layer — specified separately as A2

## 9. Implementation

`src/poison.py` exposes:

```python
poison_mask(mask, method, rng, **params) -> corrupted_mask
poison_dataset(manifest, method, rate, seed) -> (poisoned_manifest, ledger)
```

Every poisoned sample is recorded in the corruption ledger, keyed by
`sample_id`, with its method and parameters. The ledger is ground truth for
evaluating detection and localisation — it is never visible to the defender.
