# Research Statement

**Status:** proposed — awaiting sign-off from Tanay, Ria, Bhavika
**Version:** 2 (2 November 2026)
**Supersedes:** v1 (segmentation-only scope), and the original classification +
Grad-CAM proposal (August 2026)

---

## Research question

> When a brain-tumor segmentation system — a U-Net and the reasoning layer that
> interprets its output — is attacked at either stage or both, can its results
> be corrupted in ways that remain clinically plausible to expert review, and
> can the system autonomously detect, localise, and repair that corruption
> faster than full retraining?

## Summary

Deep learning models segment brain tumors from MRI, and the resulting masks
directly inform surgical margins and radiation planning. Increasingly those
masks are not read raw: a reasoning layer turns them into structured findings
and a clinical interpretation a human actually reads.

Both stages can be attacked, and neither fails loudly. A poisoned U-Net
produces masks that stay smooth, connected and anatomically sited while their
boundaries are systematically wrong. A compromised reasoning layer produces a
fluent, evidence-cited interpretation of a tumor that is not there. Attacked
together, the corruption compounds: a wrong mask feeds a clean reasoning layer,
which describes it confidently and makes the error *more* convincing, not less.

We measure how far each stage degrades under attack, whether the corrupted
output still *looks* acceptable — computationally and by asking radiologists
whether they would accept it for treatment planning — and whether the system
can detect the attack with no human in the loop, identify the corrupted
components, excise them, and recover faster than retraining from scratch.

---

## System under test

```
MRI slice
    |
    v
PHASE 1   U-Net  ->  binary tumour mask
    |
    v
          structured findings
          (location, volume, shape, topology, midline relation)
    |
    v
PHASE 2   knowledge graph
          -> Graph of Thoughts (clinical if-then rules)
          -> RAG (guideline-grounded evidence)
          -> multi-model cross-check (Claude + Gemini + open-weight)
    |
    v
          clinical interpretation  ->  human review
```

Everything above is the system. The contributions below are what we do to it.

## Attack conditions

| | Target | Method | Access |
|---|---|---|---|
| **A1** | Phase 1 | Training-data poisoning — corrupt segmentation masks | White-box, we own the weights |
| **A2** | Phase 2 | RAG corpus poisoning, knowledge-graph edge poisoning, prompt and thought injection | Black-box for API models; white-box layer probing on the open-weight model only |
| **A3** | Both | Coordinated — corrupted mask flows into a clean reasoning layer, which launders it into a confident interpretation | Combined |

A3 is the condition with no prior literature, and the strongest claim in the paper.

---

## Contributions

| # | Contribution | Owner |
|---|---|---|
| 1 | A poisoning threat model for medical image segmentation, and attacks that degrade accuracy while preserving plausibility | Tanay |
| 2 | A plausibility metric — quantifying when a wrong mask still looks right — validated against radiologist judgement | Tanay |
| 3 | Autonomous detection and localisation of poisoned training samples via a corruption ledger | Tanay |
| 4 | A reasoning layer over segmentation output: knowledge graph, Graph of Thoughts, RAG grounding, multi-model cross-check | Ria |
| 5 | Attacks on the reasoning layer, and the A3 coordinated cascade | Ria + Tanay |
| 6 | Targeted excision and repair, benchmarked against full retraining | Bhavika |

## Reported metrics

**Phase 1**
- Dice / IoU before and after poisoning, across poison rates
- Plausibility score — connectivity, boundary smoothness, anatomical siting
- **Silent-failure rate** — low Dice with high plausibility. *Headline number.*
- Clinician acceptance rate — % of corrupted masks accepted for treatment planning

**Phase 2**
- Graph edit distance, clean versus attacked reasoning graph
- Hallucination rate — claims unsupported by retrieved evidence
- Cross-model disagreement, as a candidate free attack detector
- Layer-wise activation divergence on the open-weight model

**A3**
- Compound corruption rate versus A1 and A2 alone
- Whether interpretation confidence *rises* as segmentation accuracy falls

**Recovery**
- T_detect, T_localise, T_recover — autonomous, on fixed hardware
- Targeted excise-and-repair versus full retraining
- Post-repair Dice retention and forgetting

---

## Ownership

| | Owns |
|---|---|
| **Tanay** | Data pipeline, threat model, poisoning attacks, corruption ledger, plausibility metric |
| **Ria** | All of Phase 2 — knowledge graph, Graph of Thoughts, RAG, multi-model aggregation, Phase 2 attacks |
| **Bhavika** | U-Net, training, detection triggers, excision, repair, recovery benchmark |

The corruption ledger moves to Tanay: whoever creates the corruption should log it.

---

## Sequencing — the one hard rule

**Phase 1 must be independently publishable before Phase 2 work begins.**

Concretely, that means all of:

- A trained U-Net with a reported baseline Dice
- Working poisoning attacks at several poison rates
- Measured degradation and the silent-failure rate

Once that exists as a standalone result, Phase 2 is pure upside. If Phase 2
stalls, we still have a paper. This is the guardrail that protects the project,
and it is not negotiable.

---

## Data

**Current:** BraTS 2018/2019 2D (IEEE DataPort repackage) — 12,100 slices,
285 patients, 240x240, single channel, binary masks.

Known limitations: single sequence where BraTS natively has four; tumour
sub-regions collapsed to binary; no healthy slices; third-party preprocessing
whose slice-selection criteria we cannot fully describe.

**Planned upgrades — additive, and explicitly non-blocking:**

1. **Medical Segmentation Decathlon Task01_BrainTumour** (free,
   medicaldecathlon.com). Four sequences (T1, T1ce, T2, FLAIR), official BraTS
   provenance we can cite, segmentation masks. Needs 3D NIfTI slicing code.
2. **UPENN-GBM** (TCIA, CC BY 4.0, 69 GB). 630 subjects, multi-parametric
   including advanced MRI, already skull-stripped and co-registered, segmentations
   corrected by board-certified neuroradiologists, plus clinical and radiomic data.

Migration is cheap by design: the pipeline reads `manifest.csv` and
`splits/split_v1.json`. Changing dataset means regenerating those two files.
`dataset.py`, the model, the attacks and the ledger are unaffected.

**Open question for the supervisor:** when physiological data was raised in the
August review, did that mean physiological MRI sequences (perfusion, diffusion,
spectroscopy) or patient physiological records? UPENN-GBM covers the former.
The latter does not help a segmentation task and would require adding a second
prediction task.

---

## Out of scope

- Test-time evasion attacks (FGSM / PGD) — the attack here is training-time poisoning
- Grad-CAM — not meaningful for a segmentation model, whose output is already a spatial map
- Classification — our data has masks, not class labels. Deriving grade labels
  gives 285 patients at 210/75, too small and too imbalanced to be credible.

---

## Relationship to the original proposal

The original pitch asked whether a classifier's saliency explanation could stay
convincing while its diagnosis was corrupted — the **silent failure** question.
That question survives intact, in a form the data and code actually support:

| Original (classification) | Now (segmentation + reasoning) |
|---|---|
| Diagnosis flips | Mask boundary shifts |
| Heatmap still looks reasonable | Mask stays smooth and plausible; interpretation stays fluent and cited |
| Clinician trusts the result | Surgeon plans the resection around it |
| **Silent failure** | **Silent failure, now with a second stage that amplifies it** |

---

## Working title

**Silent Corruption: Clinically Plausible Failures in Poisoned Brain-Tumor
Segmentation and Its Reasoning Layer, with Autonomous Recovery**

---

## Sign-off

- [ ] Tanay
- [ ] Ria
- [ ] Bhavika
- [ ] Reviewed with faculty supervisor
