# Research Statement

**Status:** proposed — awaiting sign-off from Tanay, Ria, Bhavika
**Date:** 2 November 2026
**Supersedes:** the original classification + Grad-CAM proposal (August 2026)

---

## Research question

> When a brain-tumor segmentation model is poisoned during training, can its
> output be corrupted in ways that remain clinically plausible to expert
> review — and can the system autonomously detect, localise, and repair that
> corruption faster than full retraining?

## Summary

Deep learning models are increasingly used to segment brain tumors from MRI,
and the resulting masks directly inform surgical margins and radiation
planning. A poisoned model does not fail loudly. It produces masks that remain
smooth, connected, and anatomically sited, while their boundaries are
systematically wrong — a failure a reviewing clinician has no obvious way to
catch.

We poison a U-Net's training data, measure how far segmentation accuracy
degrades, and independently measure whether the corrupted output still *looks*
acceptable — both computationally and by asking practising radiologists whether
they would accept each mask for treatment planning. We then measure whether the
system can detect that it was attacked with no human in the loop, identify
which training samples carry the corruption, excise them, and recover — and how
that recovery time compares to retraining from scratch.

---

## Contributions

| # | Contribution | Owner |
|---|---|---|
| 1 | A poisoning threat model for medical image segmentation, and attacks that degrade accuracy while preserving plausibility | Tanay |
| 2 | A plausibility metric — quantifying when a wrong mask still looks right — validated against radiologist judgement | Tanay + Ria |
| 3 | Autonomous detection and localisation of poisoned training samples via a corruption ledger | Ria |
| 4 | Targeted excision and repair, benchmarked against full retraining | Bhavika |

## Reported metrics

- **Dice / IoU** before and after poisoning, across poison rates
- **Plausibility score** — connectivity, boundary smoothness, anatomical siting
- **Silent-failure rate** — proportion of poisoned outputs with low Dice but
  high plausibility. *This is the headline number.*
- **Clinician acceptance rate** — percentage of corrupted masks radiologists
  would accept for treatment planning
- **T_detect, T_localise, T_recover** — autonomous recovery timings, fixed hardware
- **Recovery cost** — targeted excise-and-repair versus full retraining

---

## Scope

### In scope

- U-Net binary tumor segmentation on BraTS 2018/2019 2D slices
- Training-time data poisoning (mask corruption)
- Computational plausibility metrics
- A small radiologist rating study (2–3 clinicians, ~30 masks)
- Autonomous detection, localisation, excision, repair
- Recovery-time benchmarking against full retraining

### Out of scope

The following came from the August faculty review and are **deferred to future
work**. They are not achievable in the remaining time and we should not imply
otherwise.

- LLM interpretation layer
- Knowledge graph construction
- Graph of Thoughts reasoning
- Multi-model ensembling for hallucination reduction
- RAG evidence retrieval
- Test-time evasion attacks (FGSM / PGD)
- Grad-CAM — not meaningful for a segmentation model, whose output is already
  a spatial map

---

## How this relates to the original proposal

The original pitch asked whether a classifier's saliency explanation could stay
convincing while its diagnosis was corrupted — the **silent failure** question.

That question survives, in a form the current data and code can actually
support:

| Original (classification) | Now (segmentation) |
|---|---|
| Diagnosis flips | Mask boundary shifts |
| Heatmap still looks reasonable | Mask still looks like a competent segmentation |
| Clinician trusts the result | Surgeon plans the resection around it |
| **Silent failure** | **Silent failure** |

What changed, and why:

1. **Task: classification → segmentation.** Our BraTS 2D dataset has
   pixel-level masks and no class labels. Deriving grade labels would give
   285 patients at 210/75 — too small and too imbalanced for a credible
   classification result. The same data gives 12,100 labelled slices for
   segmentation.
2. **Attack: evasion → poisoning.** Training-time poisoning is the attack the
   self-healing question is actually about: you cannot "repair" a model from a
   test-time perturbation, but you can detect and remove poisoned training data.
3. **Explanation: Grad-CAM → plausibility of the output mask.** A segmentation
   model's output *is* its spatial explanation, so plausibility of the mask is
   the direct analogue of a convincing heatmap.

---

## Working title

**Silent Corruption: Clinically Plausible Failures in Poisoned Brain-Tumor
Segmentation, and Autonomous Recovery**

---

## Sign-off

- [ ] Tanay
- [ ] Ria
- [ ] Bhavika
- [ ] Reviewed with faculty supervisor
