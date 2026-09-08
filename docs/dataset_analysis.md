# Dataset Analysis — BraTS 2018/2019 2D

Source: `data/raw/brats`

## 1. Pairing integrity

- Images: **12100**
- Masks: **12100**
- Matched pairs: **12100**
- Images without a mask: **0**
- Masks without an image: **0**

## 2. Patients and slices

- Unique patients: **285**
- Slices per patient — min **6**, median **42**, mean **42.5**, max **78**
- Slice index range: 0 to 77
- Filenames that failed slice parsing: **0**

Patients with the fewest slices:

```
  Brats18_TCIA08_406_1: 6
  Brats18_TCIA02_171_1: 7
  Brats18_CBICA_BHB_1: 11
  Brats18_CBICA_ANP_1: 12
  Brats18_TCIA01_411_1: 12
```

## 3. Dimensions

Image dimensions:

```
  240x240: 12100
```
- All images 240x240: **True**
- Image/mask dimension mismatches: **0**

## 4. Colour modes

- Image modes: `{'L': 12100}`
- Mask modes: `{'L': 12100}`

## 5. Mask pixel values

- Unique values across all masks: `[0, 255]`
- **Binary masks** — classes already collapsed to background/tumor.

## 6. Tumor content

- Completely empty masks: **0** (0.00%)
- Slices containing tumor: **12100**
- Tumor fraction (non-empty) — min **0.00002**, median **0.00538**, mean **0.01008**, max **0.07316**
- Slices with <0.1% tumor pixels: **2511** (candidates for a minimum-size filter)

## 7. Duplicates

- Duplicate image hashes: **0**

## 8. Load failures

- Files that failed to open: **0**
