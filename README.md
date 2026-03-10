# SoFT 

This is the official repository of the [AAAI 2026 Workshop paper](https://arxiv.org/abs/2512.20781) **"Soft Filtering: Guiding Zero-shot Composed Image Retrieval with Prescriptive and Proscriptive Constraints"**.

## SoFT module

<p align="center">
  <img src="fig/main.png" alt="SoFT main figure" />
</p>

> Figure 1: Overview of SoFT, a plug-and-play soft filtering module for Zero-shot CIR. Given a reference image and a modification text, multimodal LLMs extract prescriptive and proscriptive constraints. These are used to softly reward or penalize candidate images using CLIP similarity.

## Getting Start

### Download Datasets
Place each dataset inside the repository root under datasets/ exactly as shown below.

- [CIRCO Official Repository](https://github.com/miccunifi/CIRCO)
- [CIRR Official Repository](https://github.com/Cuberick-Orion/CIRR)
- [FashionIQ Official Repository](https://github.com/XiaoxiaoGuo/fashion-iq)

```
datasets
├── CIRCO
│   ├── annotations
|   |   ├── [val | test].json
│   ├── COCO2017_unlabeled
|   |   ├── annotations
|   |   |   ├──  image_info_unlabeled2017.json
|   |   ├── unlabeled2017
|   |   |   ├── [000000243611.jpg | 000000535009.jpg | ...]
│
├── CIRR
│   ├── cirr
│   ├── dev
|   |   ├── [dev-1-0-img1.png | dev-1-3-img1.png | ...]
│   ├── test1
|   |   ├── [test1-0-0-img0.png | test1-0-1-img1.png | ...]
│   ├── train
|   |   ├── [0 | 1 | 2 | 3 | ...]
│
├── FASHIONIQ
│   ├── captions
|   |   ├── [cap.dress.test.json | cap.dress.train.json | ...]
│   ├── image_splits
|   |   ├── [split.dress.test.json | split.dress.train.json | ...]
│   ├── images
|   |   ├── [245600258X.png | 978980539X.png | ...]
```

### Construct MultiTarget Triplets
