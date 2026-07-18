# M2VG: Expression Satisfiability Modeling for Generalized Visual Grounding

M2VG is an anonymous research-code release for generalized visual grounding.
It addresses one-to-zero, one-to-one, and one-to-many referring expressions by
jointly predicting target localization and whether an expression is satisfiable
in the image.

## Overview

Conventional visual grounding methods primarily match local language and image
regions. This can produce false positives when a locally similar region exists
but the complete expression is not satisfied. M2VG formulates generalized visual
grounding as language-constraint verification: an expression is satisfiable only
when its category, attribute, spatial, and relational constraints are jointly
supported by the image.

<p align="center">
  <img src="asserts/m2vg_framework_overview.png" alt="M2VG framework overview" width="100%">
</p>

M2VG contains three main components:

- **Dynamic Focus Negative Mining (DFNM, training only):** constructs reliable
  unsatisfiable image-expression pairs through target removal and semantic
  mismatch, then supplies no-target/existence supervision.
- **Adaptive Word-aware Residual Modulation (AWRM):** aligns visual and word
  tokens, selectively modulates text-conditioned features through a residual
  gate, and produces enhanced tokens for grounding.
- **Existence branch:** aggregates top-scoring foreground queries to estimate an
  expression-level existence probability, enabling existence-aware inference
  that either returns boxes/masks or rejects an unsatisfied expression.

## Abstract

Generalized visual grounding extends conventional visual grounding to
one-to-zero, one-to-one, and one-to-many scenarios. It requires models to
localize referred targets and determine whether a language expression is
satisfiable by the image content. However, existing methods mainly rely on
target matching between text and image regions, making them vulnerable to the
target-presence assumption. They may predict incorrect regions when local
semantic cues match the expression but the global language constraints are not
satisfied.

To address this problem, we propose M2VG, an expression satisfiability modeling
framework for generalized visual grounding. M2VG reformulates the task as
language constraint verification, where an expression is satisfiable only when
its category, attribute, spatial, and relational constraints are fully supported
by object instances in the image. Specifically, we introduce dynamic focus
negative sample mining to construct reliable unsatisfiable image-expression
pairs through target removal and semantic mismatch. We further design an
adaptive word-aware residual modulation module and an auxiliary existence branch
to enhance image-conditioned token representations and predict expression-level
existence probability. Experiments show that M2VG achieves competitive
performance on generalized referring expression comprehension, generalized
referring expression segmentation, and conventional referring expression
localization. It improves target-absence recognition while maintaining strong
localization and segmentation performance.

**Keywords:** visual grounding; generalized visual grounding; referring
expression comprehension; referring expression segmentation; expression
satisfiability.

## Results

All values are percentages.

### gRefCOCO: generalized referring expression comprehension

| Model | Backbone | Val F1 | Val N-acc. | TestA F1 | TestA N-acc. | TestB F1 | TestB N-acc. | Avg. F1 | Avg. N-acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2VG | BEiT3-ViT-B | 75.27 | 76.35 | 68.94 | 74.39 | 59.38 | 67.31 | 67.86 | 72.68 |

### gRefCOCO: generalized referring expression segmentation

| Model | Backbone | Val gIoU | Val cIoU | Val N-acc. | TestA gIoU | TestA cIoU | TestA N-acc. | TestB gIoU | TestB cIoU | TestB N-acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2VG | BEiT3-ViT-B | 73.52 | 68.95 | 76.35 | 74.62 | 73.19 | 74.39 | 65.14 | 64.64 | 67.31 |

### Ref-ZOM: generalized referring expression segmentation

| Model | Backbone | oIoU | mIoU | Acc. |
| --- | --- | ---: | ---: | ---: |
| M2VG | BEiT3-ViT-B | 65.92 | 65.88 | 98.11 |

### RefCOCO, RefCOCO+, and RefCOCOg: referring expression comprehension

Precision at IoU 0.5 is the evaluation metric.

| Model | Backbone | Fine-tuning pretraining | RefCOCO Val | RefCOCO TestA | RefCOCO TestB | RefCOCO+ Val | RefCOCO+ TestA | RefCOCO+ TestB | RefCOCOg Val-U | RefCOCOg Test-U |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2VG | BEiT3-ViT-B | 28K | 91.84 | 94.17 | 88.05 | 86.47 | 90.29 | 80.43 | 84.97 | 86.11 |

## Installation

The reference environment uses Python 3.8, CUDA 11.8, PyTorch 2.0.0, and
Torchvision 0.15.1.

```bash
conda env create -f environment.yml
conda activate m2vg

pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
git clone https://github.com/IDEA-Research/detrex.git
cd detrex
git submodule update --init
pip install -e .
cd ..
pip install -e .
```

## Data preparation

Download the MS-COCO `train2014` images from the
[official COCO download page](https://cocodataset.org/#download). Download the
referring-expression and foreground annotations from the
[InstanceVG data release](https://huggingface.co/datasets/Dmmm997/InstanceVG_Data).
The expected layout is:

```text
data/
└── seqtr_type/
    ├── annotations/
    │   ├── mixed-seg/instances_nogoogle_withid.json
    │   ├── grefs/instance.json
    │   ├── ref-zom/instance.json
    │   └── rrefcoco/instance.json
    └── images/
        └── mscoco/
            └── train2014/
```

## Pretrained weights and checkpoints

Download the BEiT-3 pretrained backbone files and tokenizer from the
[official BEiT-3 release](https://github.com/microsoft/unilm/tree/master/beit3)
and place them in `pretrain_weights/`.

M2VG can directly load the compatible final checkpoints from the
[InstanceVG model release](https://huggingface.co/Dmmm997/InstanceVG). Place
the downloaded checkpoint under `work_dir/` and pair it with the matching M2VG
configuration:

| M2VG configuration | Compatible checkpoint |
| --- | --- |
| `configs/gres/M2VG-grefcoco.py` | `InstanceVG-grefcoco.pth` |
| `configs/refcoco/M2VG-B-refcoco/M2VG-B-refcoco.py` | `InstanceVG-B-refcoco.pth` |
| `configs/refcoco/M2VG-L-refcoco/M2VG-L-refcoco.py` | `InstanceVG-L-refcoco.pth` |
| `configs/refzom/M2VG-refzom.py` | `InstanceVG-refzom.pth` |
| `configs/rrefcoco/M2VG-rrefcoco.py` | `InstanceVG-rrefcoco.pth` |

## Training and evaluation

```bash
# Train
bash tools/dist_train.sh [PATH_TO_CONFIG] [NUM_GPUS]

# Evaluate
bash tools/dist_test.sh [PATH_TO_CONFIG] [NUM_GPUS] \
  --load-from [PATH_TO_CHECKPOINT]
```

For example:

```bash
bash tools/dist_test.sh configs/gres/M2VG-grefcoco.py 1 \
  --load-from work_dir/gres/InstanceVG-grefcoco.pth
```

## Demo

```bash
python tools/demo.py \
  --img "asserts/imgs/Figure_1.jpg" \
  --expression "three skateboard guys" \
  --config "configs/gres/M2VG-grefcoco.py" \
  --checkpoint /PATH/TO/InstanceVG-grefcoco.pth
```

## Acknowledgements

M2VG builds on open-source components from
[InstanceVG](https://github.com/Dmmm1997/InstanceVG),
[Detectron2](https://github.com/facebookresearch/detectron2),
[Detrex](https://github.com/IDEA-Research/detrex), and
[BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3).

## License

This project is distributed under the [MIT License](LICENSE.txt). See
[NOTICE.md](NOTICE.md) for third-party notices.
