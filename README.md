# M2VG: Expression Satisfiability Modeling for Generalized Visual Grounding

M2VG is a research-code release for generalized visual grounding.
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
one-to-zero, one-to-one, and one-to-many scenarios. Beyond localizing referred
instances, a model must determine whether a referring expression is supported by
the image. Existing approaches primarily optimize text-region matching and can
therefore return plausible but incorrect regions when a local category cue
matches while attribute, spatial, or relational conditions are violated.

We propose M2VG, an expression satisfiability modeling framework for generalized
visual grounding. M2VG treats a referring expression as a set of visual
conditions and learns to reject predictions when the image does not provide
sufficient joint evidence for these conditions. To construct informative
no-target supervision, Dynamic Focus Negative Sample Mining generates reliable
unsatisfiable image-expression pairs through target removal and semantic
mismatch under reliability constraints. We further introduce Adaptive
Word-aware Residual Modulation to refine token representations with image
context before query generation, and an auxiliary existence branch that
aggregates decoded instance-query evidence to estimate expression-level
existence. Experiments on generalized referring expression comprehension,
generalized referring expression segmentation, and conventional referring
expression comprehension show that M2VG improves no-target recognition while
maintaining competitive localization and segmentation performance.

**Keywords:** visual grounding; generalized visual grounding; referring
expression comprehension; referring expression segmentation; expression
satisfiability.

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
[annotation release](https://huggingface.co/datasets/Dmmm997/InstanceVG_Data).
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

Place the final M2VG checkpoints under `work_dir/` and pair them with the
matching M2VG configuration:

| M2VG configuration | Compatible checkpoint |
| --- | --- |
| `configs/gres/M2VG-grefcoco.py` | `M2VG-grefcoco.pth` |
| `configs/refcoco/M2VG-B-refcoco/M2VG-B-refcoco.py` | `M2VG-B-refcoco.pth` |
| `configs/refcoco/M2VG-L-refcoco/M2VG-L-refcoco.py` | `M2VG-L-refcoco.pth` |
| `configs/refzom/M2VG-refzom.py` | `M2VG-refzom.pth` |
| `configs/rrefcoco/M2VG-rrefcoco.py` | `M2VG-rrefcoco.pth` |

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
  --load-from work_dir/gres/M2VG-grefcoco.pth
```

## Demo

```bash
python tools/demo.py \
  --img "asserts/imgs/Figure_1.jpg" \
  --expression "three skateboard guys" \
  --config "configs/gres/M2VG-grefcoco.py" \
  --checkpoint /PATH/TO/M2VG-grefcoco.pth
```

## Acknowledgements

M2VG uses open-source components from
[Detectron2](https://github.com/facebookresearch/detectron2),
[Detrex](https://github.com/IDEA-Research/detrex), and
[BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3).

## License

This project is distributed under the [MIT License](LICENSE.txt). See
[NOTICE.md](NOTICE.md) for third-party notices.
