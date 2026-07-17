# Experimental results

All values are percentages. These tables record the M2VG results reported in
the anonymous manuscript and will be linked from the repository README once the
framework figure is finalized.

## gRefCOCO: generalized referring expression comprehension

| Model | Backbone | Val F1 | Val N-acc. | TestA F1 | TestA N-acc. | TestB F1 | TestB N-acc. | Avg. F1 | Avg. N-acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2VG | BEiT3-ViT-B | 75.27 | 76.35 | 68.94 | 74.39 | 59.38 | 67.31 | 67.86 | 72.68 |

## gRefCOCO: generalized referring expression segmentation

| Model | Backbone | Val gIoU | Val cIoU | Val N-acc. | TestA gIoU | TestA cIoU | TestA N-acc. | TestB gIoU | TestB cIoU | TestB N-acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2VG | BEiT3-ViT-B | 73.52 | 68.95 | 76.35 | 74.62 | 73.19 | 74.39 | 65.14 | 64.64 | 67.31 |

## Ref-ZOM: generalized referring expression segmentation

| Model | Backbone | oIoU | mIoU | Acc. |
| --- | --- | ---: | ---: | ---: |
| M2VG | BEiT3-ViT-B | 65.92 | 65.88 | 98.11 |

## RefCOCO, RefCOCO+, and RefCOCOg: referring expression comprehension

Precision at IoU 0.5 is the evaluation metric.

| Model | Backbone | Fine-tuning pretraining | RefCOCO Val | RefCOCO TestA | RefCOCO TestB | RefCOCO+ Val | RefCOCO+ TestA | RefCOCO+ TestB | RefCOCOg Val-U | RefCOCOg Test-U |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2VG | BEiT3-ViT-B | 28K | 91.84 | 94.17 | 88.05 | 86.47 | 90.29 | 80.43 | 84.97 | 86.11 |
