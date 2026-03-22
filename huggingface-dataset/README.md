---
license: mit
task_categories:
  - image-classification
  - feature-extraction
language:
  - en
tags:
  - orthography
  - visual-word-recognition
  - form-priming
  - cognitive-science
  - psycholinguistics
  - cnn
  - vit
pretty_name: orthographic DNN priming dataset
size_categories:
  - 10K<n<100K
---

# orthographic DNN priming dataset

replication data for the paper:

> Yin, D. and Davelaar, E.J. (2023). convolutional neural networks trained to identify words provide a good account of visual form priming effects. _Computational Brain & Behavior_. [doi:10.1007/s42113-023-00172-7](https://link.springer.com/article/10.1007/s42113-023-00172-7)

## dataset description

this dataset contains pre-rendered word stimulus images used to evaluate how well visual DNN models (CNNs and ViTs) predict human orthographic priming patterns from the [form priming project](https://link.springer.com/article/10.3758/s13428-013-0442-y) (adelman et al., 2014).

each image is a 224x224 black-background PNG with white text, rendered in arial at size 22, centred.

### example stimuli for the target word "design"

<table>
  <tr>
    <td align="center"><b>ID</b><br>(identity)</td>
    <td align="center"><b>TL12</b><br>(transposed 1-2)</td>
    <td align="center"><b>DL-1M</b><br>(deleted middle)</td>
    <td align="center"><b>SN-M</b><br>(substituted middle)</td>
    <td align="center"><b>RF</b><br>(reversed full)</td>
    <td align="center"><b>ALD-ARB</b><br>(all different)</td>
  </tr>
  <tr>
    <td align="center"><img src="prime_data/design/ID.png" width="120"></td>
    <td align="center"><img src="prime_data/design/TL12.png" width="120"></td>
    <td align="center"><img src="prime_data/design/DL-1M.png" width="120"></td>
    <td align="center"><img src="prime_data/design/SN-M.png" width="120"></td>
    <td align="center"><img src="prime_data/design/RF.png" width="120"></td>
    <td align="center"><img src="prime_data/design/ALD-ARB.png" width="120"></td>
  </tr>
  <tr>
    <td align="center">DESIGN</td>
    <td align="center">EDSIGN</td>
    <td align="center">DSIGN</td>
    <td align="center">DESIHN</td>
    <td align="center">NGISE</td>
    <td align="center">CBHAUX</td>
  </tr>
</table>

### what's included

```
prime_data/           <- 11,760 prime stimulus images (420 targets x 28 conditions)
  {target_word}/
    {condition}.png
metadata/
  2014-prime-types.txt   <- 28 prime condition labels
  2014-targets.txt       <- 420 target words
  2014-prime-data.json   <- prime string for each target x condition
  normalization-stats.json <- channel-wise mean/std for the training set
```

### prime conditions (28 types from the form priming project)

| code | description |
|------|-------------|
| ID | identity (e.g., prime and target are both "design") |
| TL12 | transposed letters positions 1-2 |
| TL-I | transposed letters internal |
| TL56 | transposed letters positions 5-6 |
| NATL2 | non-adjacent transposition (2 letters) |
| NATL3 | non-adjacent transposition (3 letters) |
| DL-1M | deleted letter (1, middle) |
| DL-1F | deleted letter (1, final) |
| DL-2M | deleted letters (2, middle) |
| T-All | all letters transposed |
| TH | transposed halves |
| SUB3 | subset of 3 letters |
| RH | reversed halves |
| IH | interleaved halves |
| RF | reversed full |
| SN-I | single substitution (initial) |
| SN-M | single substitution (middle) |
| SN-F | single substitution (final) |
| N1R | neighbours at distance 1 (random) |
| DSN-M | double substitution (middle) |
| IL-1M | inserted letter (1, middle) |
| IL-2M | inserted letters (2, middle) |
| EL | extra letter |
| IL-1I | inserted letter (1, initial) |
| IL-1F | inserted letter (1, final) |
| IL-2MR | inserted letters (2, middle random) |
| ALD-ARB | all-letter-different arbitrary |
| ALD-PW | all-letter-different pseudoword |

## quick start

```python
from datasets import load_dataset

dataset = load_dataset("donyin/orthographic-dnn-priming")
```

or load images directly:

```python
from pathlib import Path
from PIL import Image

prime_dir = Path("prime_data")
target = "design"
condition = "TL12"

img = Image.open(prime_dir / target / f"{condition}.png")
```

## reproducing the main result

the core analysis computes kendall's tau between model cosine-similarity patterns and human priming scores across the 28 conditions. see the [source code repository](https://github.com/Don-Yin/Orthographic-DNN) for the full pipeline:

1. fine-tune pretrained torchvision models on word classification (training images not included here; generate with `generate_data.py`)
2. extract layer-wise activations for each prime image pair (identity vs. condition)
3. compute cosine similarity at each layer
4. correlate with human priming scores using kendall's tau

## training data

training images (800k+) are not included due to size. they are fully reproducible:

```bash
cd src && python generate_data.py
```

this requires the font files (not redistributable) and generates images with configurable rotation, translation, font-size variation, and spacing jitter. see `src/utils/data_generate/main.py` for parameters.

## models evaluated

alexnet, densenet169, efficientnet-b1, resnet50, resnet101, vgg16, vgg19, vit-b/16, vit-b/32, vit-l/16, vit-l/32, all initialised from imagenet pretrained weights via torchvision.

## citation

```bibtex
@article{yin2023cnn,
  title={Convolutional Neural Networks Trained to Identify Words Provide a Good Account of Visual Form Priming Effects},
  author={Yin, Don and Davelaar, Eddy J.},
  journal={Computational Brain \& Behavior},
  year={2023},
  publisher={Springer},
  doi={10.1007/s42113-023-00172-7}
}
```

## license

MIT
