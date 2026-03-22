# Orthographic-DNN

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/DonYin/orthographic-dnn-priming)

This is the code repository for the research project *[Convolutional Neural Networks Trained to Identify Words Provide a Good Account of Visual Form Priming Effects](https://link.springer.com/article/10.1007/s42113-023-00172-7)*

[![Click to see Video Demo](src/assets/_demo.png)](https://youtu.be/PEvwzwo6gFM)

*Project objective*: comparing human orthographic perception with visual DNN models (CNNs and ViTs).

*Project outcome*: CNNs did a good job in predicting the pattern of human priming scores across conditions, with correlations ranging from τ = .49 (AlexNet) to τ = .71. (ResNet101) with all p-values < .01. The CNNs performed similarly to the various orthographic coding schemes word recognition models, and often better. This contrasts with the relatively poor performance of the Transformer networks, with τ ranging from .25 to .38.

## Rationale
1. Prime Conditions
   The Form Priming Project includes 28 prime conditions for how a letter string can be amended to form a new string. For example, the word $design$ in the "final tansposition" condition will be presented as $deigns$.
2. Measuring humans' perceptual similarity of words or letter strings:
   For a human participant, the similarity $sim(s_1, s_2)$ of two word strings $s_1$ (the target) and $s_2$ (the prime) is measured using a Lexical Decision Task (LDT), where $s_1$ and $s_2$ are presented one at a time, with a fixation cross in between, and the participant has to decide as quickly as possible whether $s_1$ is a word or not. The reaction time is compared to that when the target word is presented with an arbitrary random string $s_3$ as prime. The similarity $sim(s_1, s_2)$ is calculated as $sim(s_1, s_2) = RT_{s_1|s_2} - RT_{s_1|s_3}$. For each condition $C$ and each prime string $s_2$, the mean similarity $\bar{sim}(s_1, C)$ is calculated by averaging the similarity $sim(s_1, s_2)$ over the 420 prime strings $s_2$ for $C$.
3. Measuring models' perceptual similarity of words or letter strings:
   For the models, the similarity $sim(s_1, s_2)$ is measured by the cosine similarity $sim(s_1, s_2) = \cos(s_1, s_2)$ between the two vectors $s_1$ and $s_2$ where $s_1$ and $s_2$ are the flattened penulimate layer outputs when the models are fed with two images of the two strings. For each condition  $C$ and each prime string $s_2$, the mean similarity $\bar{sim}(s_1, C)$ is calculated by averaging the similarity $sim(s_1, s_2)$ over the 420 prime strings $s_2$ for $C$.
   
   ![Example](src/assets/feature_extraction.png)
   
4. Comparing the perceptual patterns between humans and models:
   Kendall's rank correlation coefficient $\tau$ is used to measure the correlation between the human and model priming scores across conditions. The human priming scores are taken from the Form Priming Project, and the model priming scores are calculated by the code in this repository. For a given model $M$, its similarity with human priming is calculated as $\tau(M) = \sum_{C}(\bar{sim}(s_1, C)_M - \bar{sim}(s_1, C)_{human})\text{sign}(\bar{sim}(s_1, C)_M - \bar{sim}(s_1, C)_{human})$
   where $\bar{sim}(s_1, C)_{M}$ and $\bar{sim}(s_1, C)_{human}$ are the mean similarity scores of the model $M$ and the human participant, respectively, for condition $C$.

## Data
1. the Fonts used to generate the data are in `assets/fonts` stored as `.ttf` files.
2. The human priming data is sourced from [the Form Priming Project (FPP)](https://link.springer.com/article/10.3758/s13428-013-0442-y), available at [this link](https://files.warwick.ac.uk/jadelman2/browse#FPP) or [here](src/assets/adelman.xlsx) or [here](src/assets/adelman.csv).
3. You can either download the [training data](https://drive.google.com/file/d/1w6m_57z6lVh97Cr6MJPNQbKYjh877zbL/view?usp=share_link) and the [prime data](https://drive.google.com/file/d/1qDyqdSIzwRQlqmi8kUJ34_G4Dr-qvYzZ/view?usp=sharing) as zip files or run the `generate_data.py` script to generate as many images as you like. The configurations of letter translation, rotation, variation s of font and sizes are at [here](src/utils/data_generate/main.py). The zip file of the training data contains 800,000 images which should be enough for all models used in the current research.

## Setup
1. install `python==3.10.4`
2. install [cuda driver](https://developer.nvidia.com/cuda-downloads)
3. install pytorch on [pytorch.org](pytorch.org) - >= torch-1.11.0
4. `pip install -r requirements.txt`

## Psychological Priming Models and Coding Schemes
1. The LTRS model simulator is available at [AdelmanLab](http://www.adelmanlab.org/ltrs/)
2. The Interactive Activation Model and the Spatial Coding Model are implemented using [this calculator](http://www.pc.rhul.ac.uk/staff/c.davis/SpatialCodingModel/) developed by [Prof. Colin Davis](https://www.bristol.ac.uk/people/person/Colin-Davis-49b9852f-cb91-4196-95cb-509804919a1e/).

## Model Parameters
The tested models are [Alexnet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), [DenseNet169](https://arxiv.org/abs/1608.06993), [EfficientNet-B1](https://arxiv.org/abs/1905.11946)
, [ResNet50, ResNet101](https://arxiv.org/abs/1512.03385), [VGG16, VGG19](https://arxiv.org/abs/1409.1556), [ViT-B/16, ViT-B/32, ViT-L/16 and ViT-L/32](https://arxiv.org/abs/2010.11929). The models are initiated using ImageNet pre-trained weights from [Torchvision](https://pytorch.org/vision/stable/models.html), code for loading the weights are at [tune.py](tune.py). The trained parameters are available [here](https://drive.google.com/file/d/1chprLm5YJXTzgA7KGjEsNFjeZFVZzISG/view?usp=sharing)

## Additional Findings
1. layer-wise correlation coefficient: to be added*

## Acknowledgement
The project was conducted under the auspices of the [University of Bristol Mind & Machine Lab](https://mindandmachine.blogs.bristol.ac.uk/) and supported by [European Research Council (ERC)](https://erc.europa.eu/homepage) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 741134).

## Contact
For further instructions and enquiries, please contact [Don Yin](mailto:don_yin@outlook.com).

## License
MIT License (see LICENSE file)
