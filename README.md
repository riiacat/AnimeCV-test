# AnimeCV

Pretrained computer vision tools for anime style illustrations. [日本語README](README_ja.md)

## Getting started

Install AnimeCV by pip:

```
pip install git+https://github.com/kosuke1701/AnimeCV.git
```

## Features

### Near Human-Level Character Identification

The task is to identify which character is depicted in a picture given a set of reference pictures for each character.
[Example code](examples/character_re_identification.py)

* **Update on 2021.01.11**
  - I released new pre-trained model for character face embedding!
  - See [release](https://github.com/kosuke1701/AnimeCV/releases/tag/0111_best_randaug) and demo  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kosuke1701/AnimeCV/blob/master/examples/demo_oml.ipynb).
* **Update on 2021.02.07**
  - I released new pre-trained models for character face embedding.
  - The models are trained on [ZACI-20](https://github.com/kosuke1701/ZACI-20-dataset), a public zero-shot character identification dataset.
  - **The best model achieves near human-level performance with only 1.5 times larger error rate than that of humans.**
  - You can load this model by using `animecv.general.create_OML_ImageFolder_Encoder(model_directory_name)`.
  - Download models from [release](https://github.com/kosuke1701/AnimeCV/releases/tag/0111_best_randaug).

### Object detection

The task is to detect objects in a picture and return bounding boxes around the objects.

* Face detector. [Example code](examples/character_face_detection.py)
  - **Update on 2021.01.18**
    - I released [Yet Another Character Face Annotations on Danbooru2020](https://github.com/kosuke1701/AnimeCV/releases/tag/0.0) using this pre-trained model!
    - Automatically annotated face bounding boxes for the SFW 512px downscaled subset of [Danbooru2020 dataset](https://www.gwern.net/Danbooru2020).

## Note

All models are trained on datasets which mainly consists of Japanese anime style illustrations. Thus, it may not perform well on illustrations with other styles.
