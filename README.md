# FIGROTD: A Friendly-to-Handle Dataset for Image Guided Retrieval with Optional Text 

_Accepted at MMM2026._

## Abstract

Image-Guided Retrieval with Optional Text (IGROT) unifies visual retrieval (without text) and composed retrieval (with text). Despite its relevance in applications like Google Image and Bing, progress has been limited by the lack of an accessible benchmark and methods that balance performance across subtasks. Large-scale datasets such as MagicLens are comprehensive but computationally prohibitive, while existing models often favor either visual or compositional queries. We introduce **FIGROTD**, a lightweight yet high-quality IGROT dataset with 16,474 training triplets and 1,262 test triplets across CIR, SBIR, and CSTBIR. To reduce redundancy, we propose the **Variance Guided Feature Mask (VaGFeM)**, which selectively enhances discriminative dimensions based on variance statistics. We further adopt a dual-loss design (InfoNCE + Triplet) to improve compositional reasoning. Trained on FIGROTD, VaGFeM achieves competitive results on nine benchmarks, reaching 34.8 mAP@10 on CIRCO and 75.7 mAP@200 on Sketchy, outperforming stronger baselines despite fewer triplets.

## Setup
```
micromamba create -n union python=3.9
micromamba activate union
pip install -r requirements_blip.txt 
pip install -r requirements.txt
```

### Model Download & Data Preparation
| Pretrained Model | Link | 
| ------ | ---- | 
| CLIP ViT-B/32 | [here](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) |
| CLIP ViT-L/14 | [here](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) | 
| BLIP-B (COCO) | [here](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth) |

Data is available [here](https://drive.google.com/drive/folders/16Q0yV-ROqtWGDO3M6ckgP_mQPW5BD64X?usp=sharing).

## Inference
We prepare two different files for inference stage. You can train and inference, if you run this:
```
bash scripts/inference.sh
```
or only train the model: 
```
bash scripts/train.sh
```


## Citing this work

Add citation details here, usually a pastable BibTeX snippet:

```latex
@InProceedings{le2026figrotd,
author="Le, Hoang-Bao
and Tran, Allie
and Nguyen, Binh T.
and Zhou, Liting
and Gurrin, Cathal",
editor="Loko{\v{c}}, Jakub
and Pe{\v{s}}ka, Ladislav
and Zah{\'a}lka, Jan
and Rudinac, Stevan
and Kastner, Marc
and Chen, Jingjing
and Hu, Min-Chun
and Wu, Jiaxin
and Sharma, Ujjwal",
title="FIGROTD: A Friendly-to-Handle Dataset for Image Guided Retrieval with Optional Text",
booktitle="MultiMedia Modeling",
year="2026",
publisher="Springer Nature Singapore",
address="Singapore",
pages="117--132",
isbn="978-981-95-6950-2"
}
```

## Acknowledgement 

We extend our gratitude to the open-source efforts of [TransAgg](https://github.com/Code-kunkun/ZS-CIR) and [UNION_for_IGROT](https://github.com/baohl00/UNION_for_IGROT).
