# UNION: A Lightweight Target Representation for Efficient Image-Guided Retrieval with Optional Textual Queries 

## Abstract

Image-Guided Retrieval with Optional Text (IGROT) is a general retrieval setting where a query consists of an anchor image, with or without accompanying text, aiming to retrieve semantically relevant target images. This formulation unifies two major tasks: Composed Image Retrieval (CIR) and Sketch-Based Image Retrieval (SBIR). In this work, we address IGROT under low-data supervision by introducing UNION, a lightweight and generalizable target representation that fuses the image embedding with a null-text prompt. Unlike traditional approaches that rely on frozen target features, UNION enhances semantic alignment with multimodal queries while requiring no architectural modifications to pretrained vision-language models. With only 5k training samples—from LlavaSCo for CIR and Training-Sketchy for SBIR—our method achieves competitive results across benchmarks, including CIRCO mAP@50 of 38.5 and Sketchy mAP@200 of 82.7, surpassing many heavily supervised baselines. This demonstrates the robustness and efficiency of UNION in bridging vision and language across diverse query types.

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



## Inference
We prepare two different files for inference stage. You can train and inference, if you run this:
```
bash scripts/inference.sh
```
or only train the model: 
```
bash scripts/train.sh
```

*Note*: If you want to change the target image feature type (original, sum, union), please change it in the script files, then one more in [utils.py](utils.py). 

Due to the weight conversion, the performance may be slightly different:

In **Zero-Shot Composed Image Retrieval**: `FashionIQ`, `CIRR`and `CIRCO`

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2">Backbone</th>
      <th rowspan="2"># Params</th>
      <th rowspan="2"># Triplets</th>
      <th colspan="2">FashionIQ (R)</th>
      <th colspan="2">CIRR (R)</th>
      <th colspan="2">CIRCO (mAP)</th>
    </tr>
    <tr>
      <th>@10</th>
      <th>@50</th>
      <th>@10</th>
      <th>@50</th>
      <th>@10</th>
      <th>@50</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pic2Word</td>
      <td>CLIP-L</td>
      <td>429M</td>
      <td>3M</td>
      <td>24.7</td>
      <td>43.7</td>
      <td>65.3</td>
      <td>87.8</td>
      <td>9.5</td>
      <td>11.3</td>
    </tr>
    <tr>
      <td>i-SEARLE</td>
      <td>CLIP-L</td>
      <td>442M</td>
      <td>205K</td>
      <td>29.2</td>
      <td>49.5</td>
      <td>66.7</td>
      <td>88.8</td>
      <td>13.6</td>
      <td>16.3</td>
    </tr>
    <tr>
      <td>CIReVL</td>
      <td>CLIP-L</td>
      <td>12.5B</td>
      <td>-</td>
      <td>28.6</td>
      <td>48.6</td>
      <td>64.9</td>
      <td>86.3</td>
      <td>19.1</td>
      <td>20.9</td>
    </tr>
    <tr>
      <td>MLLM-I2W</td>
      <td>CLIP-L</td>
      <td>-</td>
      <td>3M</td>
      <td>30.3</td>
      <td>50.1</td>
      <td>68.4</td>
      <td>92.4</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>PLI</td>
      <td>CLIP-L</td>
      <td>428M</td>
      <td>695K</td>
      <td><span style="color:red">35.4</span></td>
      <td><span style="color:red">57.4</span></td>
      <td>69.3</td>
      <td>89.6</td>
      <td>14.2</td>
      <td>16.4</td>
    </tr>
    <tr>
      <td>LinCIR</td>
      <td>CLIP-L</td>
      <td>-</td>
      <td>5.5M</td>
      <td>26.4</td>
      <td>46.6</td>
      <td>66.9</td>
      <td>88.2</td>
      <td>13.9</td>
      <td>16.2</td>
    </tr>
    <tr>
      <td>MagicLens</td>
      <td>CLIP-L</td>
      <td>465M</td>
      <td>36.5M</td>
      <td>30.7</td>
      <td>52.5</td>
      <td>74.4</td>
      <td>92.6</td>
      <td>30.8</td>
      <td>34.4</td>
    </tr>
    <tr>
      <td>CoLLM</td>
      <td>BLIP-B</td>
      <td>-</td>
      <td>3.4M</td>
      <td><span style="color:blue">34.6</span></td>
      <td><span style="color:blue">56.0</span></td>
      <td><span style="color:blue">78.6</span></td>
      <td><span style="color:blue">94.2</span></td>
      <td>20.4</td>
      <td>23.1</td>
    </tr>
    <tr>
      <td>TransAgg</td>
      <td>BLIP-B</td>
      <td>235M</td>
      <td>32K</td>
      <td>34.4</td>
      <td>55.1</td>
      <td>77.9</td>
      <td>93.4</td>
      <td><span style="color:blue">32.2</span></td>
      <td><span style="color:blue">36.2</span></td>
    </tr>
    <tr>
      <td>TransAgg + UNION</td>
      <td>BLIP-B</td>
      <td>235M</td>
      <td>5K</td>
      <td>31.9</td>
      <td>51.5</td>
      <td><span style="color:red">77.6</span></td>
      <td><span style="color:red">92.9</span></td>
      <td><span style="color:red">34.5</span></td>
      <td><span style="color:red">38.5</span></td>
    </tr>
  </tbody>
</table>



In **Zero-Shot Sketch-Based Image Retrieval**: `Sketchy`, `TUBerlin` and `QuickDraw`

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th rowspan="2">Backbone</th>
      <th rowspan="2"># Pairs</th>
      <th colspan="2">Sketchy</th>
      <th colspan="2">TU-Berlin</th>
      <th colspan="2">QuickDraw</th>
    </tr>
    <tr>
      <th>mAP@200</th>
      <th>Prec@200</th>
      <th>mAP</th>
      <th>Prec@100</th>
      <th>mAP</th>
      <th>Prec@200</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DCDL</td>
      <td>CLIP-B</td>
      <td>57K/15K/236K</td>
      <td><span style="color:blue">72.6</span></td>
      <td><span style="color:blue">76.9</span></td>
      <td><span style="color:red">63.4</span></td>
      <td><span style="color:red">74.1</span></td>
      <td><span style="color:red">33.6</span></td>
      <td>29.6</td>
    </tr>
    <tr>
      <td>CAT</td>
      <td>CLIP-B</td>
      <td>57K/15K/236K</td>
      <td>71.3</td>
      <td>72.5</td>
      <td>63.1</td>
      <td>72.2</td>
      <td>20.2</td>
      <td><span style="color:blue">38.8</span></td>
    </tr>
    <tr>
      <td>IVT</td>
      <td>ViT-B</td>
      <td>57K/15K/236K</td>
      <td>61.5</td>
      <td>69.4</td>
      <td>55.7</td>
      <td>62.9</td>
      <td>32.4</td>
      <td>16.2</td>
    </tr>
    <tr>
      <td>ZSE-SBIR</td>
      <td>ViT-L</td>
      <td>57K/15K/236K</td>
      <td>52.5</td>
      <td>62.4</td>
      <td>54.2</td>
      <td>65.7</td>
      <td>14.5</td>
      <td>21.6</td>
    </tr>
    <tr>
      <td>MagicLens</td>
      <td>CLIP-L</td>
      <td>36.7M</td>
      <td>68.2</td>
      <td>75.8</td>
      <td>62.9</td>
      <td><span style="color:blue">73.1</span></td>
      <td>15.1</td>
      <td>20.4</td>
    </tr>
    <tr>
      <td>TransAgg + UNION</td>
      <td>CLIP-L</td>
      <td>5K</td>
      <td><span style="color:red">82.7</span></td>
      <td><span style="color:red">79.9</span></td>
      <td>51.0</td>
      <td>69.8</td>
      <td>33.4</td>
      <td><span style="color:red">41.5</span></td>
    </tr>
  </tbody>
</table>


## Citing this work

Add citation details here, usually a pastable BibTeX snippet:

```latex

```

## Acknowledgement 

We extend our gratitude to the open-source efforts of [TransAgg](https://github.com/Code-kunkun/ZS-CIR). 
