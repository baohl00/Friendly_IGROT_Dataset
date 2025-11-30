import json
from pathlib import Path
import torch
from tqdm import tqdm
import PIL.Image as Image
# from lavis.models import load_model_and_preprocess
from transformers import pipeline, AutoProcessor
from utils import get_image, get_preprocess
import argparse 

## LLaVA Caption
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
pipe = pipeline("image-to-text", model=model_id, device=device)
processor = AutoProcessor.from_pretrained(model_id)
    
def llava_caption(image):
    #model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image in one sentence with details."},
                    {"type": "image"},
                    ],
                },
            ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    output = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 100})
    #print(output)
    output = output[0]["generated_text"].split('[/INST]')[-1].strip()
    #print(output)
    return output

dataset_path = '/home/hle/CIR/data'

DATASET = 'cirr' # 'cirr', 'fashioniq', 'circo'

if DATASET == 'circo':
    SPLIT = 'test'
    with open(f'{dataset_path}/circo/annotations/{SPLIT}.json', "r") as f:
        annotations = json.load(f)

    with open(f'{dataset_path}/circo/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json', "r") as f:
        imgs_info = json.load(f)

    img_paths = [f'{dataset_path}/circo/COCO2017_unlabeled/unlabeled2017/{img_info["file_name"]}' for img_info in
                        imgs_info["images"]]
    img_ids = [img_info["id"] for img_info in imgs_info["images"]]
    img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(img_ids)}

elif DATASET == 'cirr':
    SPLIT = 'test1'
    with open(f'{dataset_path}/CIRR/cirr/captions/cap.rc2.{SPLIT}.json') as f:
        annotations = json.load(f)

    with open(f'{dataset_path}/CIRR/cirr/image_splits/split.rc2.{SPLIT}.json') as f:
        name_to_relpath = json.load(f)
else:
    SPLIT = 'val'
    DRESS = 'dress' # 'shirt', 'toptee', 'dress'
    new_annotations = []
    with open(f'{dataset_path}/fiq/captions/cap.{DRESS}.{SPLIT}.json') as f:
        annotations = json.load(f) 

for ans in tqdm(annotations):
    if DATASET == 'circo':
        ref_img_id = ans["reference_img_id"]
        reference_img_id = str(ref_img_id)
        reference_img_path = img_paths[img_ids_indexes_map[reference_img_id]]
        target_img_path = None  # circo dataset might not have target images
    elif DATASET == 'cirr':
        ref_img_id = ans["reference"]
        rel_cap = ans["caption"]
        name_path = name_to_relpath[ref_img_id][2:]
        reference_img_path = f'{dataset_path}/CIRR/{name_path}'
        target_img_path = None  # cirr dataset might not have target images
    else:
        ref_img_name = ans["candidate"] + '.png'
        reference_img_path = f'{dataset_path}/fiq/images/{ref_img_name}'
        tar_img_id = ans["target"] + '.png'
        target_img_path = f'{dataset_path}/fiq/images/{tar_img_id}'
    
    #print(reference_img_path)
    ref_img = get_image(reference_img_path)
    try:
        if target_img_path:
            tar_img = get_image(target_img_path)
            ans["llava_target_caption"] = llava_caption(tar_img)
    except Exception as e:
        # print(f"Error generating target caption: {e}")
        continue
    
    try:
        ans["llava_reference_caption"] = llava_caption(ref_img)
        # print(ans["llava_reference_caption"])
    except Exception as e:
        # print(f"Error generating reference caption: {e}")
        continue

    if DATASET == 'fashioniq':
        new_annotations.append(ans)

if DATASET == 'circo':
    with open(f"{dataset_path}/circo/annotations/{SPLIT}_llava.json", "w") as f:
        json.dump(annotations, f, indent=4)
elif DATASET == 'cirr':
    with open(f"{dataset_path}/CIRR/cirr/captions/cap.rc2.{SPLIT}_llava.json", "w") as f:
        f.write(json.dumps(annotations, indent=4))
else:
    with open(f"{dataset_path}/fiq/captions/cap.{DRESS}.{SPLIT}_llava.json", "w") as f:
        f.write(json.dumps(new_annotations, indent=4))

print(f"DONE GENERATED CAPTION for {DATASET}")
