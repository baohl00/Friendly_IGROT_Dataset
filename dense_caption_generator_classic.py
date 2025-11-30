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
pipe = pipeline("image-to-text", model=model_id, device = device)
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

DATASET = 'circo' # 'cirr', 'fashioniq', 'circo'

if DATASET == 'circo':
    SPLIT = 'val'
    with open(f'{dataset_path}/circo/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json', "r") as f:
        annotations = json.load(f)

    img_paths = [f'{dataset_path}/circo/COCO2017_unlabeled/unlabeled2017/{img_info["file_name"]}' for img_info in
                        annotations["images"]]
    img_ids = [img_info["id"] for img_info in annotations["images"]]
    img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(img_ids)}
    annotations = img_paths

elif DATASET == 'cirr':
    SPLIT = 'train'
    with open(f'{dataset_path}/CIRR/cirr/image_splits/split.rc2.{SPLIT}.json') as f:
        annotations = json.load(f)
    annotations = list(annotations.items())
else:
    SPLIT = 'val'
    DRESS = 'shirt' # 'shirt', 'toptee', 'dress'
    
    with open(f'{dataset_path}/fiq/image_splits/split.{DRESS}.{SPLIT}.json') as f:
        annotations = json.load(f)

new_annotations = dict()

for ans in tqdm(annotations):
    if DATASET == 'circo':
        image_path = ans
    elif DATASET == 'cirr':
        img_id = ans[0]
        #print(img_id)
        image_path = f'{dataset_path}/CIRR/{ans[1][2:]}'
    else:
        image_path = f'{dataset_path}/fiq/images/{ans}.png'
        
    img = get_image(image_path)
    new_annotations[ans.replace("/home/hle/CIR/data", "")] = {
            "caption": llava_caption(img),
            "path": ans.replace("/home/hle/CIR/data", "")
            }
    #print(new_annotations)
    
if DATASET == 'circo':
    with open(f"{dataset_path}/circo/annotations/classic.{SPLIT}_llava.json", "w") as f:
        f.write(json.dumps(new_annotations, indent=4))
elif DATASET == 'cirr':
    with open(f"{dataset_path}/CIRR/cirr/captions/classic.cap.rc2.{SPLIT}_llava.json", "w") as f:
        f.write(json.dumps(new_annotations, indent=4))
else:
    with open(f"{dataset_path}/fiq/captions/classic.cap.{DRESS}.{SPLIT}_llava.json", "w") as f:
        f.write(json.dumps(new_annotations, indent=4))
    print(f"DONE with {DRESS}")

print(f"DONE GENERATED CAPTION for {DATASET}") 
