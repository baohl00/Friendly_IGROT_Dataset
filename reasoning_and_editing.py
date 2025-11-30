import json
import time
import traceback
import transformers
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path

DATASET = 'lasco' # 'cirr', 'fashioniq'

device = 'cuda' 

home_path = '/home/hle'

if DATASET == 'circo':
    SPLIT = 'test'
    input_json = 'CIRCO/annotations/test.json'
    dataset_path = Path('CIRCO')
elif DATASET == 'cirr':
    SPLIT = 'test1'
    input_json = 'CIRR/cirr/captions/cap.rc2.test1.json'
    dataset_path = Path('CIRR')
elif DATASET == 'lasco':
    SPLIT = 'train'
    input_json = f'{home_path}/spinning-storage/hle/LaSCo/llava_{SPLIT}_formatted.json'

model = "/home/support/llm/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device = device
        )

sys_prompt = """Your mission is to write a relative caption that describes how the reference image caption can be shifted into the target image caption.\nReference image caption: {ref_cap}.\nTarget image caption: {tar_cap}.\nRelative caption:"""

def write_relative_caption(reference_image_caption, target_image_caption):
    sequences = pipeline(
            sys_prompt.format(ref_cap = reference_image_caption, tar_cap = target_image_caption),
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            truncation = True,
            max_length=128,
            )
    relative_caption = sequences['generated_text']
    return relative_caption

print(write_relative_caption(
    "a cat with an apple tree", 
    "a bird is eating an apple"))

#with open(input_json, "w") as f:
#    f.write(json.dumps(annotations, indent=4))
