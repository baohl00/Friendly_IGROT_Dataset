import os
import json
from pathlib import Path
import torch
from tqdm import tqdm
import PIL.Image as Image
import torch.nn as nn 

from transformers import pipeline, AutoProcessor
from utils import get_image, get_preprocess
import argparse 

MAIN_PATH = '/home/hle/spinning-storage/hle/LaSCo' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer once (reuse for performance)
model_path = "/home/hle/spinning-storage/hle/llm/"
#model_id = "DeepHermes-3-Mistral-24B-Preview"
#model_id = "DeepSeek-R1"
#model_id = "Llama-3.1-8B"
#model_id = "Meta-Llama-3-8B"
#model_id = "Llama-3.3-70B-Instruct"
model_id = "Mistral-7B-Instruct-v0.2"
#model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct"
model_name = model_path + model_id if "/" not in model_id else model_id
local = True if "/" not in model_id else False

#tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local)
#model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local)
#model.to(device)
#model.eval()
print(model_name)
pipeline = pipeline(
        "text-generation",
        model=model_name,
        #model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
        )
def generate_target_caption(reference_caption: str, relative_caption: str, few_shot_examples=None, max_new_tokens=50) -> str:
    """
    Generate a standalone target image caption using Qwen2.5-7B-Instruct based on a reference image caption and a relative caption.
    
    Args:
        reference_caption (str): Description of the reference image.
        relative_caption (str): How the target image differs from the reference.
        few_shot_examples (list): Optional list of dicts with keys: 'reference', 'relative', 'target'.
        max_new_tokens (int): Maximum length of the generated caption.
    
    Returns:
        str: The generated caption for the target image.
    """
    # Build prompt
    prompt = (
            "Rewrite a new caption that focuses on the target image and relative caption"
           # "and not having the things listed on the reference caption. "
            "Do not mention or refer back to the reference image.\n\n"
            )
    if few_shot_examples:
        for i, ex in enumerate(few_shot_examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Reference Caption: {ex['reference']}\n"
            prompt += f"Relative Caption: {ex['relative']}\n"
            prompt += f"Target Caption: {ex['target']}\n\n"
    
    prompt += f"Reference Caption: {reference_caption}\n"
    prompt += f"Relative Caption: {relative_caption}\n"
    prompt += "Target Caption:"

    # Tokenize and generate
    messages = [
            {"role": "user", "content": prompt}
            ]
    #input_ids = tokenizer.apply_chat_template(
    #        messages,
    #        add_generation_prompt=True,
    #        return_tensors="pt").to(model.device)

    terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

    #inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            eos_token_id=terminators #tokenizer.eos_token_id,
        )
    #result = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    # Extract only the generated caption
    #generated_caption = result.split("Target Caption:")[-1].strip()
    return output[0]["generated_text"][-1]["content"].strip()

class Dataset():
    def __init__(self, split):
        self.split = split
        self.path = MAIN_PATH 
        with open(f"{self.path}/{self.split}_llava_formatted.json", "r") as f:
            self.data = json.load(f)#[214742:]
        print(f'DATA {self.split} is being preprocessed...')

    def __getitem__(self, index):
        reference_image_id = index["reference_image_id"]
        target_image_id = index["target_image_id"]

        # Read and preprocess images
        reference_caption = index["llava_reference_caption"]
        relative_caption = index["relative_caption"]
        target_caption = index["llava_target_caption"]
        return reference_image_id, relative_caption, target_image_id, reference_caption, target_caption 
    
    def __saveData__(self):
        records = list()
        f = open(f"{MAIN_PATH}/{self.split}_llava_mistral.json", "w")
        for i, record in enumerate(tqdm(self.data)):
            reference_image_id, relative_caption, target_image_id, reference_caption, target_caption = self.__getitem__(record)
            few_shots = [
                    {
                        "reference": "a boy riding a bike on a dirt road",
                        "relative": "the boy is now skateboarding in a skatepark",
                        "target": "a boy skateboarding in a skatepark"
                        }]

            new_rel_cap = generate_target_caption(target_caption, relative_caption, few_shot_examples=None, max_new_tokens=100)
            #print(qwen_rel_cap)
            
            records.append({
                "reference_image_id": reference_image_id,
                "relative_caption": relative_caption,
                "target_image_id": target_image_id, 
                "llava_reference_caption": reference_caption,
                "llava_target_caption": target_caption,
                "new_relative_caption": new_rel_cap
            }) 
        
            f.write(json.dumps(records[-1], indent=4) + "\n")
            f.flush()

        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    dataset = Dataset(args.split)
    dataset.__saveData__()
