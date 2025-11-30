import os
import json
from pathlib import Path
import torch
from tqdm import tqdm
import PIL.Image as Image
import torch.nn as nn 
#from lavis.models import load_model_and_preprocess

#from torchvision import transforms
#from transformers import OFATokenizer, OFAModel#, AutoProcessor, BlipForConditionalGeneration
#from transformers.models.ofa.generate import sequence_generator
from transformers import pipeline, AutoProcessor
from utils import get_image, get_preprocess
import argparse 

MAIN_PATH = '/home/hle/spinning-storage/hle/LaSCo' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model, vis_processors, _ = load_model_and_preprocess(
#            name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
#            )
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
pipe = pipeline("image-to-text", model=model_id, device = device)
processor = AutoProcessor.from_pretrained(model_id)
    
def llava_caption(image):
    #model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Describe the image in one sentence with details."},
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

    
#def BLIP_caption(img):
    # model_name = 'Salesforce/blip2-opt-6.7b'
#    model_name = 'Salesforce/blip2-flan-t5-xl'
    #processor = AutoProcessor.from_pretrained(model_name)
    #model = BlipForConditionalGeneration.from_pretrained(model_name).to(device) 
#    img = vis_processors["eval"](img).unsqueeze(0).to(device)
#    caption = model.generate({"image": img})
    #generated_ids = model.generate(**img, max_new_tokens=16)
    #caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#    return caption[0]

class Dataset():
    def __init__(self, data, preprocess, dimension):
        self.data_type = data 
        self.path = MAIN_PATH 
        with open(f"{self.path}/lasco_{data}.json", "r") as f:
            self.data = json.load(f)[:1]
            #self.data = json.load(f)
        print(f'DATA {self.data_type} is being preprocessed...')
        self.preprocess_type = preprocess
        self.preprocess = get_preprocess(preprocess, dimension)
        self.dimension = dimension

    def __getitem__(self, index):
        qid = index["qid"]
        reference_image_id, reference_image_path = index["query-image"]
        reference_image_path = os.path.join(self.path, reference_image_path)
        query_text = index["query-text"]
        target_image_id, target_image_path = index["target-image"]
        target_image_path = os.path.join(self.path, target_image_path)

        # Read and preprocess images
        #reference_image = self.preprocess(get_image(reference_image_path)).unsqueeze(0) #.to(device)
        reference_image = get_image(reference_image_path)
        relative_caption = query_text
        target_image = get_image(target_image_path)
        #target_image = self.preprocess(get_image(target_image_path)).unsqueeze(0) #.to(device)
        return reference_image, relative_caption, target_image, reference_image_id, target_image_id
    
    def __saveData__(self):
        records = list()
        #f = open(f"{MAIN_PATH}/{self.data_type}_{self.preprocess_type}_blip.json", "w") 
        f = open(f"{MAIN_PATH}/{self.data_type}_llava_tarcap.json", "w")
        for i, record in enumerate(tqdm(self.data)):
            reference_image, relative_caption, target_image, reference_image_id, target_image_id = self.__getitem__(record)
            #print(reference_image_id, target_image_id)
            #ofa_reference_caption = OFA_caption(reference_image) 
            #ofa_target_caption = OFA_caption(target_image)
            #blip_reference_caption = BLIP_caption(reference_image)
            #blip_target_caption = BLIP_caption(target_image)
            llava_ref_cap = llava_caption(reference_image)
            llava_tar_cap = llava_caption(target_image)
            #print(relative_caption, ofa_reference_caption, ofa_target_caption, blip_reference_caption, blip_target_caption)
            
            records.append({
                "reference_image_id": reference_image_id,
                "relative_caption": relative_caption,
                "target_image_id": target_image_id, 
                "llava_reference_caption": llava_ref_cap,
                "llava_target_caption": llava_tar_cap
                #"ofa_reference_caption": ofa_reference_caption,
                #"blip_reference_caption": blip_reference_caption,
                #"ofa_target_caption": ofa_target_caption,
                #"blip_target_caption": blip_target_caption
            }) 
        
            f.write(json.dumps(records[-1], indent=4) + "\n")
            f.flush()

        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="val")
    parser.add_argument("--preprocess", type=str, default="targetpad")
    parser.add_argument("--dimension", type=int, default=256)
    args = parser.parse_args()

    #data = json.load(open(args.data))
    #preprocess = get_preprocess(args.preprocess, args.dimension)
    dataset = Dataset(args.data, args.preprocess, args.dimension)
    dataset.__saveData__()
