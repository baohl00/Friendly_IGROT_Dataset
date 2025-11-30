import os
import json
from pathlib import Path
import torch
from tqdm import tqdm
import PIL.Image as Image
import torch.nn as nn 
from lavis.models import load_model_and_preprocess

#from torchvision import transforms
#from transformers import OFATokenizer, OFAModel#, AutoProcessor, BlipForConditionalGeneration
#from transformers.models.ofa.generate import sequence_generator
from utils import get_image, get_preprocess
import argparse 

MAIN_PATH = '/home/hle/spinning-storage/hle/LaSCo' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#def OFA_caption(img):
#    ckpt_dir='./OFA-huge'
#    with torch.inference_mode():
#        tokenizer = OFATokenizer.from_pretrained(ckpt_dir) 
#
#        txt = "Describe the image."
#        inputs = tokenizer([txt], return_tensors="pt").input_ids.to(device)
#        patch_img = img.to(device) #.unsqueeze(0).to(device)
#
#        model = OFAModel.from_pretrained(ckpt_dir, use_cache=False).to(device)
#
#        generator = sequence_generator.SequenceGenerator(
#                tokenizer=tokenizer,
#                beam_size=3,
#                max_len_b=16,
#                min_len=5,
#                no_repeat_ngram_size=3,
#                )
#        data = {}
#        data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True]).to(device)}
#    
#        gen_output = generator.generate([model], data)
#        gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]
#    
#        caption = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
#        torch.cuda.empty_cache() 
#    return caption 
model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
            )
    
def BLIP_caption(img):
    # model_name = 'Salesforce/blip2-opt-6.7b'
    model_name = 'Salesforce/blip2-flan-t5-xl'
    #processor = AutoProcessor.from_pretrained(model_name)
    #model = BlipForConditionalGeneration.from_pretrained(model_name).to(device) 
    img = vis_processors["eval"](img).unsqueeze(0).to(device)
    caption = model.generate({"image": img})
    #generated_ids = model.generate(**img, max_new_tokens=16)
    #caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption[0]

class Dataset():
    def __init__(self, data, preprocess, dimension):
        self.data_type = data 
        self.path = MAIN_PATH 
        with open(f"{self.path}/lasco_{data}.json", "r") as f:
            #self.data = json.load(f)[55624:]
            self.data = json.load(f)
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
        f = open(f"{MAIN_PATH}/{self.data_type}_blip.json", "w")
        for i, record in enumerate(tqdm(self.data)):
            reference_image, relative_caption, target_image, reference_image_id, target_image_id = self.__getitem__(record)
            #print(reference_image_id, target_image_id)
            #ofa_reference_caption = OFA_caption(reference_image) 
            #ofa_target_caption = OFA_caption(target_image)
            blip_reference_caption = BLIP_caption(reference_image)
            blip_target_caption = BLIP_caption(target_image)
            #print(relative_caption, ofa_reference_caption, ofa_target_caption, blip_reference_caption, blip_target_caption)
            
            records.append({
                "reference_image_id": reference_image_id,
                "relative_caption": relative_caption,
                "target_image_id": target_image_id, 
                #"ofa_reference_caption": ofa_reference_caption,
                "blip_reference_caption": blip_reference_caption,
                #"ofa_target_caption": ofa_target_caption,
                "blip_target_caption": blip_target_caption
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
