from torch.utils.data import Dataset
import json 
import PIL 
from PIL import Image 
import pandas as pd
Image.MAX_IMAGE_PIXELS = 2300000000
 
home_path = "/home/hle/CIR/data/PatternNet"
        
class PatternComDataset(Dataset):

    def __init__(self, split: str, mode: str, preprocess: callable):
        self.path_prefix = home_path
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['test']:
            raise ValueError("split should be in ['test']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']") 
        # get triplets made by (reference_image, target_image, relative caption)
        with open(f'{self.path_prefix}/test.json', 'r') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        self.image_folder_path = f'{self.path_prefix}/patternnet.json'

        self.name_list = json.load(open(self.image_folder_path, 'r'))

        self.name_to_path = json.load(open(f'{self.path_prefix}/name_to_path.json', 'r'))

        print(f"PatternCom {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        
        try:
            if self.mode == 'relative':
                # print(self.triplets[index]['query'])
                reference_name = self.triplets[index]['query']
                group_members = self.triplets[index]['label']
                # rel_caption = 'Same image but have ' + self.triplets[index]['text'] + ' ' + self.triplets[index]['class']
                rel_caption = 'the ' + self.triplets[index]['text'] + ' and ' + self.triplets[index]['class'] + ' in the image'
                # rel_caption = self.triplets[index]['text']
                reference_image_path = self.path_prefix + '/' + self.name_to_path[self.triplets[index]['query']][1:]
                reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert('RGB'))
                pair_id = index
                return pair_id, reference_name, rel_caption, group_members, reference_image 

            elif self.mode == 'classic':
                image_name = self.name_list[index]['name']
                image_path = self.path_prefix + '/' + self.name_list[index]['path'][1:]
                im = PIL.Image.open(image_path).convert("RGB")
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_list)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
