from torch.utils.data import Dataset
import json, os, random
import PIL 
from PIL import Image
from utils import get_image
import torch.nn.functional as F
import numpy as np
Image.MAX_IMAGE_PIXELS = 2300000000
 
home_path = "/home/hle/IGROT"
        
class IGROTDataset(Dataset):
    """
       IGROT dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
                - (reference_caption, target_caption) when self.type == blip or llava as the used model for the captions
    """

    def __init__(self, split: str, mode: str, preprocess: callable, data_amount: int = 1000):
        """
        :param split: dataset split, should be in ['train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        """
        self.path_prefix = home_path
        self.mode = mode
        self.split = split
        self.preprocess = preprocess

        if split not in ['train', 'val', 'test']:
            raise ValueError("split should be in ['train', 'val', 'test']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']") 
        # get triplets made by (reference_image, target_image, relative caption)
        # with open(f'{self.lasco_path_prefix}/{self.split}_blip_formatted.json') as f:
        with open(f'{self.path_prefix}/data/{self.split}.json') as f:
            if data_amount > 0 and self.split == "train":
                self.triplets = json.load(f)[:data_amount]
            else:
                self.triplets = json.load(f)
            
        # get a mapping from image name to relative path
        self.image_folder_path = f'{self.path_prefix}/images'

        if self.split != "train":
            with open(f'{self.path_prefix}/data/final_corpus.json') as f:
                self.name_list = json.load(f)
                # Shuffle imgs_info 
                # random.shuffle(imgs_info)

            # if self.split == "val" or self.split == "test": 
            #     self.name_list = [img_info['id'] for img_info in imgs_info]
            # else:
            #     self.name_list = list(imgs_info.keys())
        print(f"IGROT {split} dataset in {mode} mode initialized")
 
    def __getitem__(self, index):
        # Format image_id
        def image_path2name(image_path):
            cir_path = "/home/hle/CIR/data/transagg_data" 
            coco_path = "/home/hle/CIR/data/circo"
            sbir_path = "/home/hle/SBIR/TUBerlin"
            if "laion" in image_path: 
                return f"{cir_path}/" + image_path
            elif "COCO" in image_path:
                return f"{coco_path}/" + image_path
            else:
                return f"{sbir_path}/" + image_path
        def sbir_caption():
            captions = [ 
                "a real image of this sketch", 
                "an image illustrating this sketch",
                "a real image that looks like this sketch",
                "a real image that is similar to this sketch",
                "a real image that is related to this sketch"
            ]
            return np.random.choice(captions)
        try:
            if self.mode == 'relative':
                if self.split == "train": 
                    reference_image_id = self.triplets[index]["reference_image"]
                    reference_image_path = image_path2name(self.triplets[index]["reference_image"])

                    #print(reference_image_id)
                    target_image_id = self.triplets[index]["target_image"]
                    target_image_path = image_path2name(self.triplets[index]["target_image"])

                    # Read and preprocess images
                    reference_image = self.preprocess(get_image(reference_image_path)) #.unsqueeze(0) #.to(device)
                    reference_image = F.normalize(reference_image, dim = -1)
                    relative_caption = self.triplets[index]["query"]
                    target_image = self.preprocess(get_image(target_image_path)) #.unsqueeze(0) #.to(device)
                    target_image = F.normalize(target_image, dim = -1)
                    return reference_image, relative_caption, target_image, "", ""


                elif self.split == "val" or self.split == "test":
                    pair_id = self.triplets[index]["id"]
                    reference_image_name = self.triplets[index]["reference_image"]
                    group_members = self.triplets[index]["target_images"]
                    if self.triplets[index]["class"] == "sbir":
                        relative_caption = sbir_caption()
                    elif self.triplets[index]["class"] == "sbtir":
                        relative_caption = 'The sketch that ' + self.triplets[index]["query"]
                    else:
                        relative_caption = self.triplets[index]["query"]
                    reference_image = self.preprocess(get_image(image_path2name(reference_image_name)))
                    return pair_id, reference_image_name, relative_caption, group_members, reference_image

            
            elif self.mode == 'classic':
                image_name = self.name_list[index]['file_name']
                image_path = image_path2name(image_name)
                image = self.preprocess(get_image(image_path))
                image = F.normalize(image, dim = -1)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            #pass
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_list)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
