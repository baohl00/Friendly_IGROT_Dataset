from torch.utils.data import Dataset
from typing import List
import json 
import PIL 

home_path = "/home/hle/SBIR/PKUSketchRE-ID_V1"

class PKUSketchDataset(Dataset):

    def __init__(self, split: str, mode: str, preprocess: callable):

        self.fiq_path_prefix = home_path
        self.mode = mode
        self.split = split

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['train', 'val']")
    

        self.preprocess = preprocess

        # get queries
        self.queries = open(f"{home_path}/sketch.txt").readlines()

        # get the image names and captions
        self.targets = open(f"{home_path}/photo.txt").readlines()

        print(f"PKUSketch dataset finished!")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                query = self.queries[index].strip()
                #image_caption = "this {} object and same shape"
                #image_caption = "a similar image" #"the image has the same object"
                image_caption = "an image of this sketch"
                if self.split == 'val':
                    reference_image_path = query
                    class_id = int(query.split(".")[0])
                    #print(self.domain_id)
                    reference_image_path = home_path + "/sketch/" + reference_image_path
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = class_id
                    # image_caption = image_caption.format(self.all_domains[class_id])
                    return class_id, target_name, image_caption, reference_image, ""

            elif self.mode == 'classic':
                target = self.targets[index].strip()
                image_path = target 
                iid = int(target.split("_")[0])
                image_path = home_path + "/photo/" + image_path 
                image = self.preprocess(PIL.Image.open(image_path))
                return iid, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.queries)
        elif self.mode == 'classic':
            return len(self.targets)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
