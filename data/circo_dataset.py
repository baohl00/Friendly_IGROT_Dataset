from torch.utils.data import Dataset
import json 
import PIL 
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 2300000000
 
home_path = "/home/hle/CIR/data/circo"
        
class CIRCODataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.circo_path_prefix = home_path
        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        if split not in ['test1', 'test', 'val']:
            raise ValueError("split should be in ['test1', 'test', 'val'")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']") 
        # get triplets made by (reference_image, target_image, relative caption)
        with open(f'{self.circo_path_prefix}/annotations/{split}.json') as f:
            self.triplets = json.load(f)#[:10]

        # get a mapping from image name to relative path
        self.image_folder_path = f'{self.circo_path_prefix}/COCO2017_unlabeled/unlabeled2017'
        
        with open(f'{self.circo_path_prefix}/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json') as f:
            imgs_info = json.load(f)

        #self.name_to_relpath = self.triplets
        self.name_list = [img_info['id'] for img_info in imgs_info["images"]] #[:40000]

        print(f"CIRCO {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
         # format image_id
        def image_id2name(image_id):
            return str(image_id).zfill(12) + '.jpg'


        try:
            if self.mode == 'relative':
                reference_name = image_id2name(self.triplets[index]['reference_img_id'])
                group_members = [reference_name]
                rel_caption =  self.triplets[index]['shared_concept'].lower() + ' but ' + self.triplets[index]['relative_caption'].lower()
                reference_image_path = f"{self.image_folder_path}/" + reference_name
                reference_image = self.preprocess(PIL.Image.open(reference_image_path).convert('RGB'))
                pair_id = self.triplets[index]['reference_img_id']
                return pair_id, reference_name, rel_caption, group_members, reference_image 

            elif self.mode == 'classic':
                image_name = self.name_list[index]
                image_path = f"{self.image_folder_path}/" + image_id2name(image_name)
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
