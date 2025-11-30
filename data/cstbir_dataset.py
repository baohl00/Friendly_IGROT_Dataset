from torch.utils.data import Dataset
from typing import List
import json 
import PIL 

home_path = "/home/hle/SBIR/TUBerlin"

class TUBerlinDataset(Dataset):
    """
    DTIN dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, domain_type: List[str], mode: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.fiq_path_prefix = home_path
        self.mode = mode
        self.domain_type = domain_type
        self.split = split

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['train', 'val']")
    
        all_domains = open(f"{home_path}/zeroshot/cname_cid_zero.txt").readlines()
        self.all_domains = [" ".join(text.split()[:-1]) for text in all_domains]
        self.target_domain = self.domain_type 
        self.domain_id = self.all_domains.index(self.target_domain)

        self.preprocess = preprocess

        # get queries
        #self.triplets: List[dict] = []
        self.queries = open(f"{home_path}/zeroshot/png_ready_filelist_zero.txt").readlines()
        #self.queries = [query for query in self.queries if self.domain_id == query.split()[-1]]

        # get the image names and captions
        self.targets = open(f"{home_path}/zeroshot/ImageResized_ready_filelist_zero.txt").readlines()#[:6983]  

        print(f"TUBerlin dataset finished!")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                query = self.queries[index].strip()
                #image_caption = "this {} object and same shape"
                #image_caption = "a similar image" #"the image has the same object"
                image_caption = ""
                if self.split == 'val':
                    reference_image_path = " ".join(query.split()[:-1])
                    class_id = int(query.split()[-1])
                    #print(self.domain_id)
                    reference_image_path = home_path + "/" + reference_image_path 
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = class_id
                    image_caption = image_caption.format(self.all_domains[class_id])
                    #if class_id == self.domain_id:  
                    return class_id, target_name, image_caption, reference_image, ""

            elif self.mode == 'classic':
                target = self.targets[index].strip()
                image_path = " ".join(target.split()[:-1])
                iid = target.split()[-1]
                image_path = home_path + "/" + image_path 
                image = self.preprocess(PIL.Image.open(image_path))
                return int(iid), image

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
