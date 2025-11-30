from torch.utils.data import Dataset
from typing import List
import json 
import PIL 

home_path = "/home/hle/CIR/data/fiq"

class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable):
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
        self.dress_types = dress_types
        self.split = split

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        #self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(f'{home_path}/captions/cap.{dress_type}.{split}_llava.json') as f:
                self.triplets = json.load(f)

        # get the image names and captions
        for dress_type in dress_types:
            with open(f'{home_path}/image_splits/split.{dress_type}.{split}.json') as f:
                self.image_names = json.load(f)
            
            with open(f'{home_path}/captions/classic.cap.{dress_type}.{split}_llava.json') as f:
                self.classic_image_captions = json.load(f)

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        def preprocess_cap(caption):
            caption = caption.replace('t - shirt', 't-shirt')
            def replace_word(s, w):
                if w not in s:
                    return s
                id = s.find(w)
                s = s[id+len(w):]
                return s 
            for word in ['man', 'men', 'women', 'woman', 'person']:
                w = f" {word} "
                caption = replace_word(caption, w)
            return caption

        try:
            if self.mode == 'relative':
                image_caption = " and ".join(self.triplets[index]['captions'])
                addition = " with " + preprocess_cap(self.triplets[index]['llava_target_caption'])
                #image_captions = [[i[0], i[1] + addition] for i in image_captions]
                #image_caption += addition
                #image_caption = "a photo that " + image_caption
                #image_caption = "find the similar image"
                reference_name = self.triplets[index]['candidate']

                if self.split == 'train':
                    reference_image_path = f'{home_path}/images/{reference_name}.png'
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_name = self.triplets[index]['target']
                    target_image_path = f'{home_path}/images/{target_name}.png'
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, image_caption

                elif self.split == 'val':
                    reference_image_path = f'{home_path}/images/{reference_name}.png'
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    reference_cap = self.triplets[index]['llava_reference_caption']
                    target_cap = self.triplets[index]['llava_target_caption']
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_caption, reference_image, reference_cap 

                elif self.split == 'test':
                    reference_image_path = f'{home_path}/images/{reference_name}.png'
                    reference_cap = self.triplets[index]['llava_reference_caption']
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_caption

            # elif self.mode == 'classic':
            #     image_name = self.image_names[index]
            #     image_path = f'{home_path}/images/{image_name}.png'
            #     image = self.preprocess(PIL.Image.open(image_path))
            #     return image_name, image  

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = f'{home_path}/images/{image_name}.png'
                image = self.preprocess(PIL.Image.open(image_path))
                caption = self.classic_image_captions[image_name]
                return image_name, image#, caption

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
