from torch.utils.data import Dataset
from typing import List
import json 
import PIL 
import pandas as pd
import os.path as osp

home_path = "/home/hle/lifelogQA"
data_path = "/home/hle/spinning-storage/tranl/LSC23"

class LSCDataset(Dataset):
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
        self.path_prefix = home_path
        self.image_folder_path = data_path
        self.mode = mode
        self.split = split

        if mode not in ['relative', 'event']:
            raise ValueError("mode should be in ['relative', 'event']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")

        self.preprocess = preprocess

        self.data = pd.read_csv(osp.join(self.path_prefix, f"{self.mode}_data.csv"), encoding='ISO-8859-1', on_bad_lines='skip')

        # get the event information 
        self.event_data = pd.read_csv(osp.join(self.path_prefix, "event_description.csv"), encoding='ISO-8859-1', on_bad_lines='skip') 

        print(f"LSC - LifelogQA {split} mode is initialized")

    def __getitem__(self, index):
        def get_image(date):
            day = date.split('_')[0]
            folder = day[:6]
            day = day[6:]
            image_path = osp.join(self.image_folder_path, folder, day)
            image = self.preprocess(get_image(image_path)) 
            return image
        try:
            if self.mode == 'relative':
                data_i = self.data.iloc[index]
                question = data_i['question']
                answer = data_i['answer']
                ImageID = data_i['ImageID']
                
                event_date = data_i['event_date']
                # print(even_path)
                return question, answer, ImageID, event_date 

            elif self.mode == 'event':
                data_i = self.event_data.iloc[index]
                date = data_i['date']
                description = data_i['description']
                event_date = data_i['event_date']
                new_name = data_i['new_name']
                city = data_i['city']
                local_time = data_i['local_time']
                context = data_i['context']
                image = get_image(date)
                return image, date, description, event_date, new_name, city, local_time, context

            else:
                raise ValueError("mode should be in ['relative', 'event']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.data)
        elif self.mode == 'event':
            return len(self.event_data)
        else:
            raise ValueError("mode should be in ['relative', 'event']")
