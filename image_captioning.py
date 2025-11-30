from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator

mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 256

patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
                transforms.ToTensor(), 
                    transforms.Normalize(mean=mean, std=std)
                    ])

ckpt_dir='./OFA-huge'
tokenizer = OFATokenizer.from_pretrained(ckpt_dir) 

txt = "Describe the image."
inputs = tokenizer([txt], return_tensors="pt").input_ids
#image_id = 'val2014/COCO_val2014_000000245874.jpg'
#image_id = 'val2014/COCO_val2014_000000515660.jpg'
image_id = "val2014/COCO_val2014_000000136795.jpg"
#image_id = 'val2014/COCO_val2014_000000508443.jpg'
#image_id = 'val2014/COCO_val2014_000000277005.jpg'
#image_id = 'val2014/COCO_val2014_000000028714.jpg'
path_to_image = '/home/hle/spinning-storage/hle/LaSCo/' + image_id #val2014/COCO_val2014_000000028714.jpg'
img = Image.open(path_to_image)
patch_img = patch_resize_transform(img).unsqueeze(0)

model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)

generator = sequence_generator.SequenceGenerator(
            tokenizer=tokenizer,
            beam_size=3,
            max_len_b=16,
            min_len=5,
            no_repeat_ngram_size=3,
            )

import torch
data = {}
data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}

gen_output = generator.generate([model], data)
gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

# display(img)
print(tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip())
