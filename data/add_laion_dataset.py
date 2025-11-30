from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm 

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("/home/support/llm/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("/home/support/llm/Mistral-7B-Instruct-v0.2")

instruction = """Your mission is to reverse the meaning of the following sentence. You just need return ONE best result only. You donot need to explain your answer.
### Example
    Sentence: cars have been added.
    Result: cars have been removed.
### Sentence: {sentence}"""

def reverse_sentence(sentence):
    messages = [
            #{"role": "user", "content": 'reverse the meaning of "cabins has been added." and just return the result only. Just ONE!'}
            {
                "role": "user", 
                "content": instruction.format(sentence=sentence)
            }
            ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    result = decoded[0].split("[/INST]")[-1].strip().replace('</s>','').replace("Result: ","")
    return result

data = json.load(open("laion_combined_info.json"))
new_data = list()

for i in tqdm(range(len(data))):
    data_i = data[i]
    new_data.append(data_i)
    new_data_i = {
            "ref_image_id": data_i['tgt_image_id'], 
            "relative_cap": reverse_sentence(data_i['relative_cap']),
            "tgt_image_id": data_i['ref_image_id'], 
            "tgt_caption_opt": data_i['blip2_caption_opt'], 
            "blip2_caption_opt": data_i['tgt_caption_opt']
            }
    new_data.append(new_data_i)

with open("new_laion_combined.json", "w") as f:
    json.dump(new_data, f, indent = 4)



