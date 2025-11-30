import torch, tqdm, random, PIL, os
from PIL import Image
from torch import optim
import torch.nn.functional as F
from transform import targetpad_transform, squarepad_transform
from model import TransAgg
import numpy as np
from torch.utils.data import DataLoader
from clip import clip
from torchsummary import summary
from data.laion_dataset_template import LaionDataset_Template
from data.laion_dataset_llm import LaionDataset_LLM
from data.laion_dataset_combined import LaionDataset_Combined

#device = "cuda" if torch.cuda.is_available() else "gpu"

def get_model(cfg):
    model = TransAgg(cfg)
    model = model.to(cfg.device)
    return model

def get_preprocess(preprocess, input_dim):
    if preprocess == "squarepad":
        preprocess = squarepad_transform(input_dim) 
        print("Squarepad is used...")
    elif preprocess == "targetpad":
        target_ratio = 1.25
        preprocess = targetpad_transform(target_ratio, input_dim) 
        print("Targetpad is used...")
    else:
        raise ValueError("Invalid preprocess type!!!")
    return preprocess

def get_image(path):
    path = path.strip()
    image = PIL.Image.open(path).convert('RGB')

    #if image.mode == 'RGB':
    #    image = image.convert('RGB')
    #else:
    #    image = image.convert('RGBA')
    #print('DONE!')
    return image

def get_laion_cirr_dataset(preprocess, laion_type):
    #relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    #classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    if laion_type == 'laion_template':
        relative_train_dataset = LaionDataset_Template('train', preprocess)
    elif laion_type == 'laion_llm':
        relative_train_dataset = LaionDataset_LLM('train', preprocess)
    elif laion_type == 'laion_combined':
        relative_train_dataset = LaionDataset_Combined('train', preprocess)
    else:
        raise ValueError("laion_type should be in ['laion_template', 'laion_llm', 'laion_combined']")

    return relative_train_dataset#relative_val_dataset, classic_val_dataset

def set_grad(cfg, model):
    if cfg.encoder == 'text':
        for param in model.pretrained_model.visual.parameters():
            param.requires_grad = False
    elif cfg.encoder == 'both':
        print('Both encoders will be fine-tuned')
    elif cfg.encoder == 'neither':
        for param in model.pretrained_model.parameters():
            param.requires_grad = False    
    else:
        raise ValueError("encoder parameter should be in ['text', 'both', 'neither']")

def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_optimizer(model, cfg):
    pretrained_params = list(map(id, model.model.parameters()))
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and id(p) not in pretrained_params], 'weight_decay': cfg.weight_decay, 'lr': cfg.learning_rate},
      {'params': [p for n, p in model.named_parameters() if p.requires_grad and id(p) in pretrained_params], 'weight_decay': cfg.weight_decay, 'lr': 1e-6},
      ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    return optimizer 

def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    train_running_results['accumulated_train_loss'] += loss.item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch

def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    if train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'] < 0:
        print(train_running_results['accumulated_train_loss'], train_running_results['images_in_epoch'])
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
            f"train loss : {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )

def extract_index_features(dataset, model, return_local=True):
    feature_dim = model.feature_dim
    # feature_dim = model.reduced_dim
    model_name = model.model_name
    #print(dataset)
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=8,
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = torch.empty((0, feature_dim)).to(model.device, non_blocking=True) 
    index_total_features = []
    index_names = []
    # target_type = "fuse"
    for names, images in tqdm.tqdm(classic_val_loader):
        images = images.to(model.device, non_blocking=True)
        with torch.no_grad():
            #if type == "original":
            # print(images.shape)
            # if isinstance()
            if model_name.startswith("siglip"):
                # Normalize image tensor values to [0, 1] before processing
                images = torch.clamp(images, min=0.0, max=1.0)
                result = model.processor(images=images, return_tensors="pt").to(model.device)
            else:
                result = model.model.encode_image(images, return_local)
            if isinstance(result, tuple) and len(result) == 2:
                batch_features, batch_total_features = result
            else:
                batch_features = result
                batch_total_features = None 
            #texts = ["a photo similar" + captions[i] for i in range(len(images))]
            texts = [""] * len(images)
            # if model_name.startswith("clip"):
            #     tokenized_null_texts = clip.tokenize(texts, truncate=True).to(model.device)
            # elif model_name.startswith("blip"):
            #     tokenized_null_texts = model.model.tokenizer(texts, padding='max_length', truncation=True, max_length=35,return_tensors='pt').to(model.device)
            
            # if target_type == "sum":
            #     batch_features += model.model.encode_text(tokenized_null_texts)[0]
            # #print(captions)
            # elif target_type == "union":
            #     batch_features = model.union_features(images, texts) #+ model.combine_features(images, captions)
            # elif target_type == "fuse":
            #     batch_features = model.fuse_features(images, texts)
            # elif target_type == "combine":
            #     batch_features = model.combine_features(images, texts)
            batch_features = model.target_features(images)

            index_features = torch.vstack((index_features, batch_features))
            index_total_features.append(batch_total_features)
            index_names.extend(names)
    if return_local:
        with torch.no_grad():
            index_total_features = torch.cat(index_total_features, dim=0).to(model.device, non_blocking=True)
    else:
        index_total_features = None
    #print(index_features.shape)
    return index_features, index_names, index_total_features

# Decompose matrices to reduce memory usage and speed up similarity computation
def decompose_matrices(Q, I, B):
    U, S, Vt = torch.svd(torch.cat((Q, I), dim=0))
    V_B = Vt[:, :B]
    Q_B = torch.mm(Q, V_B)
    I_B = torch.mm(I, V_B)
    return Q_B, I_B 

# from sklearn.utils.extmath import randomized_svd

# def fast_decompose_matrices(Q, I, B):
#     X = np.vstack((Q, I))
#     U, S, Vt = randomized_svd(X, n_components=B, n_iter=3, random_state=42)
#     V_B = Vt.T
#     Q_B = Q @ V_B
#     I_B = I @ V_B
#     return Q_B, I_B

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt.detach().cpu().numpy()

def sim_matrix_mm(a, b):
    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt.detach().cpu().numpy()

def softmax(x, dim = 0):
    x = torch.tensor(x)
    x = F.softmax(x, dim = dim)
    return x.detach().cpu().numpy() 

# From https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/1e66a417afa2247edde6d35f3a9a2a465778a3a8/cirtorch/utils/evaluate.py#L3

def compute_ap(ranks, nres):
    nimgranks = len(ranks)
    ap = 0
    recall_step = 1.0 / (nres + 1e-5)
    for j in np.arange(nimgranks):
        rank = ranks[j]
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.0
    return ap

def compute_map(correct):
    map = 0.0
    nq = correct.shape[0]
    ap_list = []
    for i in np.arange(nq):
        pos = np.where(correct[i] != 0)[0]
        ap = compute_ap(pos, len(pos))
        ap_list.append(ap)
        map = map + ap
    map = map / (nq)
    return np.around(map * 100, decimals=2) , ap_list

def recall_at_k(labels, k):
    if not isinstance(labels, torch.Tensor):
        raise ValueError("Labels should be a torch Tensor.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k should be a positive integer.")
    if labels.ndim != 2:
        raise ValueError("Labels should be a 2D tensor.")

    labels_at_k = labels[:, :k] 
    score = 0
    for i in range(len(labels)):
        pred_score = labels_at_k[i]
        score += 1 if torch.sum(pred_score)>0 else 0 
    return np.around(score/len(labels) * 100, decimals=2)

def precision_at_k(labels, k):
    if not isinstance(labels, torch.Tensor):
        raise ValueError("Labels should be a torch Tensor.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k should be a positive integer.")
    if labels.ndim != 2:
        raise ValueError("Labels should be a 2D tensor.")

    labels_at_k = labels[:, :k]
    score = 0 
    for i in range(len(labels)):
        pred_score = labels_at_k[i]
        score += torch.sum(pred_score) / k
    return np.around(score/len(labels) * 100, decimals=2)
