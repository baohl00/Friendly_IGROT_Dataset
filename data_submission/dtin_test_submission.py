import json
import multiprocessing
from typing import List, Tuple, Dict
import numpy as np
import torch, gc 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import TransAgg
from utils import get_preprocess, extract_index_features, compute_map, recall_at_k, precision_at_k, sim_matrix, sim_matrix_mm, softmax
from data.dtin_dataset import DTINDataset
#from tabulate import tabulate 
import tabulate
import os

SEED = 42
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:512"


torch.cuda.empty_cache()
gc.collect()
@torch.no_grad()
def dtin_generate_val_predictions(relative_val_dataset: DTINDataset, model, device) -> Tuple[torch.Tensor, List[str], List[str]]:
        """
        Generates features predictions for the validation set of Fashion IQ.
        """
        # Create data loader
        print(f"Compute DTIN val predictions")
        relative_val_loader = DataLoader(dataset = relative_val_dataset, batch_size = 256, num_workers = multiprocessing.cpu_count(), pin_memory=False)

        predicted_features = []
        reference_names = [] 
        target_names = []

        # Compute features
        for batch_reference_names, batch_target_names, image_captions, batch_reference_images, batch_reference_captions in tqdm(relative_val_loader):
            with torch.no_grad():
                batch_reference_images = batch_reference_images.to(device)
                batch_predicted_features = model.combine_features(batch_reference_images, image_captions) \
                        + model.union_features(batch_reference_images, image_captions) * 0.1
                if type(batch_predicted_features) == tuple:
                    batch_predicted_features = batch_predicted_features[0]
                predicted_features.append(batch_predicted_features / batch_predicted_features.norm(dim = -1, keepdim = True))

            reference_names.extend(batch_reference_names)
            target_names.extend(batch_target_names)

        predicted_features = torch.cat(predicted_features, dim = 0)
        predicted_features = F.normalize(predicted_features.float())
        #print(predicted_features) #, target_names, len(target_names))
        return predicted_features, target_names 

@torch.no_grad()
def dtin_compute_val_metrics(relative_val_dataset: DTINDataset, model, index_features: torch.Tensor, index_names: List[str], device) -> Dict[str, float]:
        """
        Compute the retrieval metrics on the FashionIQ validation set given the dataset, pseudo tokens and the reference names
        """
        # Generate the predicted features
        predicted_features, target_names = dtin_generate_val_predictions(relative_val_dataset, model, device)

        # Move the features to the device
        index_features = index_features.to(device)
        predicted_features = predicted_features.to(device)
        # Normalize the features
        index_features = F.normalize(index_features.float())
    
        ranking_type = 'normal' # normal, dual
        # Compute the distances and sort the results
        if ranking_type == 'dual':
            similarity = sim_matrix_mm(predicted_features, index_features)
            similarity = softmax(similarity/0.1, dim = 1) * similarity
            similarity = softmax(similarity, dim = 0)
            distances = 1 - similarity #predicted_features@index_features.T
            sorted_indices = np.argsort(distances, axis=1)
        else:
            distances = 1 - F.normalize(predicted_features @ index_features.T, dim = -1)
            sorted_indices = torch.argsort(distances, dim=-1).cpu()
        
        sorted_index_names = np.array(index_names)[sorted_indices]
        # Remove reference images in predicted list 
        for i in range(len(sorted_index_names)):
            try: 
                sorted_index_names[i].remove(index_names[i])
            except:
                pass 
        # Check if the target names are in the top 10 and top 50

        labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
  
        answers = dict()
        for i in range(len(target_names)):
            key = target_names[i]
            value = list(sorted_index_names[i][:50])
            answers[key] = value

        # Compute the metrics
        recall_at10 = recall_at_k(labels, 10)
        recall_at50 = recall_at_k(labels, 50)
        #print(*labels.int().cpu().numpy()[0])
        # Remove the reference images
        for i in range(len(sorted_index_names)):
            try:
                sorted_index_names[i].remove(index_names[i])
            except:
                pass
        # Check if the target names are in the top 10 and top 50
        labels = torch. tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
   
        # Compute the metrics
        #recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
        #recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

        map_score, _ = compute_map(labels[:,:].int().cpu().numpy())
        return {
                'recall_at10': recall_at10, 
                'recall_at50': recall_at50, 
                'mAP': map_score
                }

@torch.no_grad()
def dtin_val_retrieval(domain_type: str, model, preprocess: callable, device) -> Dict[str, float]:          
    """
    Compute the retrieval metrics on the FashionIQ validation set given the pseudo tokens and the reference names
    """

    # Extract the index features
    classic_val_dataset = DTINDataset('val', domain_type, 'classic', preprocess)
    index_features, index_names, captions = extract_index_features(classic_val_dataset, model, return_local=False)
    
    # Define the relative dataset
    relative_val_dataset = DTINDataset('val', domain_type, 'relative', preprocess)
    return dtin_compute_val_metrics(relative_val_dataset, model, index_features, index_names, device)

def main_dtin(cfg):
    model = bachbi(cfg)
    #unicom = unicom(cfg)
    device = cfg.device 
    model = model.to(device)
    model.load_state_dict(torch.load(cfg.val_load_path))

    if cfg.model.startswith("blip"):
        input_dim = 384
    elif cfg.model.startswith("clip"):
        input_dim = model.model.visual.input_resolution

    preprocess = get_preprocess(cfg.preprocess, input_dim=input_dim)

    model.eval()
    all_domains = ['cartoon', 'origami', 'toy', 'sculpture'] 
    results = list() 
    if cfg.val_dataset.lower() == 'dtin':
        recalls_at10 = []
        recalls_at50 = []
        mAP = []
        for domain_type in all_domains:
            metrics = dtin_val_retrieval(domain_type, model, preprocess, device)
            results.append([
                domain_type, 
                metrics['recall_at10'], 
                metrics['recall_at50'], 
                metrics['mAP']])
            recalls_at10.append(metrics['recall_at10'])
            recalls_at50.append(metrics['recall_at50']) 
            mAP.append(metrics['mAP'])
            
    results.append(['Average', 
        sum(recalls_at10)/len(all_domains), 
        sum(recalls_at50)/len(all_domains),
        sum(mAP)/len(all_domains)])
    results.append(['Final avg', '->', 
        (sum(recalls_at10)+sum(recalls_at50))/(len(all_domains)*2), '-'])

    print(tabulate.tabulate(results, headers=['Type', 'R@10', 'R@50', 'mAP']))
