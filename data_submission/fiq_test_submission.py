import json
import multiprocessing
from typing import List, Tuple, Dict
import numpy as np
import torch, gc 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import TransAgg
from utils import \
    get_preprocess, extract_index_features, \
    sim_matrix, sim_matrix_mm, softmax, decompose_matrices
from data.fiq_dataset import FashionIQDataset
#from tabulate import tabulate 
import tabulate
import os
from collections import defaultdict

SEED = 42
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:512"


torch.cuda.empty_cache()
gc.collect()
@torch.no_grad()
def fiq_generate_val_predictions(relative_val_dataset: FashionIQDataset, model, device) -> Tuple[torch.Tensor, List[str], List[str]]:
        """
        Generates features predictions for the validation set of Fashion IQ.
        """
        # Create data loader
        print(f"Compute FashionIQ val predictions")
        relative_val_loader = DataLoader(dataset = relative_val_dataset, batch_size = 256, num_workers = multiprocessing.cpu_count(), pin_memory=False)

        predicted_features = []
        reference_names = [] 
        target_names = []

        # Compute features
        for batch_reference_names, batch_target_names, image_captions, batch_reference_images, batch_reference_captions in tqdm(relative_val_loader):
            with torch.no_grad():
                batch_reference_images = batch_reference_images.to(device)
                #print(batch_reference_captions)
                #null_captions = [''] * len(batch_reference_images)
                batch_predicted_features = model.final_features(batch_reference_images, image_captions) 
                        #+ model.union_features(batch_reference_images, image_captions) * 0.1 #, batch_reference_captions)
                if type(batch_predicted_features) == tuple:
                    batch_predicted_features = batch_predicted_features[0]
                predicted_features.append(batch_predicted_features / batch_predicted_features.norm(dim = -1, keepdim = True))

            reference_names.extend(batch_reference_names)
            target_names.extend(batch_target_names)

        predicted_features = torch.cat(predicted_features, dim = 0)
        predicted_features = F.normalize(predicted_features.float())

        return predicted_features, target_names 

@torch.no_grad()
def fiq_compute_val_metrics(relative_val_dataset: FashionIQDataset, model, index_features: torch.Tensor, index_names: List[str], device) -> Dict[str, float]:
        """
        Compute the retrieval metrics on the FashionIQ validation set given the dataset, pseudo tokens and the reference names
        """
        # Generate the predicted features
        predicted_features, target_names = fiq_generate_val_predictions(relative_val_dataset, model, device)

        # Move the features to the device
        index_features = index_features.to(device)
        predicted_features = predicted_features.to(device)
        # Normalize the features
        index_features = F.normalize(index_features.float())

        ranking_type = 'decom' # normal, dual, decom
        # Compute the distances and sort the results
        if ranking_type == 'dual':
            similarity = sim_matrix_mm(predicted_features, index_features)
            similarity = softmax(similarity/0.1, dim = 1) * similarity
            similarity = softmax(similarity, dim = 0)
            distances = 1 - similarity #predicted_features@index_features.T
            sorted_indices = np.argsort(distances, axis=1)
        elif ranking_type == 'decom':
            predicted_features_decom, index_features_decom = decompose_matrices(predicted_features, index_features, 128)
            distances = 1 - predicted_features_decom @ index_features_decom.T
            sorted_indices = torch.argsort(distances, dim=-1).cpu()
        else:
            distances = 1 - predicted_features @ index_features.T
            sorted_indices = torch.argsort(distances, dim=-1).cpu()

        sorted_index_names = np.array(index_names)[sorted_indices]

        # Remove the reference images
        for i in range(len(sorted_index_names)):
            try: 
                sorted_index_names[i].remove(index_names[i])
            except:
                pass 
        # Check if the target names are in the top 10 and top 50
        labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))                                                     
        assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
        
        answers = dict()
        for i in range(len(target_names)):
            key = target_names[i]
            value = list(sorted_index_names[i][:50])
            answers[key] = value 

        dress = relative_val_dataset.dress_types[0] 

        # Compute the metrics
        recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
        recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
        
        return {'fiq_recall_at10': recall_at10, 'fiq_recall_at50': recall_at50}

def fiq_compute_decompose_val_metrics(relative_val_dataset: FashionIQDataset, model, index_features: torch.Tensor, index_names: List[str], device) -> Dict[str, float]:
        """
        Compute the retrieval metrics on the FashionIQ validation set given the dataset, pseudo tokens and the reference names
        """
        # Generate the predicted features
        predicted_features, target_names = fiq_generate_val_predictions(relative_val_dataset, model, device)

        # Move the features to the device
        index_features = index_features.to(device)
        predicted_features = predicted_features.to(device)
        # Normalize the features
        index_features = F.normalize(index_features.float())

        # Compute the distances and sort the results
        K = [768, 256, 128, 64, 32, 16]
        
        results = []
        for k_value in K: 
            predicted_features_decom, index_features_decom = decompose_matrices(predicted_features, index_features, k_value)
            distances = 1 - predicted_features_decom @ index_features_decom.T
            sorted_indices = torch.argsort(distances, dim=-1).cpu()
            
            sorted_index_names = np.array(index_names)[sorted_indices]

            # Remove the reference images
            for i in range(len(sorted_index_names)):
                try: 
                    sorted_index_names[i].remove(index_names[i])
                except:
                    pass 
            # Check if the target names are in the top 10 and top 50
            labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))                                                     
            assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
            
            answers = dict()
            for i in range(len(target_names)):
                key = target_names[i]
                value = list(sorted_index_names[i][:50])
                answers[key] = value 

            dress = relative_val_dataset.dress_types[0] 

            # Compute the metrics
            recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
            recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
            
            results.append({k_value: [recall_at10, recall_at50]})
        return results 

@torch.no_grad()
def fiq_val_retrieval(dress_type: str, model, preprocess: callable, device) -> Dict[str, float]:          
    """
    Compute the retrieval metrics on the FashionIQ validation set given the pseudo tokens and the reference names
    """

    # Extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    index_features, index_names, captions = extract_index_features(classic_val_dataset, model, return_local=False)
    
    # Define the relative dataset
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
    return fiq_compute_val_metrics(relative_val_dataset, model, index_features, index_names, device)

def fiq_decom_val_retrieval(dress_type: str, model, preprocess: callable, device) -> List[List[float]]:
    """
    Compute the retrieval metrics on the FashionIQ validation set given the pseudo tokens and the reference names
    """
    # Extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    index_features, index_names, captions = extract_index_features(classic_val_dataset, model, return_local=False)
    
    # Define the relative dataset
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
    return fiq_compute_decompose_val_metrics(relative_val_dataset, model, index_features, index_names, device)

def main_fiq(cfg):
    model = TransAgg(cfg)
    #unicom = unicom(cfg)
    device = cfg.device 
    model = model.to(device)
    model.load_state_dict(torch.load(cfg.val_load_path))

    if cfg.model.startswith("blip"):
        input_dim = 384
    elif cfg.model.startswith("clip"):
        input_dim = model.model.visual.input_resolution
    elif cfg.model.startswith("siglip"): 
        input_dim = 384
    preprocess = get_preprocess(cfg.preprocess, input_dim=input_dim)
    all_types = ['dress', 'shirt', 'toptee']
    model.eval()
    decom = False 
    results = list() 

    if decom: 
        # Calculate average of each value by dimension (key)
        # fiq_metrics is a list of dicts: [{k: [r10, r50]}, ...] for each dress_type
        # We want to aggregate by key (k) and average r10, r50 across dress_types

        # Collect all results for each k
        k_results = defaultdict(list)
        for dress_type in all_types:
            fiq_metrics = fiq_decom_val_retrieval(dress_type, model, preprocess, device)
            for d in fiq_metrics:
                for k, v in d.items():
                    k_results[k].append(v)

        # Compute averages
        results = []
        for k in sorted(k_results.keys(), reverse=True):
            vals = np.array(k_results[k])
            avg_r10 = np.mean(vals[:, 0])
            avg_r50 = np.mean(vals[:, 1])
            results.append([f'k={k}', avg_r10, avg_r50])

        print(tabulate.tabulate(results, headers=['Decompose K', 'Avg R@10', 'Avg R@50']))
        # result = {}
        # for dress_type in all_types:
        #     fiq_metrics = fiq_decom_val_retrieval(dress_type, model, preprocess, device)


    else:  
        if cfg.val_dataset.lower() == 'fiq':
            recalls_at10 = []
            recalls_at50 = []
            
            for dress_type in all_types:
                fiq_metrics = fiq_val_retrieval(dress_type, model, preprocess, device)
                results.append(
                        [dress_type, fiq_metrics['fiq_recall_at10'], fiq_metrics['fiq_recall_at50']])
                recalls_at10.append(fiq_metrics['fiq_recall_at10'])
                recalls_at50.append(fiq_metrics['fiq_recall_at50'])
                
        results.append(
            ['Average', sum(recalls_at10)/len(all_types), sum(recalls_at50)/len(all_types)])
        results.append(
                ['Final avg', '->', (sum(recalls_at10)+sum(recalls_at50))/(len(all_types)*2)])
        #print(pd.DataFrame(results))
        print(tabulate.tabulate(results, headers=['Type', 'R@10', 'R@50']))
