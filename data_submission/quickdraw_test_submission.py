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
from data.quickdraw_dataset import QuickDrawDataset
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
def generate_val_predictions(relative_val_dataset: QuickDrawDataset, model, device) -> Tuple[torch.Tensor, List[str], List[str]]:
        """
        Generates features predictions for the validation set of QuickDraw.
        """
        # Create data loader
        print(f"Compute QuickDraw val predictions")
        relative_val_loader = DataLoader(dataset = relative_val_dataset, batch_size = 32, num_workers = multiprocessing.cpu_count(), pin_memory=False)

        predicted_features = []
        reference_names = [] 
        target_names = []

        # Compute features
        for batch_reference_names, batch_target_names, image_captions, batch_reference_images, batch_reference_captions in tqdm(relative_val_loader):
            with torch.no_grad():
                batch_reference_images = batch_reference_images.to(device)
                batch_predicted_features = model.final_features(batch_reference_images, image_captions)
                if type(batch_predicted_features) == tuple:
                    batch_predicted_features = batch_predicted_features[0]
                predicted_features.append(batch_predicted_features / batch_predicted_features.norm(dim = -1, keepdim = True))

            reference_names.extend(batch_reference_names)
            target_names.extend(batch_target_names)

        predicted_features = torch.cat(predicted_features, dim = 0)
        predicted_features = F.normalize(predicted_features.float())
        return predicted_features, target_names 

@torch.no_grad()
def compute_val_metrics(relative_val_dataset: QuickDrawDataset, model, index_features: torch.Tensor, index_names: List[str], device) -> Dict[str, float]:
        """
        Compute the retrieval metrics on the FashionIQ validation set given the dataset, pseudo tokens and the reference names
        """
        # Generate the predicted features
        predicted_features, target_names = generate_val_predictions(relative_val_dataset, model, device)

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
            distances = 1 - predicted_features@index_features.T
            sorted_indices = np.argsort(distances.detach().cpu().numpy(), axis=1)
        sorted_index_names = np.array(index_names)[sorted_indices]
        
        # Remove the reference images
        for i in range(len(sorted_index_names)):
            try: 
                sorted_index_names[i].remove(index_names[i])
            except:
                pass 
        # Check if the target names are in the top 10 and top 50

        labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
        
        # Compute median rank metric in labels
        # median_rank = []
        # for i in range(len(labels)):
        #     rank_indices = np.where(labels[i] == 1)[0]
        #     #print(rank_indices[0])
        #     if len(rank_indices) > 0:
        #         median_rank.append(rank_indices[0])
        # median_rank = np.mean(median_rank) if median_rank else float('nan')
        # print(f"Median rank: {median_rank}")

        # map_gtn = []
        # for i in range(len(labels)):
        #     number_of_correct_predictions = torch.sum(labels[i])
        #     labels_k = labels[i][:number_of_correct_predictions]
        #     # Calculate mean average precision of labels_k
        #     map = compute_map(labels_k.int().cpu().numpy())[0]
        #     map_gtn.append(map)
        # map_gtn = np.mean(map_gtn) if map_gtn else float('nan')
        # print(f"Mean Average Precision per Ground Truth Number: {map_gtn}")

        answers = dict()
        for i in range(len(target_names)):
            key = target_names[i]
            value = list(sorted_index_names[i][:50])
            answers[key] = value    

        # Compute the metrics
        metrics = dict()
        for k in [100, 200]:
            metrics[f"prec_at{k}"] = precision_at_k(labels, k)
        metrics['mAP'] = compute_map(labels[:,:].int().cpu().numpy())[0]
        return metrics

@torch.no_grad()
def val_retrieval(domain_type: str, model, preprocess: callable, device) -> Dict[str, float]:          
    """
    Compute the retrieval metrics on the Sketchy validation set given the pseudo tokens and the reference names
    """

    # Extract the index features
    classic_val_dataset = QuickDrawDataset('val', domain_type, 'classic', preprocess)
    index_features, index_names, captions = extract_index_features(classic_val_dataset, model, return_local=False)
    
    # Define the relative dataset
    relative_val_dataset = QuickDrawDataset('val', domain_type, 'relative', preprocess)
    return compute_val_metrics(relative_val_dataset, model, index_features, index_names, device)

def main_quickdraw(cfg):
    model = TransAgg(cfg)
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
    all_domains = open("/home/hle/SBIR/QuickDraw/zeroshot/cname_cid_zero.txt").readlines() 
    all_domains = [" ".join(text.split()[:-1]) for text in all_domains]#[:]
    results = list() 

    if cfg.val_dataset.lower() == 'quickdraw':
        precs_at50 = []
        precs_at100 = []
        mAP = []
        for domain_type in all_domains[:1]:
            metrics = val_retrieval(domain_type, model, preprocess, device)
            results.append([
                "Score",
                metrics['prec_at100'], 
                metrics['prec_at200'], 
                metrics['mAP']])
            precs_at50.append(metrics['prec_at100'])
            precs_at100.append(metrics['prec_at200'])
            mAP.append(metrics['mAP'])

    print(tabulate.tabulate(results, headers=['Metric', 'P@100', 'P@200', 'mAP@all']))
