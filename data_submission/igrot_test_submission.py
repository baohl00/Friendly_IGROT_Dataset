import json
import multiprocessing
from typing import List, Tuple
import numpy as np
import torch, gc 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import TransAgg
from utils import \
    get_preprocess, extract_index_features, \
    sim_matrix, sim_matrix_mm, softmax, decompose_matrices
from data.igrot_dataset import IGROTDataset

import os
SEED = 42
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:512"


torch.cuda.empty_cache()
gc.collect()

from torch.nn.utils.rnn import pad_sequence

def calculate_map_improved(ground_truth, prediction, at=[1000, 2500, 5000]):
    """
    Calculate mean Average Precision (mAP) and other metrics between ground truth and prediction
    
    Args:
        ground_truth: List of relevant items
        prediction: List of predicted items (ranked)
        at: List of k values for P@k and R@k calculation
    
    Returns:
        dict: Dictionary containing AP, P@k, R@k metrics
    """
    metrics = {}
    
    # Convert ground truth to set for faster lookup
    gt_set = set(ground_truth)
    
    # Create relevance array: 1 if prediction[i] is in ground truth, 0 otherwise
    is_relevant = [1 if pred in gt_set else 0 for pred in prediction]
    
    # Calculate Average Precision (AP)
    precisions = []
    relevant_count = 0
    for k, rel in enumerate(is_relevant, start=1):
        if rel:
            relevant_count += 1
            precisions.append(relevant_count / k)
    
    ap = sum(precisions) / len(precisions) if precisions else 0
    metrics["AP"] = round(ap * 100, 2)
    
    # Calculate Precision@k and Recall@k
    total_relevant = len(gt_set)  # Total number of relevant items
    for k in at:
        if k <= len(is_relevant):
            relevant_at_k = sum(is_relevant[:k])
            precision_at_k = relevant_at_k / k if k else 0
            recall_at_k = relevant_at_k / total_relevant if total_relevant else 0
            
            metrics[f"P@{k}"] = round(precision_at_k * 100, 2)
            metrics[f"R@{k}"] = round(recall_at_k * 100, 2)
    
    return metrics

# Update your evaluation code to use the improved function
def evaluate_predictions_improved(ground_truths, predictions, at=[1, 5, 10, 15, 20]):
    """
    Evaluate all predictions and return aggregated metrics
    """
    all_metrics = []
    
    for gt, pred in zip(ground_truths, predictions):
        metrics = calculate_map_improved(gt, pred, at)
        all_metrics.append(metrics)
    
    # print(all_metrics)
    # Aggregate metrics
    aggregated = {}
    if not all_metrics:
        raise ValueError("No metrics were calculated. Ensure ground_truths and predictions are not empty.")
    for key in all_metrics[0].keys():
        aggregated[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return aggregated

# Calculate mAP by test class using the improved function
def calculate_map_by_class_improved(data, predictions, class_i, at=[5, 10, 20, 50, 100, 200]):
    """
    Calculate mean Average Precision (mAP) by test class using the improved function
    """
    # Filter data by class
    indices = [i for i, item in enumerate(data) if item['class'] == class_i]
    gt_list = [data[i]['target_images'] for i in indices]
    pred_list = [predictions[i] for i in indices if i < len(predictions)]

    # Calculate mAP for each sample using the improved function
    scores = evaluate_predictions_improved(gt_list, pred_list, at)
    return scores


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length data"""
    batch_pairs_id, batch_reference_names, captions, batch_group_members, reference_images = zip(*batch)
    
    # Handle captions if they're variable length
    if isinstance(captions[0], torch.Tensor):
        # Pad captions to same length
        captions = pad_sequence(captions, batch_first=True, padding_value=0)
    else:
        # If captions are strings, keep as list
        captions = list(captions)
    
    # Stack images if they're tensors of same size
    if isinstance(reference_images[0], torch.Tensor):
        try:
            reference_images = torch.stack(reference_images)
        except:
            # If images have different sizes, keep as list
            reference_images = list(reference_images)
    
    return (list(batch_pairs_id), list(batch_reference_names), captions, 
            list(batch_group_members), reference_images)


def generate_test_submissions(file_name, model, preprocess, device):
    type = 'test'
    classic_test_dataset = IGROTDataset(type, 'classic', preprocess)
    index_features, index_names, _ = extract_index_features(classic_test_dataset, model, return_local=False)
    relative_test_dataset = IGROTDataset(type, 'relative', preprocess)
    pairid_to_predictions = generate_test_dicts(relative_test_dataset, index_features, index_names, model, device)

   
    data =  json.load(open('/home/hle/IGROT/data/test.json', 'r')) 
    ground_truths = [i['target_images'] for i in data]
    predictions = list(pairid_to_predictions.values())
    # print(predictions[:5])
    classes = set([i['class'] for i in data])
    sum = 0
    for class_i in classes:
        print(f"Class {class_i}")
        scores = calculate_map_by_class_improved(data, predictions, class_i)
        for metric, value in scores.items():
            if metric == 'AP':
                print(f"--- {metric}: {value}")
                sum += value
            if class_i == 'sbir':
                if metric in ['P@5', 'P@10', 'P@20', 'P@50', 'P@100']:
                    print(f"--- {metric}: {value}")
            else: 
                if metric in ['R@5', 'R@10', 'R@20', 'R@50']:
                    print(f"--- {metric}: {value}")
    print(f"Average AP: {sum / len(classes)}")
    # Save all the printed scores into a file
    with open(f'./submission/figrotd/{file_name}.txt', 'w') as f:
        f.write(f"Average AP: {sum / len(classes)}\n")
        for class_i in classes:
            f.write(f"Class {class_i}\n")
            scores = calculate_map_by_class_improved(data, predictions, class_i)
            for metric, value in scores.items():
                f.write(f"--- {metric}: {value}\n")


def generate_test_dicts(relative_test_dataset, index_features, index_names, model, device):
    # Generate predictions
    predicted_features, reference_names, group_members, pairs_id = generate_patterncom_test_predictions(relative_test_dataset, model, device)
    print(f"Compute FIGROTD prediction dicts")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()
    
    ranking_type = 'normal' # normal, dual, decom
    # Compute the distances and sort the results
    if ranking_type == 'dual':
        similarity = sim_matrix_mm(predicted_features, index_features)
        similarity = softmax(similarity/0.1, dim = 1) * similarity
        similarity = softmax(similarity, dim = 0)
        distances = 1 - similarity #predicted_features@index_features.T
        sorted_indices = np.argsort(distances, axis=1)
    elif ranking_type == 'decom':
        predicted_features_decom, index_features_decom = decompose_matrices(predicted_features, index_features, 32)
        distances = 1 - predicted_features_decom @ index_features_decom.T
        sorted_indices = torch.argsort(distances, dim=-1).cpu()
    else:
        distances = 1 - predicted_features @ index_features.T
        sorted_indices = torch.argsort(distances, dim=-1).cpu()
               
    #sorted_indices = torch.topk(similarity, dim=-1, k=100).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    #sorted_index_names = np.array(index_names)[sorted_indices]
    print(sorted_index_names.shape)
    # Generate prediction dictsi
    def remove_duplicate(data):
        seen = set()
        seen_add = seen.add
        return [x for x in data if not (x in seen or seen_add(x))]

    pairid_to_predictions = {}
    for i in range(len(pairs_id)):
        sorted_results = sorted_index_names[i].tolist()
        #print(len(sorted_results))
        topk = remove_duplicate(sorted_results)[:100]
        # try: 
        #     topk.remove(int(reference_names[i][:-4]))
        # except:
        #     pass
        pairid_to_predictions[str(i)] = topk

    # Save predictions to a file
    with open(f'./submission/figrotd/answer.json', 'w') as f:
        json.dump(pairid_to_predictions, f)

    return pairid_to_predictions

def generate_patterncom_test_predictions(relative_test_dataset: IGROTDataset, model, device) -> Tuple[torch.tensor, List[str], List[List[str]], List[str]]:
    print(f"Compute PatternCom test predictions")
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=64,
                                      num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    predicted_features = []
    group_members = []
    reference_names = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members, reference_images in tqdm(
            relative_test_loader):  # Load data
        batch_group_members = list(map(list, zip(*batch_group_members)))

        # Compute the predicted features
        with torch.no_grad():
            reference_images = reference_images.to(device)
            batch_predicted_features = model.final_features(reference_images, captions)
                    # + model.union_features(reference_images, captions) * 0
            if type(batch_predicted_features) == tuple:
                batch_predicted_features = batch_predicted_features[0]
            predicted_features.append(batch_predicted_features / batch_predicted_features.norm(dim=-1, keepdim=True))

        torch.cuda.empty_cache()
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)

    predicted_features = torch.cat(predicted_features, dim=0)

    return predicted_features, reference_names, group_members, pairs_id


def main_igrot(cfg):
    model = TransAgg(cfg)
    device = cfg.device
    model = model.to(device)
    if not os.path.exists(cfg.val_load_path):
        raise FileNotFoundError(f"Model file not found at {cfg.val_load_path}")
    try:
        model.load_state_dict(torch.load(cfg.val_load_path))
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load model state dict. Ensure the file is a valid PyTorch model file. Error: {e}")

    if cfg.model.startswith("blip"):
        input_dim = 384
    elif cfg.model.startswith("clip"):
        input_dim = model.model.visual.input_resolution
    #input_dim = model.model.visual.input_resolution
    preprocess = get_preprocess(preprocess = cfg.preprocess, input_dim=input_dim)

    model.eval()

    generate_test_submissions(cfg.submission_name, model, preprocess, device=device)
