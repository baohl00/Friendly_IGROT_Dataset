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
from data.circo_dataset import CIRCODataset

import os
SEED = 42
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:512"


torch.cuda.empty_cache()
gc.collect()

def generate_circo_test_submissions(file_name, model, preprocess, device):
    type = 'test'
    classic_test_dataset = CIRCODataset(type, 'classic', preprocess)
    index_features, index_names, _ = extract_index_features(classic_test_dataset, model, return_local=False)
    relative_test_dataset = CIRCODataset(type, 'relative', preprocess)
    pairid_to_predictions = generate_circo_test_dicts(relative_test_dataset, index_features, index_names, model, device)

    print(f"Saving CIRCO test predictions")
    if type == 'val':
        name = 'val_' + file_name
    else:
        name = file_name
    output_file = f"./submission/circo_igrot/{name}.json"
    with open(output_file, 'w') as file:
        json.dump(pairid_to_predictions, file, indent=4)

    print('Results are written to file ', output_file)

def generate_circo_test_dicts(relative_test_dataset, index_features, index_names, model, device):
    # Generate predictions
    predicted_features, reference_names, group_members, pairs_id = generate_circo_test_predictions(relative_test_dataset, model, device)

    print(f"Compute CIRCO prediction dicts")

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
        predicted_features_decom, index_features_decom = decompose_matrices(predicted_features, index_features, 128)
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
        try: 
            topk.remove(int(reference_names[i][:-4]))
        except:
            pass
        pairid_to_predictions[str(i)] = topk[:50]
    
    return pairid_to_predictions

def generate_circo_test_predictions(relative_test_dataset: CIRCODataset, model, device) -> Tuple[torch.tensor, List[str], List[List[str]], List[str]]:
    print(f"Compute CIRCO test predictions")
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=128,
                                      num_workers=multiprocessing.cpu_count(), pin_memory=True)

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    predicted_features = []
    group_members = []
    reference_names = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members, reference_images in tqdm(
            relative_test_loader):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()

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


def main_circo(cfg):
    model = TransAgg(cfg)
    device = cfg.device
    model = model.to(device)
    model.load_state_dict(torch.load(cfg.val_load_path))

    if cfg.model.startswith("blip"):
        input_dim = 384
    elif cfg.model.startswith("clip"):
        input_dim = model.model.visual.input_resolution
    #input_dim = model.model.visual.input_resolution
    preprocess = get_preprocess(preprocess = cfg.preprocess, input_dim=input_dim)

    model.eval()

    generate_circo_test_submissions(cfg.submission_name, model, preprocess, device=device)

