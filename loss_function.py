import torch
import torch.nn.functional as F

# Extended InfoNCE loss function with memory-safe auxiliary negatives

def extended_infonce_loss_chunked(query_embed, target_embed, target_aux_embed=None, temperature=0.07, chunk_size=128):
    """
    Memory-safe contrastive loss with optional auxiliary negatives.

    query_embed: [B, D] (fused)
    target_embed: [B, D] (positive targets)
    target_aux_embed: [N, D] (auxiliary negatives), optional
    """
    B, D = query_embed.shape
    device = query_embed.device

    # Normalize
    q = F.normalize(query_embed, dim=-1)  # [B, D]
    t = F.normalize(target_embed, dim=-1)  # [B, D]

    # Positive sims
    pos_sim = torch.sum(q * t, dim=-1, keepdim=True)  # [B, 1]

    # In-batch negatives (excluding diagonal)
    full_sim = torch.matmul(q, t.T) / temperature  # [B, B]
    mask = ~torch.eye(B, dtype=torch.bool, device=device)
    batch_neg = full_sim[mask].view(B, B - 1)  # [B, B-1]

    # Auxiliary negatives in chunks to avoid OOM
    aux_neg = []
    if target_aux_embed is not None:
        target_aux_embed = F.normalize(target_aux_embed, dim=-1)
        N = target_aux_embed.size(0)
        for i in range(0, N, chunk_size):
            chunk = target_aux_embed[i:i+chunk_size]  # [chunk, D]
            sim_chunk = torch.matmul(q, chunk.T) / temperature  # [B, chunk]
            aux_neg.append(sim_chunk)
        aux_neg = torch.cat(aux_neg, dim=1)  # [B, total_aux]

    # Combine all negatives
    if target_aux_embed is not None:
        all_neg = torch.cat([batch_neg, aux_neg], dim=1)  # [B, total_neg]
    else:
        all_neg = batch_neg  # [B, B-1]

    logits = torch.cat([pos_sim / temperature, all_neg], dim=1)  # [B, 1 + neg]
    labels = torch.zeros(B, dtype=torch.long, device=device)

    loss = F.cross_entropy(logits, labels)
    return loss

def extended_infonce_loss(query_embed, target_embed, target_aux_embed=None, temperature=0.07):
    """
    query_embed: [B, D]  — fused (reference, caption) query
    target_embed: [B, D] — ground-truth target
    target_aux_embed: [B, D] — auxiliary distractor targets (optional)
    """

    B = query_embed.size(0)

    # Normalize features
    q = F.normalize(query_embed, dim=-1)
    t = F.normalize(target_embed, dim=-1)

    # Positive logits: sim(q_i, t_i)
    pos_sim = torch.sum(q * t, dim=-1, keepdim=True)  # [B, 1]

    # Batch negatives: sim(q_i, t_j) for j != i
    all_sim = torch.matmul(q, t.T)  # [B, B]
    mask = ~torch.eye(B, dtype=torch.bool, device=q.device)
    neg_sim = all_sim[mask].view(B, B - 1)  # [B, B-1]

    # Optional: Auxiliary negatives
    if target_aux_embed is not None:
        t_aux = F.normalize(target_aux_embed, dim=-1)
        aux_sim = torch.sum(q * t_aux, dim=-1, keepdim=True)  # [B, 1]
        logits = torch.cat([pos_sim, neg_sim, aux_sim], dim=1)  # [B, B+1]
    else:
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, B]

    # Apply temperature
    logits /= temperature

    # Ground truth always at index 0
    labels = torch.zeros(B, dtype=torch.long, device=q.device)

    # Use standard cross-entropy loss now
    loss = F.cross_entropy(logits, labels)
    return loss

def infonce_loss(query_embed, target_embed, temperature=0.07, eps=1e-6):
    """
    Improved InfoNCE loss function with numerical stability and optional temperature scaling.

    query_embed: [B, D]  — fused (reference, caption) query
    target_embed: [B, D] — ground-truth target
    temperature: float — scaling factor for logits
    eps: float — small value to avoid division by zero
    """
    B = query_embed.size(0)

    # Normalize features with added numerical stability
    q = F.normalize(query_embed, dim=-1, eps=eps)
    t = F.normalize(target_embed, dim=-1, eps=eps)

    # Positive logits: sim(q_i, t_i)
    pos_sim = torch.sum(q * t, dim=-1, keepdim=True)  # [B, 1]

    # Batch negatives: sim(q_i, t_j) for j != i
    all_sim = torch.matmul(q, t.T)  # [B, B]
    mask = ~torch.eye(B, dtype=torch.bool, device=q.device)
    neg_sim = all_sim[mask].view(B, B - 1)  # [B, B-1]

    # Combine positive and negative logits
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, B]
    
    # Apply temperature scaling
    logits /= temperature

    # Ground truth always at index 0
    labels = torch.zeros(B, dtype=torch.long, device=q.device)

    # Use standard cross-entropy loss now
    loss = F.cross_entropy(logits, labels)

    # Optional: Add regularization term to encourage feature diversity
    diversity_loss = -torch.mean(torch.sum(q * q, dim=-1))  # Penalize high self-similarity
    loss += 0.01 * diversity_loss  # Weight for diversity regularization

    return loss

# Sigmoid loss function for binary classification
def sigmoid_loss(query_embed, target_embed):
    """
    Sigmoid loss function for binary classification.

    query_embed: [B, D]  — fused (reference, caption) query
    target_embed: [B, D] — ground-truth target
    """
    # Normalize features
    q = F.normalize(query_embed, dim=-1)
    t = F.normalize(target_embed, dim=-1)

    # Compute logits
    logits = torch.sum(q * t, dim=-1)  # [B]

    # Apply sigmoid and compute binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    return loss

def sigmoid_loss_with_negatives(query_embed, target_embed, negative_embed):
    """
    Sigmoid loss function with negatives for binary classification.
    query_embed: [B, D]  — fused (reference, caption)
    target_embed: [B, D] — ground-truth target
    negative_embed: [B, D] — negative samples
    """
    # Normalize features
    q = F.normalize(query_embed, dim=-1)
    t = F.normalize(target_embed, dim=-1)
    n = F.normalize(negative_embed, dim=-1)

    # Compute logits
    pos_logits = torch.sum(q * t, dim=-1)  # [B]
    neg_logits = torch.sum(q * n, dim=-1)  # [B]

    # Apply sigmoid and compute binary cross-entropy loss
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))

    return pos_loss + neg_loss 

def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Computes triplet loss.

    Args:
        anchor:   [B, D] tensor, e.g. fused query (image + caption)
        positive: [B, D] tensor, e.g. matching target
        negative: [B, D] tensor, e.g. non-matching target
        margin:   float, margin hyperparameter (default: 0.2)
                                                        
    Returns: Scalar triplet loss
    """
    # L2-normalize for cosine distance if needed
    anchor   = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)

    # Compute pairwise distances
    pos_dist = F.pairwise_distance(anchor, positive, p=2)  # [B]
    neg_dist = F.pairwise_distance(anchor, negative, p=2)  # [B]

    # Compute loss
    losses = F.relu(pos_dist - neg_dist + margin)
    return losses.mean()

