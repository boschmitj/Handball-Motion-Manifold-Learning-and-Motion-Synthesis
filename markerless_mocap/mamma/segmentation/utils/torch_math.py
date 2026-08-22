import torch.nn.functional as F
import torch


def cosine_knn(query_feats, gallery_feats, topk=1):
    """
    query_feats: Tensor [N_query, D]  – features to match (e.g., new detections)
    gallery_feats: Tensor [N_gallery, D] – stored embeddings (e.g., person of interest in multiple views)
    topk: number of matches to return
    Returns indices and similarity scores of topk matches
    """
    # Ensure numerical compatibility under mixed precision / autocast.
    if query_feats.dtype != gallery_feats.dtype:
        query_feats = query_feats.to(dtype=torch.float32)
        gallery_feats = gallery_feats.to(dtype=torch.float32)
    if query_feats.device != gallery_feats.device:
        gallery_feats = gallery_feats.to(query_feats.device)

    # Normalize both sets
    query_feats = F.normalize(query_feats, dim=1)
    gallery_feats = F.normalize(gallery_feats, dim=1)

    # Compute cosine similarity matrix [N_query, N_gallery]
    sim = query_feats @ gallery_feats.T

    # Get topk matches
    sim_scores, indices = sim.topk(k=topk, dim=1)
    return indices, sim_scores
