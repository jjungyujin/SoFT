from unittest import result
import torch
import os
from typing import Dict, List, Literal
import numpy as np
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent


def save_cache(index_names, similarity, target_names, result_path, **kwargs):
    os.makedirs(result_path, exist_ok=True)
    np.save(
        os.path.join(
            result_path,
            f"{kwargs['preload_str']}_index_names.npy",
        ),
        np.array(index_names, dtype=object),
    )
    np.save(
        os.path.join(
            result_path,
            f"{kwargs['preload_str']}_base.npy",
        ),
        similarity.cpu().numpy(),
    )
    np.save(
        os.path.join(
            result_path,
            f"{kwargs['preload_str']}_penalty.npy",
        ),
        kwargs["penalty"].cpu().numpy(),
    )
    np.save(
        os.path.join(
            result_path,
            f"{kwargs['preload_str']}_reward.npy",
        ),
        kwargs["reward"].cpu().numpy(),
    )
    np.save(
        os.path.join(
            result_path,
            f"{kwargs['preload_str']}_target_names.npy",
        ),
        np.array(target_names, dtype=object),
    )


def save_result(
    sorted_distances, sorted_index_names, lambda_val, rerank_type, result_path, **kwargs
):
    np.savetxt(
        os.path.join(
            result_path,
            f"{rerank_type}_{int(lambda_val*10)}_{kwargs['preload_str']}_sorted_distances.csv",
        ),
        sorted_distances,
        delimiter=",",
    )
    np.save(
        os.path.join(
            result_path,
            f"{rerank_type}_{int(lambda_val*10)}_{kwargs['preload_str']}_sorted_index_names.npy",
        ),
        sorted_index_names,
    )


def compute_map_at_k(labels: np.ndarray, k: int) -> float:
    num_queries = labels.shape[0]
    average_precisions = []
    for i in range(num_queries):
        relevant = labels[i][:k]
        if not np.any(relevant):
            average_precisions.append(0.0)
            continue
        precision_at_i = [
            np.sum(relevant[: j + 1]) / (j + 1) for j in range(k) if relevant[j]
        ]
        ap = np.mean(precision_at_i) if precision_at_i else 0.0
        average_precisions.append(ap)
    return round(np.mean(average_precisions) * 100, 2)


def get_fiq_metrics(score, index_names, target_names, weight):
    sorted_distances, sorted_indices = torch.sort(score, descending=True)
    sorted_distances = sorted_distances.cpu()
    sorted_indices = sorted_indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names
        == np.repeat(np.array(target_names), len(index_names)).reshape(
            len(target_names), -1
        )
    )
    assert torch.equal(
        torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int()
    )

    output_metrics = {
        f"Recall@1": (torch.sum(labels[:, :1]) / len(labels)).item() * 100,
        f"Recall@5": (torch.sum(labels[:, :5]) / len(labels)).item() * 100,
        f"Recall@10": (torch.sum(labels[:, :10]) / len(labels)).item() * 100,
        f"Recall@50": (torch.sum(labels[:, :50]) / len(labels)).item() * 100,
    }
    return output_metrics, sorted_distances, sorted_index_names


def get_mt_fiq_metrics(score, index_names, target_names) -> float:
    sorted_distances, sorted_indices = torch.sort(score, descending=True)
    sorted_distances = sorted_distances.cpu()
    sorted_indices = sorted_indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = np.zeros_like(sorted_index_names, dtype=bool)
    for i, targets in enumerate(target_names):
        labels[i] = np.isin(sorted_index_names[i], targets)

    output_metrics = {
        "mAP@5": compute_map_at_k(labels, k=5),
        "mAP@10": compute_map_at_k(labels, k=10),
        "mAP@25": compute_map_at_k(labels, k=25),
        "mAP@50": compute_map_at_k(labels, k=50),
    }
    return output_metrics, sorted_distances, sorted_index_names


@torch.no_grad()
def fiq(
    device: torch.device,
    predicted_features: torch.Tensor,
    target_names: List,
    index_features: torch.Tensor,
    index_names: List,
    is_save_cache: bool = False,
    is_save_result: bool = True,
    lambda_val: float = 1.0,
    baseline: Literal["cirevl", "searle"] = "cirevl",
    rerank_type: Literal["soft", "reward", "penalty"] = "soft",
    split: str = "val",
    **kwargs,
) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the Fashion-IQ validation set fiven the dataset, pseudo tokens and reference names.
    Computes Recall@10 and Recall@50.
    """
    index_features = torch.nn.functional.normalize(index_features).to(device)
    predicted_features = torch.nn.functional.normalize(predicted_features).to(device)

    similarity = predicted_features @ index_features.T
    if is_save_cache:
        save_cache(
            index_names,
            similarity,
            target_names,
            result_path=root_path / "results" / baseline,
            **kwargs,
        )

    if rerank_type == "soft":
        soft_score = similarity * (kwargs["reward"] + 1 - kwargs["penalty"]) / 2
        final_score = (1 - lambda_val) * similarity + lambda_val * soft_score

    else:
        print(f"Reranking with {rerank_type} scores.")
        final_score = (1 - lambda_val) * similarity + lambda_val * (
            similarity * kwargs[rerank_type]
        )

    output_metrics, sorted_distances, sorted_index_names = get_fiq_metrics(
        final_score, index_names, target_names, lambda_val
    )

    if is_save_result:
        save_result(
            sorted_distances,
            sorted_index_names,
            lambda_val,
            rerank_type,
            result_path=root_path / "results" / baseline,
            **kwargs,
        )

    return output_metrics


@torch.no_grad()
def mt_fiq(
    device: torch.device,
    predicted_features: torch.Tensor,
    target_names: List,
    index_features: torch.Tensor,
    index_names: List,
    is_save_cache: bool = False,
    is_save_result: bool = True,
    lambda_val: float = 1.0,
    baseline: Literal["cirevl", "searle"] = "cirevl",
    rerank_type: Literal["soft", "reward", "penalty"] = "soft",
    **kwargs,
):
    index_features = torch.nn.functional.normalize(index_features).to(device)
    predicted_features = torch.nn.functional.normalize(predicted_features).to(device)

    similarity = predicted_features @ index_features.T
    if is_save_cache:
        save_cache(
            index_names,
            similarity,
            target_names,
            result_path=root_path / "results" / baseline,
            **kwargs,
        )

    if rerank_type == "soft":
        soft_score = similarity * (kwargs["reward"] + 1 - kwargs["penalty"]) / 2
        final_score = (1 - lambda_val) * similarity + lambda_val * soft_score

    else:
        print(f"Reranking with {rerank_type} scores.")
        final_score = (1 - lambda_val) * similarity + lambda_val * (
            similarity * kwargs[rerank_type]
        )

    if kwargs["target_type"] == "single":
        target_names = sum(target_names, [])
        output_metrics, sorted_distances, sorted_index_names = get_fiq_metrics(
            final_score, index_names, target_names, lambda_val
        )
    else:
        output_metrics, sorted_distances, sorted_index_names = get_mt_fiq_metrics(
            final_score, index_names, np.array(target_names)
        )

    if is_save_result:
        save_result(
            sorted_distances,
            sorted_index_names,
            lambda_val,
            rerank_type,
            result_path=root_path / "results" / baseline,
            **kwargs,
        )
    return output_metrics
