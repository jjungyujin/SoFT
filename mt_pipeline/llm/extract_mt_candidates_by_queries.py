import argparse
import json
import os
from os import sys

from PIL import Image
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import clip

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
from generate_query import generate_query
from utils import FIQ_DATASET_PATH, CIRR_DATASET_PATH


def remove_duplicate(triplets):
    seen = set()
    new_triplets = []
    for triplet in triplets:
        if triplet["candidate"] not in seen:
            seen.add(triplet["candidate"])
            new_triplets.append(triplet)
    return new_triplets


def image_loader(
    dataset, dataset_path, split, dress_type, no_duplicates=False, sampling_number=None
):
    dataset_path = Path(dataset_path)
    triplets = []

    ref_image_names = []
    captions = []
    target_iamge = []

    all_val_image_names = []
    all_val_images = []

    if dataset == "fiq":
        with open(dataset_path / "captions" / f"cap.{dress_type}.{split}.json") as f:
            triplets.extend(json.load(f))

        if no_duplicates:
            triplets = remove_duplicate(triplets)

        for triplet in triplets:
            ref_image_names.append(triplet["candidate"])
            captions.append(triplet["captions"])
            target_iamge.append(triplet["target"])

        with open(
            dataset_path / "image_splits" / f"split.{dress_type}.{split}.json"
        ) as f:
            all_val_image_names.extend(json.load(f))

        for val_img_name in all_val_image_names:
            val_image_path = dataset_path / "images" / f"{val_img_name}.png"
            val_image = Image.open(val_image_path).convert("RGB")

            all_val_images.append(val_image)

    elif dataset == "cirr":
        with open(dataset_path / "cirr" / "captions" / f"cap.rc2.{split}.json") as f:
            triplets = json.load(f)

        if no_duplicates:
            triplets = remove_duplicate(triplets)

        for triplet in triplets:
            ref_image_names.append(triplet["reference"])
            captions.append([triplet["caption"]])
            target_iamge.append(triplet["target_hard"])

        with open(
            dataset_path / "cirr" / "image_splits" / f"split.rc2.{split}.json"
        ) as f:
            all_val_image_names_dict = json.load(f)

        for key, value in all_val_image_names_dict.items():
            val_image_path = dataset_path / value
            val_image = Image.open(val_image_path).convert("RGB")

            all_val_image_names.append(key)
            all_val_images.append(val_image)

    return ref_image_names, captions, target_iamge, all_val_image_names, all_val_images


def calculate_clip_similarity(
    model, preprocess, images, query_texts, device, image_feature=None
):
    text_tokens = clip.tokenize([query_texts]).to(device)

    if image_feature is None:
        image_inputs = torch.stack([preprocess(img) for img in images]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_inputs)
            text_features = model.encode_text(text_tokens)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

    else:
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, p=2, dim=1)
        image_features = torch.load(image_feature)
    similarities = image_features @ text_features.T
    similarities = similarities.squeeze(1)

    return similarities


def process_dataset_and_save_results(args, dataset, dataset_path, dress_type=None):
    results_1 = []
    results_2 = []
    costs_result = []

    top_k_rank = 20
    device = "cuda"
    model, preprocess = clip.load("ViT-L/14", device=device)

    output_prefix = (
        f"mt_candidates_{dress_type}" if dataset == "fiq" else "mt_candidates_cirr"
    )
    image_features_file = (
        f"{dress_type}_val_image_features_all.pt"
        if dataset == "fiq"
        else "cirr_val_image_features_all.pt"
    )

    image_names, captions, target_iamge_name, all_val_image_names, all_val_images = (
        image_loader(
            dataset, dataset_path, "val", dress_type=dress_type, no_duplicates=False
        )
    )

    query_lists_by_llm, costs_list = generate_query(
        dataset, dataset_path, image_names, captions
    )

    for idx, query_list in enumerate(query_lists_by_llm):
        caption_query = query_list[0]
        joined_query = " ".join(query_list).strip()

        similarities_1 = calculate_clip_similarity(
            model,
            preprocess,
            all_val_images,
            caption_query,
            device,
            image_features_file,
        )

        sorted_distances, sorted_indices = torch.sort(similarities_1, descending=True)
        sorted_distances = sorted_distances.cpu()
        sorted_indices = sorted_indices.cpu().numpy()

        top_k_names = np.array(all_val_image_names)[sorted_indices[:top_k_rank]]
        top_k_scores = np.array(similarities_1.cpu())[sorted_indices[:top_k_rank]]

        results_1.append(
            {
                "ref_image_name": image_names[idx],
                "relative_captions": captions[idx],
                "target_image_name": target_iamge_name[idx],
                "query": caption_query,
                "top_k_names": top_k_names.tolist(),
                "top_k_scores": top_k_scores.tolist(),
            }
        )

        if joined_query != caption_query:
            similarities_2 = calculate_clip_similarity(
                model,
                preprocess,
                all_val_images,
                joined_query,
                device,
                image_features_file,
            )

            sorted_distances, sorted_indices = torch.sort(
                similarities_2, descending=True
            )
            sorted_distances = sorted_distances.cpu()
            sorted_indices = sorted_indices.cpu().numpy()

            top_k_names = np.array(all_val_image_names)[sorted_indices[:top_k_rank]]
            top_k_scores = np.array(similarities_2.cpu())[sorted_indices[:top_k_rank]]

            results_2.append(
                {
                    "ref_image_name": image_names[idx],
                    "relative_captions": captions[idx],
                    "target_image_name": target_iamge_name[idx],
                    "query": joined_query,
                    "top_k_names": top_k_names.tolist(),
                    "top_k_scores": top_k_scores.tolist(),
                }
            )
        else:
            results_2.append(
                {
                    "ref_image_name": image_names[idx],
                    "relative_captions": captions[idx],
                    "target_image_name": target_iamge_name[idx],
                    "query": "",
                    "top_k_names": [],
                    "top_k_scores": [],
                }
            )

        costs_result.append({"cost": costs_list[idx]})

    os.makedirs(f"{BASE_DIR}/multi_target_candidate", exist_ok=True)

    with open(
        f"{BASE_DIR}/multi_target_candidate/{output_prefix}_by_caption_query_20.json",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        json.dump(results_1, f, ensure_ascii=False, indent=2)

    with open(
        f"{BASE_DIR}/multi_target_candidate/{output_prefix}_by_joined_query_20.json",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        json.dump(results_2, f, ensure_ascii=False, indent=2)

    with open(
        f"{BASE_DIR}/multi_target_candidate/{output_prefix}_cost_result_20.json",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        json.dump(costs_result, f, ensure_ascii=False, indent=2)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, default="fiq", choices=["fiq", "cirr"]
    )

    args = parser.parse_args()

    if args.dataset == "fiq":
        dress_types = ["dress", "shirt", "toptee"]

        for dress_type in dress_types:
            process_dataset_and_save_results(
                args, args.dataset, FIQ_DATASET_PATH, dress_type
            )

    elif args.dataset == "cirr":
        process_dataset_and_save_results(args, args.dataset, CIRR_DATASET_PATH)
