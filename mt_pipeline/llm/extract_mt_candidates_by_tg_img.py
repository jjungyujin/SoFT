import argparse
import json
import os
from os import sys
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import clip

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
from utils import CIRR_DATASET_PATH, FIQ_DATASET_PATH


# %%
def remove_duplicate(triplets):
    seen = set()
    new_triplets = []
    for triplet in triplets:
        if triplet["candidate"] not in seen:
            seen.add(triplet["candidate"])
            new_triplets.append(triplet)
    return new_triplets


def image_loader(
    dataset,
    dataset_path,
    split,
    dress_type=None,
    no_duplicates=False,
    sampling_number=None,
):
    dataset_path = Path(dataset_path)

    triplets = []
    image_names = []
    captions = []

    target_image_name = []
    target_images = []

    all_val_image_names = []
    all_val_images = []

    if dataset == "fiq":
        with open(dataset_path / "captions" / f"cap.{dress_type}.{split}.json") as f:
            triplets.extend(json.load(f))

        if no_duplicates:
            triplets = remove_duplicate(triplets)

        for triplet in triplets:
            image_names.append(triplet["candidate"])
            captions.append(triplet["captions"])

            target_img_name = triplet["target"]
            target_img_path = dataset_path / "images" / f"{target_img_name}.png"
            target_img = Image.open(target_img_path).convert("RGB")

            target_image_name.append(target_img_name)
            target_images.append(target_img)

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
            image_names.append(triplet["reference"])
            captions.append(triplet["caption"])

            target_img_name = triplet["target_hard"]
            target_img_path = dataset_path / "dev" / f"{target_img_name}.png"
            target_img = Image.open(target_img_path).convert("RGB")

            target_image_name.append(target_img_name)
            target_images.append(target_img)

        with open(
            dataset_path / "cirr" / "image_splits" / f"split.rc2.{split}.json"
        ) as f:
            all_val_image_names_dict = json.load(f)

        for key, value in all_val_image_names_dict.items():
            val_image_path = dataset_path / value
            val_image = Image.open(val_image_path).convert("RGB")

            all_val_image_names.append(key)
            all_val_images.append(val_image)

    return (
        image_names,
        captions,
        target_image_name,
        target_images,
        all_val_image_names,
        all_val_images,
    )


def calculate_clip_image_similarity(
    model, preprocess, query_img, images, device, image_feature=None
):
    query_img_input = preprocess(query_img).unsqueeze(0).to(device)
    if image_feature is None:
        image_inputs = torch.stack([preprocess(img) for img in images]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_inputs)
            query_img_features = model.encode_image(query_img_input)

        image_features = F.normalize(image_features, p=2, dim=1)
        query_img_features = F.normalize(query_img_features, p=2, dim=1)

    else:
        with torch.no_grad():
            query_img_features = model.encode_image(query_img_input)
        query_img_features = F.normalize(query_img_features, p=2, dim=1)
        image_features = torch.load(image_feature)
    similarities = query_img_features @ image_features.T
    similarities = similarities.squeeze(0)

    return similarities


def process_dataset_and_save_results(dataset, dataset_path, dress_type=None):
    device = "cuda"
    model, preprocess = clip.load("ViT-L/14", device=device)
    top_k_rank = 20

    output_prefix = (
        f"mt_candidates_{dress_type}_by_tg_img"
        if dataset == "fiq"
        else "similarity_val_target_image_cirr"
    )
    image_features_file = (
        f"{dress_type}_val_image_features_all.pt"
        if dataset == "fiq"
        else "cirr_val_image_features_all.pt"
    )

    (
        image_names,
        captions,
        target_image_name,
        target_images,
        all_val_image_names,
        all_val_images,
    ) = image_loader(
        dataset, dataset_path, "val", dress_type=dress_type, no_duplicates=False
    )

    results = []

    for idx, target_img in tqdm(enumerate(target_images)):
        similarities = calculate_clip_image_similarity(
            model, preprocess, target_img, all_val_images, device, image_features_file
        )

        sorted_distances, sorted_indices = torch.sort(similarities, descending=True)
        sorted_distances = sorted_distances.cpu().numpy()
        sorted_indices = sorted_indices.cpu().numpy()

        top_k_names = np.array(all_val_image_names)[sorted_indices[:top_k_rank]]
        top_k_scores = sorted_distances[:top_k_rank]

        results.append(
            {
                "ref_image_name": image_names[idx],
                "relative_captions": captions[idx],
                "target_image_name": target_image_name[idx],
                "top_k_names": top_k_names.tolist(),
                "top_k_scores": top_k_scores.tolist(),
            }
        )

    os.makedirs(f"{BASE_DIR}/multi_target_candidate", exist_ok=True)
    with open(
        f"{BASE_DIR}/multi_target_candidate/{output_prefix}_20.json",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, default="fiq", choices=["fiq", "cirr"]
    )

    args = parser.parse_args()

    if args.dataset == "fiq":
        dress_types = ["shirt", "toptee", "dress"]

        for dress_type in dress_types:
            process_dataset_and_save_results(args.dataset, FIQ_DATASET_PATH, dress_type)

    elif args.dataset == "cirr":
        process_dataset_and_save_results(args.dataset, CIRR_DATASET_PATH)
