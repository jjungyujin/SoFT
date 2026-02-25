# %%
import glob
import json
from collections import defaultdict
import os
import argparse


# %%
def get_dataset_config(dataset):
    if dataset == "fiq":
        return {"dress_types": ["shirt", "toptee", "dress"], "has_dress_types": True}
    elif dataset == "cirr":
        return {"dress_types": [None], "has_dress_types": False}
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def get_file_pattern_and_output_path(dataset, dress_type, input_dir, output_dir):
    if dataset == "fiq":
        pattern = os.path.join(input_dir, f"{dress_type}_*.json")
        output_path = os.path.join(
            output_dir, f"{dress_type}_multi_target_selection.json"
        )
    elif dataset == "cirr":
        pattern = os.path.join(input_dir, "cirr_*.json")
        output_path = os.path.join(output_dir, "cirr_multi_target_selection.json")
    return pattern, output_path


def construct_mt_dataset(dataset, threshold=0.85):
    # set paths relative to this script file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "candidate_score")
    output_dir = os.path.join(base_dir, "multi_target_selection")

    os.makedirs(output_dir, exist_ok=True)

    config = get_dataset_config(dataset)

    for dress_type in config["dress_types"]:
        grouped = defaultdict(dict)

        pattern, output_path = get_file_pattern_and_output_path(
            dataset, dress_type, input_dir, output_dir
        )

        grouped = select_multi_target(pattern, grouped, threshold)

        dataset_label = dress_type if config["has_dress_types"] else dataset
        save_results(grouped, output_path, threshold, dataset_label)


def select_multi_target(pattern, grouped, threshold):
    for filepath in glob.glob(pattern):
        print(f"Processing file: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            candidate_scores = json.load(f)

        for score_dict in candidate_scores:
            result_dict = score_dict["result"]
            grouped_key = result_dict["item_index"]
            if grouped_key not in grouped:
                grouped[grouped_key] = {
                    "reference_image_name": result_dict["reference_image_name"].split(
                        "."
                    )[0],
                    "threshold": threshold,
                    "target_image_name": result_dict["target_image"].split(".")[0],
                    "relative_captions": result_dict["relative_captions"],
                    "confidence_scores": {},
                }

            for candidate_file, score in result_dict["confidence_scores"].items():
                candidate_name = candidate_file.split(".")[0]
                if score >= threshold:
                    if (
                        candidate_name not in grouped[grouped_key]["confidence_scores"]
                        or score
                        > grouped[grouped_key]["confidence_scores"][candidate_name]
                    ):
                        grouped[grouped_key]["confidence_scores"][
                            candidate_name
                        ] = score

    return grouped


def save_results(grouped, output_path, threshold, dataset_type):
    output_list = []
    for mt_triplets in grouped.values():
        if len(mt_triplets["confidence_scores"].items()) > 0:
            output_list.append(mt_triplets)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)

    print(f"Finished! {len(output_list)} target groups written to {output_path}")
    print(f"Dataset: {dataset_type}, Confidence threshold: {threshold}")

    total_candidates = sum(len(item["confidence_scores"]) for item in output_list)
    avg_candidates = total_candidates / len(output_list) if output_list else 0
    print(f"Average candidates per reference image: {avg_candidates:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multi-target candidate results."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fiq", "cirr"],
        help="Dataset to process (fiq or cirr)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for filtering candidates (default: 0.85)",
    )

    args = parser.parse_args()

    print(f"Processing dataset: {args.dataset}")
    print(f"Confidence threshold: {args.threshold}")
    print("-" * 50)

    construct_mt_dataset(args.dataset, args.threshold)
    print("-" * 50)
    print("Processing completed!")


# %%
if __name__ == "__main__":
    main()
