import json
import random
import openai
import os
from os import sys
import base64
import argparse

from tqdm import tqdm
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional


BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

from utils import calculate_cost, FIQ_DATASET_PATH, CIRR_DATASET_PATH
from prompt.single_target_caption import GEN_SINGLE_TARGET_CAP

load_dotenv()
SEED_NUM = 42
random.seed(SEED_NUM)


class SelectionSingleTarget:
    def __init__(self, dataset, json_file_path: str):
        self.dataset = dataset
        self.json_file_path = json_file_path
        self.client = openai.OpenAI()
        self.data = self.load_data()

    def load_data(self) -> List[Dict]:
        with open(self.json_file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def filter_mt_triplets(self, min_targets: int = 3) -> List[Dict]:
        filtered_triplets = []
        for triplet in self.data:
            confidence_count = len(triplet.get("confidence_scores", {}))
            if confidence_count >= min_targets:
                filtered_triplets.append(triplet)
        return filtered_triplets

    def select_target_and_comparisons(
        self, item: Dict, n_comparisons: int
    ) -> Tuple[str, List[str]]:
        confidence_scores = item["confidence_scores"]
        available_targets = list(confidence_scores.keys())

        # random sampling for single target
        single_target = random.choice(available_targets)

        # random sampling for contrastive distractors
        remaining_targets = [t for t in available_targets if t != single_target]
        comparison_images = random.sample(remaining_targets, n_comparisons)

        return single_target, comparison_images

    def encode_image_to_base64(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Image Encoding Error ({image_path}): {e}")
            return ""

    def get_image_path(self, image_name: str) -> str:
        if self.dataset == "fiq":
            images_dir = f"{FIQ_DATASET_PATH}/images"
        elif self.dataset == "cirr":
            images_dir = f"{CIRR_DATASET_PATH}/dev"
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        return os.path.join(images_dir, image_name)

    def generate_refined_caption(
        self,
        reference_image: str,
        single_target: str,
        comparison_images: List[str],
        original_captions: List[str],
    ) -> str:

        ref_image_path = f"{self.get_image_path(reference_image)}.png"
        target_image_path = f"{self.get_image_path(single_target)}.png"
        comparison_image_paths = [
            f"{self.get_image_path(img)}.png" for img in comparison_images
        ]

        ref_image_base64 = self.encode_image_to_base64(ref_image_path)
        target_image_base64 = self.encode_image_to_base64(target_image_path)
        comparison_images_base64 = [
            self.encode_image_to_base64(path) for path in comparison_image_paths
        ]

        messages = [
            {
                "role": "system",
                "content": "You are a helpful image description expert for various types of images.",
            }
        ]

        if ref_image_base64:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reference image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{ref_image_base64}"
                            },
                        },
                    ],
                }
            )

        if target_image_base64:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Target image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{target_image_base64}"
                            },
                        },
                    ],
                }
            )

        if comparison_images_base64:
            comparison_content = [{"type": "text", "text": "Comparison images:"}]
            for img_base64 in comparison_images_base64:
                if img_base64:
                    comparison_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        }
                    )

            if len(comparison_content) > 1:
                messages.append({"role": "user", "content": comparison_content})

        text_prompt = GEN_SINGLE_TARGET_CAP.format(
            {
                "n_comparison": len(comparison_images),
                "original_caption": ", ".join(original_captions),
            }
        )
        messages.append({"role": "user", "content": text_prompt})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
            )
            api_cost = calculate_cost(response.usage)
            return response.choices[0].message.content.strip(), api_cost

        except Exception as e:
            print(f"GPT API ERROR: {e}")
            return ". ".join(original_captions) + " (refined)", 0

    def generate_st_triplet(
        self, mt_triplet: Dict, n_comparisons: int
    ) -> Optional[Dict]:
        try:
            single_target, comparison_images = self.select_target_and_comparisons(
                mt_triplet, n_comparisons
            )

            refined_caption, api_cost = self.generate_refined_caption(
                reference_image=mt_triplet["reference_image_name"],
                single_target=single_target,
                comparison_images=comparison_images,
                original_captions=mt_triplet["relative_captions"],
            )

            st_triplet = {
                "reference_image_name": mt_triplet["reference_image_name"],
                "target_image_name": single_target,
                "original_target": mt_triplet["target_image_name"],
                "original_captions": mt_triplet["relative_captions"],
                "refined_caption": refined_caption,
                "comparison_images": comparison_images,
                "target_confidence": mt_triplet["confidence_scores"][single_target],
                "original_confidence_scores": mt_triplet["confidence_scores"],
                "api_cost": api_cost,
            }

            return st_triplet

        except Exception as e:
            print(f"Error: {e}")
            return None

    def construct_st_dataset(self, output_file: str, min_targets: int = 3) -> None:
        filtered_triplets = self.filter_mt_triplets(min_targets)
        print(f"Number of single target triplets: {len(filtered_triplets)}")

        st_triplets = []

        for mt_triplet in tqdm(
            filtered_triplets, desc="Processing items", total=len(filtered_triplets)
        ):
            st_triplet = self.generate_st_triplet(
                mt_triplet, n_comparisons=min_targets - 1
            )
            if st_triplet:
                st_triplets.append(st_triplet)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(st_triplets, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["fiq", "cirr"])
    args = parser.parse_args()
    print(f"Processing dataset: {args.dataset}")
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MT_SELECTION_DIR = f"{BASE_DIR}/multi_target_selection"
    SINGLE_SELECTION_DIR = f"{BASE_DIR}/single_target_selection"

    if args.dataset == "fiq":
        dress_types = ["dress", "toptee", "shirt"]

        for dress_type in dress_types:
            print(f"\n{'='*50}")
            print(f"Processing FashionIQ - {dress_type}")

            input_file = f"{MT_SELECTION_DIR}/{dress_type}_multi_target_selection.json"
            output_file = f"{SINGLE_SELECTION_DIR}/{dress_type}_selection_single_target_seed{SEED_NUM}.json"

            print(f"Input: {input_file}")
            print(f"Output: {output_file}")

            processor = SelectionSingleTarget(
                dataset=args.dataset,
                json_file_path=input_file,
            )

            processor.construct_st_dataset(output_file=output_file, min_targets=3)

            print(f"Completed processing {dress_type}")

    elif args.dataset == "cirr":
        print(f"\n{'='*50}")
        print(f"Processing CIRR")

        input_file = f"{MT_SELECTION_DIR}/cirr_multi_target_selection.json"
        output_file = (
            f"{SINGLE_SELECTION_DIR}/cirr_selection_single_target_seed{SEED_NUM}.json"
        )

        print(f"Input: {input_file}")
        print(f"Output: {output_file}")

        processor = SelectionSingleTarget(
            dataset=args.dataset,
            json_file_path=input_file,
        )

        processor.construct_st_dataset(output_file=output_file, min_targets=3)

        print(f"Completed processing CIRR")
