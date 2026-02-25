import json
import base64
import os
import argparse
import sys

from tqdm import tqdm
from datasets.CIRCO.src import dataset
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

sys.path.append(os.path.dirname(__file__))
from utils import calculate_cost, ROOT_PATH, FIQ_DATASET_PATH, CIRR_DATASET_PATH
from prompt.multi_target_scoring import SCORING_PROMPT


load_dotenv(os.path.join(ROOT_PATH, ".env"))


class CandidateScorer:
    def __init__(self, llm_model, api_key=None):
        self.llm_model = llm_model
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def create_prompt(
        self, ref_image_name, relative_captions: List[str], top_k_names: List[str]
    ) -> str:
        prompt = SCORING_PROMPT.format(
            {
                "ref_image_name": ref_image_name,
                "input_relative_captions": chr(10).join(
                    [f"- {caption}" for caption in relative_captions]
                ),
                "candidate_images": chr(10).join([f"- {name}" for name in top_k_names]),
                "output_relative_captions": relative_captions,
            }
        )
        return prompt

    def get_score_by_llm(
        self,
        item_data: Dict[str, Any],
        image_base_path: str = "",
        top_k_sample=10,
    ) -> Dict[str, Any]:
        ref_image_name = item_data["ref_image_name"] + ".png"
        relative_captions = item_data["relative_captions"]
        target_image_name = item_data["target_image_name"] + ".png"

        top_k_names_list = list(map(lambda x: x + ".png", item_data["top_k_names"]))
        top_k_names = top_k_names_list[:top_k_sample]

        total_cost = 0

        try:
            ref_image_path = os.path.join(image_base_path, ref_image_name)
            ref_image_b64 = self.encode_image(ref_image_path)

            candidate_images_b64 = {}
            for candidate_name in top_k_names:
                candidate_path = os.path.join(image_base_path, candidate_name)
                candidate_images_b64[candidate_name] = self.encode_image(candidate_path)

            system_prompt = self.create_prompt(
                ref_image_name, relative_captions, top_k_names
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is the reference image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{ref_image_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Here are the candidate images to evaluate:",
                        },
                    ],
                },
            ]

            for i, (candidate_name, candidate_img) in enumerate(
                candidate_images_b64.items()
            ):
                messages[1]["content"].extend(
                    [
                        {"type": "text", "text": f"Candidate {i+1}: {candidate_name}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{candidate_img}"
                            },
                        },
                    ]
                )

            messages[1]["content"].append(
                {
                    "type": "text",
                    "text": "Please analyze all images and provide your selection in the specified JSON format.",
                }
            )

            response = self.client.chat.completions.create(
                model=self.llm_model, messages=messages, temperature=0.0
            )

            # Parse response
            response_text = response.choices[0].message.content
            total_cost = calculate_cost(response.usage)

            try:
                response_text = response_text.split("{", 1)[1]
                response_text = "}".join(response_text.split("}")[:-1])
                response_text = "{" + response_text + "}"

                result = json.loads(response_text)
                result["target_image"] = target_image_name

                return {
                    "cost": total_cost,
                    "success": True,
                    "result": result,
                    "raw_response": response_text,
                }

            except json.JSONDecodeError:
                return {
                    "cost": total_cost,
                    "success": False,
                    "error": "Failed to parse JSON from GPT response",
                    "raw_response": response_text,
                }

        except Exception as e:
            return {
                "cost": total_cost,
                "success": False,
                "error": str(e),
            }

    def get_candidate_scores(
        self,
        json_file_path: str,
        image_base_path: str = "",
        output_file: str = None,
    ) -> List[Dict[str, Any]]:
        with open(json_file_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        results = []
        n_items = len(items)
        print(f"Processing {n_items} items...")
        for i, item in tqdm(enumerate(items), total=n_items):
            if len(item["top_k_names"]) > 0:
                try:
                    result = self.get_score_by_llm(
                        self.llm_model, item, image_base_path, top_k_sample=10
                    )
                    result["item_index"] = i
                    results.append(result)

                except Exception as e:
                    print(f"  ✗ Error processing item {i+1}: {str(e)}")
                    results.append(
                        {
                            "success": False,
                            "error": str(e),
                            "item_index": i,
                            "item_info": {
                                "ref_image": item.get("ref_image_name", "Unknown")
                            },
                        }
                    )
            else:
                result = {
                    "ref_image_name": item["ref_image_name"] + ".png",
                    "relative_captions": item["relative_captions"],
                    "confidence_scores": {},
                    "selected_top_3": [],
                    "target_image": item["target_image_name"] + ".png",
                }
                results.append(
                    {
                        "cost": 0,
                        "success": True,
                        "result": result,
                        "raw_response": "",
                        "item_index": i,
                    }
                )

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")

        successful = sum(1 for r in results if r["success"])
        print(
            f"\nProcessing complete: {successful}/{n_items} items processed successfully"
        )
        return results


def main(args):
    dataset = args.dataset
    llm_model = args.llm_model

    scorer = CandidateScorer(llm_model=llm_model)
    batch_types = ["caption_query", "joined_query", "tg_img"]

    for batch_type in batch_types:
        if dataset == "fiq":
            image_base_path = f"{FIQ_DATASET_PATH}/images"
            dress_types = ["shirt", "toptee", "dress"]

        elif dataset == "cirr":
            image_base_path = f"{CIRR_DATASET_PATH}/dev"
            dress_types = ["cirr"]

        for dress_type in dress_types:
            json_file_path = f"{ROOT_PATH}/mt_pipeline/llm/multi_target_candidate/mt_candidates_{dress_type}_by_{batch_type}_20.json"
            output_file = f"{dress_type}_by_{batch_type}_20_scores.json"
            _results = scorer.get_candidate_scores(
                json_file_path, image_base_path, output_file
            )
            print("\n" + "=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process multi-target confidence calculation."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, default="fiq", choices=["fiq", "cirr"]
    )
    parser.add_argument(
        "--llm_model", type=str, default="gpt-4o", help="LLM model to use for scoring"
    )
    args = parser.parse_args()

    main(args)
