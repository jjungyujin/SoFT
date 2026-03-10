import sys
from openai import OpenAI
import os
import tqdm
import json
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

from llm import generate_dual_constraints


def get_ref_img_path(dataset_name, query_dataset, reference_names, i):
    if dataset_name == "circo":
        ref_img_path = query_dataset.img_paths[
            query_dataset.img_ids_indexes_map[reference_names[i]]
        ]
    elif "cirr" in dataset_name:
        ref_img_path = os.path.join(
            query_dataset.dataset_path,
            query_dataset.name_to_relpath[reference_names[i]],
        )
    elif "fashioniq" in dataset_name:
        if "multi" in dataset_name or "single" in dataset_name:
            ref_img_path = os.path.join(
                query_dataset.dataset_path,
                f"images/{query_dataset.triplets[i]['reference_image_name']}.png",
            )
        else:
            ref_img_path = os.path.join(
                query_dataset.dataset_path,
                f"images/{query_dataset.triplets[i]['candidate']}.png",
            )
    else:
        raise ValueError("unvalid dataset name")
    return ref_img_path


def get_dual_constraints(
    preload_dir: str,
    dataset_name: str,
    query_dataset,
    mod_texts,
    reference_names,
    index_features,
    device,
    clip_model,
    openai_model: str,
):
    os.makedirs(preload_dir, exist_ok=True)
    preload_path = os.path.join(
        preload_dir, f"{dataset_name}_const_{openai_model.replace('.', '')}.json"
    )

    all_attributes, all_queries, input_token, output_token = [], [], 0, 0

    if os.path.exists(preload_path):
        with open(preload_path, "r") as f:
            preload_data = json.load(f)
            all_attributes = preload_data.get("all_attributes", [])
            all_queries = preload_data.get("all_queries", [])
            input_token = preload_data.get("input_token", 0)
            output_token = preload_data.get("output_token", 0)

    start_idx = len(all_queries)
    load_dotenv()
    client = OpenAI()

    for i in tqdm.trange(
        start_idx, len(mod_texts), desc="Generating dual constraints with LLM ..."
    ):
        try:
            ref_img_path = get_ref_img_path(
                dataset_name, query_dataset, reference_names, i
            )
            mod_text = mod_texts[i]
            response_dict, usage_dict = generate_dual_constraints(
                client, ref_img_path, mod_text, openai_model
            )
            all_attributes.append(response_dict["attributes"])
            all_queries.append(response_dict["queries"])
            input_token += usage_dict.prompt_tokens
            output_token += usage_dict.completion_tokens

            dump_dict = {
                "model_name": openai_model,
                "input_token": input_token,
                "output_token": output_token,
                "all_queries": all_queries,
                "all_attributes": all_attributes,
            }
            with open(preload_path, "w") as f:
                json.dump(dump_dict, f, indent=4)

        except Exception as e:
            print(f"Error at index {i}: {e}\nSaving progress and exiting...")
            with open(preload_path, "w") as f:
                json.dump(
                    {
                        "model_name": openai_model,
                        "input_token": input_token,
                        "output_token": output_token,
                        "all_queries": all_queries,
                        "all_attributes": all_attributes,
                    },
                    f,
                    indent=4,
                )
            break

    return all_queries
