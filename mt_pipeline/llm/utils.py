import os
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

GPT_4O_COST = {
    "input": 2.5 / 1000000,
    "output": 10.00 / 1000000,
}

FIQ_DATASET_PATH = f"{ROOT_PATH}/datasets/FASHIONIQ"
CIRR_DATASET_PATH = f"{ROOT_PATH}/datasets/CIRR"


def calculate_cost(usage_dict: dict) -> float:
    input_token = usage_dict.prompt_tokens
    output_token = usage_dict.completion_tokens
    return round(
        output_token * GPT_4O_COST["output"] + input_token * GPT_4O_COST["input"], 5
    )
