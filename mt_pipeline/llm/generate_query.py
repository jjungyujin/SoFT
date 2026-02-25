# %%
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import base64
import os
import json
from os import sys

sys.path.append(os.path.dirname(__file__))

from utils import calculate_cost, ROOT_PATH
from prompt.multi_target_candidates import GEN_FIQ_QUERY, GEN_CIRR_QUERY


# %%
def generate_query(dataset, dataset_path, images_names, captions):
    load_dotenv(os.path.join(ROOT_PATH, ".env"))
    client = OpenAI()

    captions_list = []
    costs = []
    for image_name, caps in tqdm(zip(images_names, captions)):
        if dataset == "fiq":
            img_path = f"{dataset_path}/images/{image_name}.png"
            prompt_text = GEN_FIQ_QUERY.format(caption1=caps[0], caption2=caps[1])
        if dataset == "cirr":
            img_path = f"{dataset_path}/dev/{image_name}.png"
            prompt_text = GEN_CIRR_QUERY.format(caption=caps)

        with open(img_path, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        data_uri = {"url": f"data:image/jpeg;base64,{b64}"}

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": data_uri,
                        },
                    ],
                }
            ],
        )
        total_cost = calculate_cost(response.usage)

        output_text = response.choices[0].message.content.split("{", 1)[1]
        output_text = "}".join(output_text.split("}")[:-1])
        output_text = "{" + output_text + "}"

        try:
            output = json.loads(output_text)
        except:
            output = {"sentence1": " and ".join(caps), "sentence2": ""}

        costs.append(total_cost)

        temp = []
        for _, query in output.items():
            temp.append(query)
        captions_list.append(temp)
    return captions_list, costs
