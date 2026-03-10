import re
import base64
import json
import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

from prompt import DUAL_CONSTRAINT_PROMPT


def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
    return image_b64


def build_vision_prompt(image_b64, prompt_text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
    ]


def generate_dual_constraints(
    client, ref_img_path, mod_text, openai_model, max_retry=5
):
    image_b64 = encode_image(ref_img_path)
    filled_prompt = DUAL_CONSTRAINT_PROMPT.format(mod_text=mod_text)
    messages = build_vision_prompt(image_b64, filled_prompt)

    retry_count = 0
    while retry_count < max_retry:
        try:
            response = client.chat.completions.create(
                model=openai_model, temperature=0.0, messages=messages
            )

            # Extract the first JSON object from the response string
            match = re.search(
                r"\{.*?\}", response.choices[0].message.content, re.DOTALL
            )
            if match:
                response_dict = json.loads(match.group(0))
            else:
                raise ValueError("No JSON object found in response.")

            return response_dict, response.usage

        except Exception as e:
            retry_count += 1
            print(f"Failed to get_response -> {e}")

    return {
        "attributes": {"keep": [], "add": [], "remove": []},
        "queries": {"prescriptive_query": "", "proscriptive_query": ""},
    }, response.usage
