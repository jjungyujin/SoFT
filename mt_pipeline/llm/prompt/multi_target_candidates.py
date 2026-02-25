GEN_FIQ_QUERY = """You are given two pieces of input:
1. A reference image description that tells you what the original product looks like.
2. A User's Modifications text that describes how the user wants the product to be changed.

Your task is to write a two-sentence description of the product the user wants.

- The sentence1 should focus on the only User's modifications — describe the product primarily according to the User's Modifications.
- The sentence2 should preserve the elements from the reference image that do **not** conflict with the User's Modifications, describing details from the original product that can be retained. 
  If there are no additional elements to preserve beyond what is stated in the User's Modifications, return an empty string.
  
Describe the image using only concrete, observable attributes (e.g., color, shape, texture, pattern, material).
Avoid indirect or relative terms like “more,” “less,” “similar,” or “retain.” Use precise, objective language.
Make sure both sentences are concise and consistent. Do not repeat the same attribute across both sentences.

## User's Modifications
1. "{caption1}"
2: "{caption2}"

## Output form (JSON)
{{
    "sentence1" : "...",
    "sentence2" : "..."
}}
"""

GEN_CIRR_QUERY = """You are given two pieces of input:
1. A reference image description that tells you what the original image looks like.
2. A User's Modifications text that describes how the user wants the image to be changed.

Your task is to write a two-sentence description of the image the user wants.

- The sentence1 should focus on the only User's modifications — describe the image primarily according to the User's Modifications.
- The sentence2 should preserve the elements from the reference image that do **not** conflict with the User's Modifications, describing details from the original image that can be retained. 
  If there are no additional elements to preserve beyond what is stated in the User's Modifications, return an empty string.
  
Describe the image using only concrete, observable attributes.
Make sure both sentences are concise and consistent. Do not repeat the same attribute across both sentences.

## User's Modifications
"{caption}"

## Output form (JSON)
{{
    "sentence1" : "...",
    "sentence2" : "..."
}}
"""
