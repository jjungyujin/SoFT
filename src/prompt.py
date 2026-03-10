DUAL_CONSTRAINT_PROMPT = """You are a helpful assistant for a composed image retrieval system.
You are given a reference image and a modification text that describes how the image should be changed.
Your task is to extract meaningful attribute information and generate two types of semantic queries from a reference image and a modification text.

## Step 1. Attribute Classification
Analyze the modification text in the context of the reference image and extract three types of attribute-value lists.
When assigning attributes, ensure that any visually or semantically important nouns (such as objects, items, or main scene elements mentioned in any query) are included in the `keep` or `add` lists — even if they also appear in `remove`.

- `keep`: A list of key attribute values that should remain unchanged in the reference image according to the modification text.
- `add`: A list of attribute values that are not present in the reference image but are explicitly or implicitly required by the modification text.
- `remove`: A list of attribute values that are present in the reference image but are explicitly removed by the modification text.

Do NOT use vague or relative expressions such as “more”, “less”, “better”, “similar”, "same”, “different", or any phrase that refers to the reference image implicitly or explicitly.
Only include attributes in each list when they are explicitly supported by the reference image and modification text.
Do not infer `remove` attributes solely from the `add` list; if an element is missing in the image but required by the modification, include it only in `add`.
If no attributes meet the criteria, leave the list empty.

## Step 2. Query Generation
Using the attribute-value lists from Step 1, generate text queries that can be directly used for image retrieval.
Each query must be fluent, self-contained. If an attribute is described in relative terms, you must resolve it into a concrete absolute caption using visual clues from the reference image.

- `prescriptive_query`: Write a short, specific image caption that describes the visual features that should be included in the target image, focusing on the attribute values in the `keep` and `add` lists.
- `proscriptive_query`: Write a short, specific image caption that describes the reference image as it is, using positive, natural language, focusing on the attribute values in the `remove` list.

## Output Format:
{{
  "attributes" : {{
    "keep": [...], "add": [...], "remove": [...]
  }},
  ""queries" : {{
    "prescriptive_query": "...", "proscriptive_query": "..."
  }}
}}

Do not include any explanations, only return the output in the format specified above.

---

## Input:
Modification Text: "{mod_text}"
Reference image:
"""
