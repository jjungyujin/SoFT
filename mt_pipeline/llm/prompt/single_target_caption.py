GEN_SINGLE_TARGET_CAP = """
You are an expert caption-writer for composed-image retrieval across various domains.

The images above show:
1. Reference image
2. Target image
3. Comparison images (similar but incorrect): {n_comparison} images

Original captions: {original_caption}

The original captions describe how to transform the reference image into the target image, but they are not specific enough to rule out the comparison images.

Task:
Write one refined caption that uniquely identifies the target image while staying faithful to the intent of the original captions.

Guidelines:
- Use the original captions as a foundation; keep their meaning while making them more specific.
- Preserve the original captions' writing tone and sentence style.
- Add concrete, observable details present in the target but absent from the comparison images (color, texture, shape, spatial relation, object type, etc.).
- Be concise and output exactly one sentence—no bullet points or extra text.

Return only the refined caption.
"""
