"""Prompts scoped for handwritten Arabic OCR (diacritics optional in wild handwriting)."""

USER_OCR_PROMPT = (
    "You are an expert Arabic handwriting recognition system.\n"
    "Look at the image and transcribe exactly what is written — nothing more, nothing less.\n"
    "Rules:\n"
    "- If the image contains one short line, output only that line.\n"
    "- Keep Eastern Arabic digits (٠١٢٣٤٥٦٧٨٩) exactly as they appear.\n"
    "- Keep slashes, dots, and punctuation exactly as visible.\n"
    "- If the writing has no diacritics (tashkeel), do not invent diacritics.\n"
    "- Do not continue, explain, translate, or add any text beyond what is in the image.\n"
    "- Preserve original spelling.\n"
    "Output only the transcribed Arabic text."
)

SYSTEM_PROMPT = None
