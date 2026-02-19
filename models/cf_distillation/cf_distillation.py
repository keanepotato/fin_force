import os
import re
import json
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import spacy
import yaml

nlp = spacy.load("en_core_web_sm")

class TopicWord(BaseModel):
    topic_word: str

def get_noun_phrases(text: str) -> List[str]:
    """Return a list of noun phrases from the input text."""
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]

def mask_phrases(text: str, phrases: List[str], mask_token: str = "[MASK]") -> str:
    """
    Replace each whole-phrase match in `phrases` with `mask_token`.
    Longer phrases first to avoid partial masking.
    """
    masked = text
    # sort by length (desc) so e.g. "lazy dog" is masked before "dog"
    for phrase in sorted(set(phrases), key=len, reverse=True):
        # \b boundaries ensure we only match whole words/phrases
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        masked = pattern.sub(mask_token, masked)
    return masked

def openai_inference(
    json_input_file_path: str,
    output_file_path: str,
    model: str,
    api_key: str,
):
    """
    1. Reads `items = json.load(input_path)`, expects each item to have "headline".
    2. Loads or creates `processed = []` from output_path.
    3. For each un-processed headline:
       • Calls OpenAI to get a TopicWord
       • Extracts noun phrases
       • Masks topic word + noun phrases in the headline
       • Appends a record {headline, topic_word, noun_phrases, masked_content}
         to `processed` and immediately writes it back to disk.
    """
    # initialize client
    client = OpenAI(api_key=api_key)

    # load all items to process
    with open(json_input_file_path, "r") as f:
        items = json.load(f)
    print(f"Found {len(items)} items to process")

    # load existing processed (if any)
    if os.path.exists(output_file_path):
        with open(output_file_path, "r") as f:
            processed = json.load(f)
    else:
        processed = []

    # keep track of which headlines we've done
    done = {rec["headline"] for rec in processed}

    system_role = """
    You are a topic word extractor. Your task is to extract the most relevant topic word from the given text.
    """

    for item in tqdm(items, desc="Processing items"):
        content = item["headline"]
        if content in done:
            continue

        # 1) API call to get topic word
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user",   "content": content}
        ]
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=TopicWord,
        )
        tw = TopicWord.model_validate(response.choices[0].message.parsed)
        topic = tw.topic_word

        # 2) get noun phrases
        noun_phrases = get_noun_phrases(content)

        # 3) mask topic word + noun phrases in the content
        masked = mask_phrases(content, [topic] + noun_phrases)

        # assemble record
        record = {
            "headline": content,
            "topic_word": topic,
            "noun_phrases": noun_phrases,
            "masked_content": masked
        }
        processed.append(record)
        done.add(content)

        # 5) immediately save the growing list
        with open(output_file_path, "w") as f_out:
            json.dump(processed, f_out, indent=2)

    print(f"Done! Processed {len(processed)} items; saved to {output_file_path}")
    return processed

if __name__ == "__main__":
    with open("cf_distillation_config.yaml", "r") as yf:
        cfg = yaml.safe_load(yf)

    inp    = cfg["input"]
    outp   = cfg["output"]
    model  = cfg["model"]
    key    = cfg.get("api_key") or os.getenv("OPENAI_API_KEY")

    if not key:
        raise RuntimeError(
            "No api_key in config and OPENAI_API_KEY is not set in the environment."
        )

    openai_inference(
        json_input_file_path=inp,
        output_file_path=outp,
        model=model,
        api_key=key,
    )
