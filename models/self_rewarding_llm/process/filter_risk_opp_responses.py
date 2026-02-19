"""
Process the current responses for risk and opportunity scenarios 
by removing the input prompts and system instructions.
This is for scoring the responses.
"""

import json
import os
from pathlib import Path
import re

# import importlib.util
# import sys
# from pathlib import Path

# # Adjust this path as needed
# file_path = Path("/home/llm_scenario_modelling/models/self_rewarding_llm/scripts/02_gen_responses.py")
# module_name = "gen_responses_dynamic"

# spec = importlib.util.spec_from_file_location(module_name, file_path)
# module = importlib.util.module_from_spec(spec)
# sys.modules[module_name] = module
# spec.loader.exec_module(module)

# # Now use the function
# extract_chatml_completion = module.extract_chatml_completion

def extract_chatml_completion_risk_opp(answer):
    """
    Extracts the last Risk and Opportunity counterfactuals from a ChatML-formatted LLM response.
    Ensures that 'Scenario' variants are not mistakenly extracted from instructions (must appear more than once).
    
    # TODO: There is a functionality to add. Basically where there is just a risk scenario, without any stop symbol (e.g. <|end_text|> or comma)
    # TODO: Where the risk scenario just ends abruptly - We need to add a check for that.

    Returns:
        dict: {'risk': str or None, 'opportunity': str or None}
    """
    def clean_base_response(text):
        # Strip everything up to and including the last <|im_end|>
        if "<|im_end|>" in text:
            text = text.rsplit("<|im_end|>", 1)[-1].strip()
        return text

    def find_last_valid_match(text, labels):
        """
        Find the last match from a list of labels, skipping 'Scenario' matches if they only occur once.
        Returns the last valid match object, or None.
        """
        for label in labels:
            pattern = re.compile(re.escape(label) + r"[:\-]?\s*", re.IGNORECASE)
            matches = list(pattern.finditer(text))
            if matches:
                # If label ends with 'Scenario' and there's only one match, skip it (likely instruction)
                if label in ["Risk Counterfactual Scenario", "Opportunity Counterfactual Scenario"] and len(matches) <= 3:
                    continue
                 # check for risk scenario and counterfactual scenario in the output directly
                elif label in ["Risk Scenario", "Opportunity Scenario"] and len(matches) <= 1:
                    # has to be more than 1 match for this
                    continue
                else:
                    return matches[-1]
        return None

    def extract_following_text(text, match, stop_labels):
        """
        Extract text following a match until the next label (from stop_labels) or end of text.
        """
        if not match:
            return None
        start = match.end()
        remainder = text[start:].strip()

        stop_pattern = re.compile(r"|".join([
            re.escape(label) + r"[:\-]?\s*"
            for label in stop_labels
        ]), re.IGNORECASE)

        stop_match = stop_pattern.search(remainder)
        return remainder[:stop_match.start()].strip() if stop_match else remainder.strip()

    # Main logic (do not clean for now)
    # text = clean_base_response(answer)
    text = answer

    # risk_labels = ["Risk Counterfactual Scenario", "Risk Counterfactual", "Risk Scenario"]
    # opp_labels = ["Opportunity Counterfactual Scenario", "Opportunity Counterfactual", "Opportunity Scenario"]

    risk_labels = ["Risk Counterfactual Scenario", "Risk Scenario"]
    opp_labels = ["Opportunity Counterfactual Scenario", "Opportunity Scenario"]

    risk_match = find_last_valid_match(text, risk_labels)
    opp_match = find_last_valid_match(text, opp_labels)

    return {
        "risk": extract_following_text(text, risk_match, opp_labels),
        "opportunity": extract_following_text(text, opp_match, risk_labels)
    }


def extract_completion_only(answer):
    """
    Parse the response using extract_chatml_completion_risk_opp to get risk and opportunity
    counterfactuals, then combine them into a single formatted string.
    
    Args:
        answer (str): The full response from the model
        
    Returns:
        str: Combined risk and opportunity counterfactuals, or the original answer if extraction fails
    """
    try:
        parsed = extract_chatml_completion_risk_opp(answer)
        risk = parsed.get("risk")
        opportunity = parsed.get("opportunity")
        
        # Check if both risk and opportunity are missing
        if risk is None and opportunity is None:
            print("Warning: No risk or opportunity scenarios found in the response")
            return ""
        
        # Construct the combined result
        combined = ""
        if risk:
            combined += f"Risk Counterfactual Scenario:\n{risk}\n\n"
        if opportunity:
            combined += f"Opportunity Counterfactual Scenario:\n{opportunity}"
            
        return combined.strip()
    except Exception as e:
        print(f"Error extracting completion: {e}")
        return answer  # Return original answer if extraction fails


def process_responses_file(input_path, output_path):
    """
    Process the responses JSONL file and generate a new filtered file.
    
    Args:
        input_path (str): Path to the input JSONL file
        output_path (str): Path to save the output JSONL file
    """
    filtered_responses = []
    # counter = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:

        for line in f:
            # counter += 1
            # if counter == 28:
            #     print("Hello world")
            if line.strip():
                try:
                    entry = json.loads(line)
                    prompt_id = entry.get("prompt_id", "")
                    prompt = entry.get("prompt", "")
                    completion = entry.get("completion", "")
                    
                    # Extract the completion text using the imported function
                    cleaned_completion = extract_completion_only(completion)
                    
                    # Create a filtered entry
                    filtered_entry = {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "completion": cleaned_completion
                    }
                    
                    filtered_responses.append(filtered_entry)
                        
                except Exception as e:
                    print(f"Error processing line: {e}")
                    continue
    
    # Write the filtered responses to the output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in filtered_responses:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Processed {len(filtered_responses)} responses and saved to {output_path}")

def main():
    # Define paths
    # base_dir = Path("/home/llm_scenario_modelling/models/self_rewarding_llm/data")
    input_file = "/home/llm_scenario_modelling/models/self_rewarding_llm/data/CF0/generated/old_responses.jsonl"
    output_file = "/home/llm_scenario_modelling/models/self_rewarding_llm/data/CF0/generated/responses.jsonl"
    
    # Process the responses
    process_responses_file(input_file, output_file)

if __name__ == "__main__":
    main()