#!/usr/bin/env python3
import yaml
import json
import os
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
from typing import List, Dict

### LLM EVALUATION Metrics

### VALIDITY/ DIRECTIONALITY (different outcomes from input)

### Forward-Looking (does the edited opportunity and risk logically follow from the original news article?)

### LOGICAL COHERENCE (are the counterfactuals LOGICALLY COHERENT? I.E. do they describe a logical counterfactual)

# --- Evaluator Class ---
class LLMEvaluator:
    def __init__(self, api_key: str, input_file: str, rubric_filepath: str,
                 model: str, response_model=None, opportunity_rubric_filepath: str = None, raw_output_file: str = None, load_prompting_directly: bool = False):
        """
        Initializes the evaluator with:
         - API key (set as an environment variable)
         - Path to the JSONL file with prompt/response entries
         - Path to the external evaluation rubric text file for risk counterfactual
         - The API client and configuration dictionary (which must include an "output_base" key)
         - Optional: Custom response model (defaults to MultiEvaluationOutput if not provided)
         - Optional: Path to a separate rubric file for opportunity counterfactual
        """
        self.input_file = input_file
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.response_model = response_model
        self.load_prompting_directly = load_prompting_directly
        self.raw_output_file = raw_output_file

        # Load evaluation rubric for risk counterfactual
        self.risk_system_role = self.load_rubric_file(rubric_filepath)
        
        # If a separate rubric for opportunity counterfactual is provided, load it
        # Otherwise, use the same rubric for both risk and opportunity
        if opportunity_rubric_filepath:
            self.opportunity_system_role = self.load_rubric_file(opportunity_rubric_filepath)
        else:
            self.opportunity_system_role = self.risk_system_role

    def load_rubric_file(self, filepath: str) -> str:
        """
        Reads and returns the content of the evaluation rubric file.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                rubric = f.read()
            return rubric
        except Exception as e:
            print(f"Error reading rubric file {filepath}: {e}")
            return ""
        
    def load_json_prompting_format(self) -> list:
        """
        Reads a JSON file (not JSONL) and returns a list of dictionaries with:
        - "prompt": from output["original_headline"]
        - "risk counterfactual": from output["risk_counterfactual"]
        """
        data = []
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)  # Load full JSON object/array
                for entry in json_data:
                    output = entry.get("output", {})
                    data.append({
                        "prompt_id": entry.get("item_index",""),
                        "prompt": output.get("original_headline", ""),
                        "risk counterfactual": output.get("risk_counterfactual", ""),
                        "opportunity counterfactual": output.get("opportunity_counterfactual", "")
                    })
        except Exception as e:
            print(f"Error reading JSON file in prompting format from {self.input_file}: {e}")
        return data


    def load_jsonl(self) -> list:
        """
        Reads the JSONL file and returns a list of dictionaries.
        """
        data = []
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        except Exception as e:
            print(f"Error reading JSONL file {self.input_file}: {e}")
        return data

    def prepare_evaluation_input(self, prompt: str, response: str) -> str:
        """
        Prepares the evaluation input by combining the prompt and response.
        """
        evaluation_input = (
            f"<news>{prompt}</news>\n"
            f"<response>{response}</response>\n"
            "Please evaluate the above response based on the provided rubric."
        )
        return evaluation_input

    def call_openai_api(self, content: str, counterfactual_type: str = "risk") -> dict:
        """
        Calls the API using the provided client and configuration.
        The system message is the rubric loaded from the external file.
        The user message is the formatted evaluation input.
        
        Args:
            content: The formatted evaluation input
            counterfactual_type: Either "risk" or "opportunity" to select the appropriate system role
        """
        # Select the appropriate system role based on counterfactual type
        if counterfactual_type.lower() == "opportunity":
            system_role = self.opportunity_system_role
        else:  # Default to risk
            system_role = self.risk_system_role
            
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": content}
        ]
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=self.response_model,
            )
            # Validate and parse the structured response using the provided model
            parsed_response = self.response_model.model_validate(response.choices[0].message.parsed)
            return parsed_response.dict()
        except Exception as e:
            print(f"Error during API call for {counterfactual_type} evaluation: {e}")
            return {}

    def evaluate_responses(self) -> list:
        """
        Iterates over each entry in the JSONL file containing prompt, risk counterfactual, and opportunity counterfactual.
        Evaluates each counterfactual using the API with the appropriate system role
        and returns a list of evaluation results, saving each result to the raw output file.
        """
        if self.load_prompting_directly:
            data = self.load_json_prompting_format()
        else:
            data = self.load_jsonl()

        all_results = []

        for entry in tqdm(data, desc="Evaluating counterfactuals", unit="entry"):
            prompt_id = entry.get("prompt_id", "")
            prompt = entry.get("prompt", "")
            risk_counterfactual = entry.get("risk counterfactual", "")
            opportunity_counterfactual = entry.get("opportunity counterfactual", "")
            
            # Evaluate risk counterfactual with risk rubric
            risk_evaluation_input = self.prepare_evaluation_input(prompt, risk_counterfactual)
            risk_evaluation_result = self.call_openai_api(risk_evaluation_input, counterfactual_type="risk")
            
            # Evaluate opportunity counterfactual with opportunity rubric
            opportunity_evaluation_input = self.prepare_evaluation_input(prompt, opportunity_counterfactual)
            opportunity_evaluation_result = self.call_openai_api(opportunity_evaluation_input, counterfactual_type="opportunity")
            
            # Store evaluation results
            result_entry = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "risk_counterfactual": risk_counterfactual,
                "opportunity_counterfactual": opportunity_counterfactual,
                "risk_evaluation": risk_evaluation_result,
                "opportunity_evaluation": opportunity_evaluation_result
            }
            
            all_results.append(result_entry)
            
            # Save each result to the raw output file if specified
            if self.raw_output_file:
                try:
                    with open(self.raw_output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result_entry) + '\n')
                except Exception as e:
                    print(f"Error saving to raw output file: {e}")
        
        return all_results