import json
import os
from evaluate import load


class NonLLMEval:
    def __init__(self, input_file_path, load_prompting_directly=False, lang="en"):
        """
        Initializes the evaluator with the path to a JSONL or JSON file.
        Each entry should contain:
            - "prompt": The original news headline.
            - "risk counterfactual": The risk counterfactual candidate.
            - "opportunity counterfactual": The opportunity counterfactual candidate.

        Args:
            input_file_path (str): Path to the input JSONL or JSON file.
            load_prompting_directly (bool): If True, expects a JSON array where each entry
                has an "output" key containing the fields above (prompting output format).
            lang (str): Language code (default "en").
        """
        self.input_file_path = input_file_path
        self.load_prompting_directly = load_prompting_directly
        self.lang = lang
        self.data = []
        self.prompts = []
        self.risk_candidates = []
        self.opportunity_candidates = []
        self._load_data()

    def _load_data(self):
        if self.load_prompting_directly:
            with open(self.input_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            for entry in json_data:
                output = entry.get("output", {})
                self.data.append(output)
                self.prompts.append(output.get("original_headline", ""))
                self.risk_candidates.append(output.get("risk_counterfactual", ""))
                self.opportunity_candidates.append(output.get("opportunity_counterfactual", ""))

        else:
            with open(self.input_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        self.data.append(obj)
                        self.prompts.append(obj.get("prompt", ""))
                        self.risk_candidates.append(obj.get("risk counterfactual", ""))
                        self.opportunity_candidates.append(obj.get("opportunity counterfactual", ""))

    def compute_perplexity_scores(self, model_id: str = "gpt2"):
        """
        Compute sentence-level perplexity for each candidate using a language model.
        Assigns None to empty candidates to preserve list size.

        Args:
            model_id (str): HuggingFace model ID to use for perplexity (default "gpt2").

        Returns:
            dict with keys:
                - "risk_perplexity": list of perplexity scores for risk counterfactuals.
                - "opp_perplexity": list of perplexity scores for opportunity counterfactuals.
                - "org_perplexity": list of perplexity scores for original headlines.
        """
        ppl_metric = load("perplexity", module_type="metric")

        def compute_with_placeholders(predictions):
            results = []
            non_empty_preds = [p for p in predictions if p.strip() != ""]
            if non_empty_preds:
                non_empty_ppls = ppl_metric.compute(predictions=non_empty_preds, model_id=model_id)["perplexities"]
            else:
                non_empty_ppls = []
            idx = 0
            for p in predictions:
                if p.strip() == "":
                    results.append(None)
                else:
                    results.append(non_empty_ppls[idx])
                    idx += 1
            return results

        self.risk_perplexity = compute_with_placeholders(self.risk_candidates)
        self.opp_perplexity  = compute_with_placeholders(self.opportunity_candidates)
        self.org_perplexity  = compute_with_placeholders(self.prompts)

        return {
            "risk_perplexity": self.risk_perplexity,
            "opp_perplexity": self.opp_perplexity,
            "org_perplexity": self.org_perplexity
        }


if __name__ == "__main__":
    # Example usage — replace with your actual input file path
    input_file = "path/to/your/results.jsonl"

    evaluator = NonLLMEval(
        input_file_path=input_file,
        load_prompting_directly=False,
        lang="en"
    )

    perplexity_scores = evaluator.compute_perplexity_scores(model_id="gpt2")

    risk_ppls = [s for s in perplexity_scores["risk_perplexity"] if s is not None]
    opp_ppls  = [s for s in perplexity_scores["opp_perplexity"] if s is not None]
    org_ppls  = [s for s in perplexity_scores["org_perplexity"] if s is not None]

    print(f"Risk Perplexity     — Avg: {sum(risk_ppls)/len(risk_ppls):.4f}" if risk_ppls else "No risk results")
    print(f"Opportunity Perplexity — Avg: {sum(opp_ppls)/len(opp_ppls):.4f}" if opp_ppls else "No opportunity results")
    print(f"Original Perplexity — Avg: {sum(org_ppls)/len(org_ppls):.4f}" if org_ppls else "No original results")
