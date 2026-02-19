from pydantic import BaseModel
from typing import Dict, List, Any, Type, Union, get_args, get_origin

class Counterfactuals(BaseModel):
    original_headline: str
    opportunity_counterfactual: str
    risk_counterfactual: str

class Counterfactuals_COT(BaseModel):
    original_headline: str
    reasoning: str
    opportunity_counterfactual: str
    risk_counterfactual: str

class Counterfactual_masked(BaseModel):
    original_masked_headline: str
    opportunity_counterfactual: str
    risk_counterfactual: str