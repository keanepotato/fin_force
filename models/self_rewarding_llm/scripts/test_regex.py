import re


pattern = r"[Ss]core: (10|[1-9])"

# test_string = "[/INST]\n\nscore: 8\n\nJustification:"

test_string =  """
<s>[INST] Review the base news and the corresponding counterfactual scenarios using the additive 10-point scoring system described below.\n\nThe original news is enclosed between <news> and </news>, and the generated scenario is enclosed between <response> and </response>.\n\nPoints are accumulated based on the satisfaction of each criterion:\n\n- Add 1 point if the risk counterfactual is topically relevant and reflects a modification or extrapolation of the original news.\n- Add another point if the opportunity counterfactual is also relevant to the original news.\n- Add 1 point if the risk counterfactual makes minimal edits to the original news (preserving structure and intent).\n- Add another point if the opportunity counterfactual also uses minimal edits appropriately.\n- Add 1 point if the risk counterfactual is forward-compatible, describing a plausible future development that does not contradict the original news.\n- Add another point if the opportunity counterfactual is also forward-compatible.\n- Add 1 point if the risk counterfactual clearly reflects an adverse market development.\n- Add another point if the opportunity counterfactual clearly reflects a favorable market development.\n- Add 1 point if the risk counterfactual is cohesive, meaning it is clear, logical, and internally consistent.\n- Add another point if the opportunity counterfactual is cohesive in the same way.\n- If either counterfactual is incoherent, irrelevant, or fails to fulfill any criteria, award 0 points.\n\n<news>U.S. consumer spending drops sharply in June amid rising inflation - Commerce Department</news>\n<response></response>\n\nAfter examining the news and the counterfactual scenario:\n- output the score of the evaluation using this exact format: \"score: <total points>\", where <total points> is between 0 and 10\n- Briefly justify your total score, up to 100 words.\n [/INST] [/INST] [/INST] [/INST]\n\nOutput:\nscore: 8\n\nJustification: The risk counterfactual scenario maintains the original news's structure and intent while introducing a plausible future development. The opportunity counterfactual scenario also makes minimal edits and presents a favorable market development. However, the risk counterfactual scenario does not clearly reflect an adverse market development, which prevents it from scoring higher. The cohesion of both counterfactuals is satisfactory, but the lack of clear market implications limits the score. Overall, the evaluation yields a score of 8, indicating a generally good job in generating counterfactual scenarios. [/INST] [/INST] [/INST] [/INST] [/INST] [/INST] [/INST] [/INST]

"""
matches = re.findall(pattern, test_string)
generated_score = int(matches[0]) if matches else -1

print("Found Score: ", generated_score)
