"""Ghost Coordinator: compara candidatii si alege ordinea finala."""

import json


ROLE = "ghost_coordinator"
FIELDS = {"decision", "ranking", "reason"}


                                                                                                                                       
def build_examples(cases):
    examples = []
    for case in cases:
        good = {
            "file": case["patch"]["file"],
            "line": case["patch"]["line"],
            "qa": True,
            "security": True,
        }
        bad = {
            "file": "../outside_project.py",
            "line": case["patch"]["line"],
            "qa": False,
            "security": False,
        }
        scenarios = [
            ("a_good", good, bad, "A", ["A", "B"], "A trece QA si Security; B este respins."),
            ("b_good", bad, good, "B", ["B", "A"], "B trece QA si Security; A este respins."),
            ("both_bad", bad, bad, "reject_all", [], "Ambii candidati sunt respinsi de QA si Security."),
        ]
        for suffix, candidate_a, candidate_b, decision, ranking, reason in scenarios:
            examples.append(
                {
                    "id": f"{case['id']}-{suffix}",
                    "case_id": case["id"],
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                    "expected": {"decision": decision, "ranking": ranking, "reason": reason},
                }
            )
    return examples


                                                                                        
def format_prompt(example):
    candidate_a = json.dumps(example["candidate_a"], ensure_ascii=False, separators=(",", ":"))
    candidate_b = json.dumps(example["candidate_b"], ensure_ascii=False, separators=(",", ":"))
    return (
        f"CAZ:{example['case_id']}\nA:{candidate_a}\nB:{candidate_b}\n"
        "ORDONEAZA SAU RESPINGE JSON:\n"
    )


                                                                           
def format_target(example):
    return json.dumps(
        example["expected"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


                                                                                                                     
def parse_output(text):
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Iesirea Coordinator nu este JSON valid.") from error
    if not isinstance(document, dict) or set(document) != FIELDS:
        raise ValueError("Coordinator trebuie sa produca exact decision, ranking si reason.")
    if document["decision"] not in {"A", "B", "reject_all"}:
        raise ValueError("decision trebuie sa fie A, B sau reject_all.")
    if not isinstance(document["ranking"], list):
        raise ValueError("ranking trebuie sa fie lista.")
    if any(item not in {"A", "B"} for item in document["ranking"]):
        raise ValueError("ranking poate contine numai A si B.")
    if len(set(document["ranking"])) != len(document["ranking"]):
        raise ValueError("ranking nu poate contine duplicate.")
    if document["decision"] == "reject_all" and document["ranking"]:
        raise ValueError("reject_all cere ranking gol.")
    if document["decision"] in {"A", "B"} and (
        not document["ranking"] or document["ranking"][0] != document["decision"]
    ):
        raise ValueError("Primul element din ranking trebuie sa fie decizia.")
    if not isinstance(document["reason"], str) or not document["reason"].strip():
        raise ValueError("reason trebuie sa fie text nevid.")
    return document


                                                                                               
def final_selection_allowed(decision, candidate_a, candidate_b, human_confirmed):
    if decision == "reject_all" or not human_confirmed:
        return False
    selected = candidate_a if decision == "A" else candidate_b
    return selected["qa"] and selected["security"]


                                                                                                  
def training_pairs(cases):
    return [(format_prompt(item), format_target(item)) for item in build_examples(cases)]


build_coordinator_examples = build_examples
format_coordinator_prompt = format_prompt
format_coordinator_target = format_target
parse_coordinator_output = parse_output


                                                                                        
def main():
    from .pipeline import describe_role

    print(describe_role(ROLE))


if __name__ == "__main__":
    main()
