"""Ghost Critic: evalueaza riscul si poate recomanda acceptarea sau respingerea."""

import json


ROLE = "ghost_critic"
FIELDS = {"risk", "recommendation", "reason"}


                                                                                                               
def build_examples(cases):
    examples = []
    for case in cases:
        positive_patch = case["patch"]
        negative_patch = {
            **case["patch"],
            "file": "../outside_project.py",
            "new_lines": ["if broken"],
        }
        examples.extend(
            [
                {
                    "id": f"{case['id']}-positive",
                    "case": case,
                    "candidate": positive_patch,
                    "qa_passed": True,
                    "security_passed": True,
                    "expected": {
                        "risk": "low",
                        "recommendation": "accept",
                        "reason": "Candidatul are tinta permisa si a trecut QA si Security.",
                    },
                },
                {
                    "id": f"{case['id']}-negative",
                    "case": case,
                    "candidate": negative_patch,
                    "qa_passed": False,
                    "security_passed": False,
                    "expected": {
                        "risk": "high",
                        "recommendation": "reject",
                        "reason": "Candidatul tinteste o cale interzisa si nu trece verificarile.",
                    },
                },
            ]
        )
    return examples


                                                                                             
def format_prompt(example):
    case = example["case"]
    candidate = {
        key: example["candidate"][key]
        for key in ("file", "line", "old_line", "new_lines")
    }
    return (
        f"TIP:{case['diagnosis']['type']}\n"
        f"LOCATIE:{case['diagnosis']['location']['file']}:{case['diagnosis']['location']['line']}\n"
        f"CANDIDAT:{json.dumps(candidate, ensure_ascii=False, separators=(',', ':'))}\n"
        f"QA:{'PASS' if example['qa_passed'] else 'FAIL'}\n"
        f"SECURITY:{'PASS' if example['security_passed'] else 'FAIL'}\n"
        "RISC SI RECOMANDARE JSON:\n"
    )


                                                                           
def format_target(example):
    return json.dumps(
        example["expected"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


                                                                                                
def parse_output(text):
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Iesirea Critic nu este JSON valid.") from error
    if not isinstance(document, dict) or set(document) != FIELDS:
        raise ValueError("Critic trebuie sa produca exact risk, recommendation si reason.")
    if document["risk"] not in {"low", "high"}:
        raise ValueError("risk trebuie sa fie low sau high.")
    if document["recommendation"] not in {"accept", "reject"}:
        raise ValueError("recommendation trebuie sa fie accept sau reject.")
    if not isinstance(document["reason"], str) or not document["reason"].strip():
        raise ValueError("reason trebuie sa fie text nevid.")
    return document


                                                                                            
def final_gate_allows(critic_output, qa_passed, security_passed, human_confirmed):
    return (
        critic_output["recommendation"] == "accept"
        and qa_passed
        and security_passed
        and human_confirmed
    )


                                                         
def training_pairs(cases):
    return [(format_prompt(item), format_target(item)) for item in build_examples(cases)]


build_critic_examples = build_examples
format_critic_prompt = format_prompt
format_critic_target = format_target
parse_critic_output = parse_output


                                                                   
def main():
    from .pipeline import describe_role

    print(describe_role(ROLE))


if __name__ == "__main__":
    main()
