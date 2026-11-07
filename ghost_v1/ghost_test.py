"""Ghost Test: propune teste care verifica daca reparatia chiar functioneaza."""

import json


ROLE = "ghost_test"


                                                                           
def format_prompt(case):
    patch = json.dumps(case["patch"], ensure_ascii=False, separators=(",", ":"))
    return (
        f"COD INITIAL:\n{case['initial_code']}"
        f"EROARE:{case['error']}\n"
        f"PATCH:{patch}\n"
        "PROPUNE TESTE SUPLIMENTARE JSON:\n"
    )


                                                                              
def expected_tests(case):
    return {"tests": [{"description": text} for text in case["verdict"]["tests"]]}


                                                                   
def format_target(case):
    return json.dumps(
        expected_tests(case), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


                                                                                 
def parse_output(text):
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Iesirea Ghost Test nu este JSON valid.") from error
    if not isinstance(document, dict) or set(document) != {"tests"}:
        raise ValueError("Iesirea trebuie sa contina exact campul tests.")
    tests = document["tests"]
    if not isinstance(tests, list) or not tests:
        raise ValueError("tests trebuie sa fie o lista nevida.")
    for test in tests:
        if not isinstance(test, dict) or set(test) != {"description"}:
            raise ValueError("Fiecare test trebuie sa aiba exact description.")
        if not isinstance(test["description"], str) or not test["description"].strip():
            raise ValueError("description trebuie sa fie text nevid.")
    return document


                                                             
def training_pairs(cases):
    return [(format_prompt(case), format_target(case)) for case in cases]


format_test_prompt = format_prompt
format_test_target = format_target
parse_test_output = parse_output


                                                                       
def main():
    from .pipeline import describe_role

    print(describe_role(ROLE))


if __name__ == "__main__":
    main()
