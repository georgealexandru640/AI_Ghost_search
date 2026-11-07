"""Ghost Diagnostic: citeste codul si eroarea, apoi localizeaza problema."""

import json


ROLE = "diagnostic"
TOP_LEVEL_FIELDS = {"type", "location", "evidence"}
LOCATION_FIELDS = {"file", "line"}


                                                                                             
def format_prompt(case):
    return (
        f"COD:\n{case['initial_code']}"
        f"EROARE:\n{case['error']}\n"
        f"LOG:\n{case['log']}\n"
        "JSON:\n"
    )


                                                                
def format_target(case):
    return json.dumps(
        case["diagnosis"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


                                                                                                 
def parse_output(text):
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Iesirea nu este JSON valid.") from error
    if not isinstance(document, dict) or set(document) != TOP_LEVEL_FIELDS:
        raise ValueError("Campurile obligatorii sunt type, location si evidence.")
    if not isinstance(document["type"], str) or not document["type"].strip():
        raise ValueError("type trebuie sa fie text nevid.")
    if not isinstance(document["evidence"], str) or not document["evidence"].strip():
        raise ValueError("evidence trebuie sa fie text nevid.")
    location = document["location"]
    if not isinstance(location, dict) or set(location) != LOCATION_FIELDS:
        raise ValueError("location trebuie sa contina exact file si line.")
    if not isinstance(location["file"], str) or not location["file"].strip():
        raise ValueError("location.file trebuie sa fie text nevid.")
    if (
        not isinstance(location["line"], int)
        or isinstance(location["line"], bool)
        or location["line"] < 1
    ):
        raise ValueError("location.line trebuie sa fie un intreg pozitiv.")
    return document


                                                                                               
def training_pairs(cases):
    return [(format_prompt(case), format_target(case)) for case in cases]


                                                                                             
format_diagnostic_prompt = format_prompt
format_diagnostic_target = format_target
parse_diagnostic_output = parse_output


                                                                         
def main():
    from .pipeline import describe_role

    print(describe_role(ROLE))


if __name__ == "__main__":
    main()
