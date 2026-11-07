"""Ghost Repair A: propune cea mai mica modificare care repara eroarea."""

import json


ROLE = "repair_a"
STRATEGY = "A"
FIELDS = {"file", "line", "old_line", "new_lines", "explanation", "provenance"}


                                                                          
def expected_repair(case, strategy=STRATEGY):
    patch = case["patch"]
    return {
        "file": patch["file"],
        "line": patch["line"],
        "old_line": patch["old_line"],
        "new_lines": patch["new_lines"],
        "explanation": f"Corecteaza {case['diagnosis']['type']} la linia indicata.",
        "provenance": f"ghost-repair-{strategy.lower()}",
    }


                                                                   
def format_prompt(case, strategy=STRATEGY):
    diagnosis = json.dumps(
        {"type": case["diagnosis"]["type"], "location": case["diagnosis"]["location"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if strategy == "A":
        return (
            f"DIAGNOSTIC:{diagnosis}\n"
            f"COD:\n{case['initial_code']}"
            "PROPUNE PATCH MINIM JSON:\n"
        )
    if strategy == "B":
        return (
            f"COD DEFECT:\n{case['initial_code']}"
            f"EROARE:{case['error']}\n"
            f"DOVEZI:{case['diagnosis']['evidence']}\n"
            "ALEGE REPARATIA SIGURA JSON:\n"
        )
    raise ValueError(f"Strategie necunoscuta: {strategy}")


                                                           
def format_target(case, strategy=STRATEGY):
    return json.dumps(
        expected_repair(case, strategy),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


                                                                                  
def parse_output(text):
    try:
        patch = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Patchul nu este JSON valid.") from error
    if not isinstance(patch, dict) or set(patch) != FIELDS:
        raise ValueError("Patchul nu are exact campurile contractului.")
    for field in ("file", "old_line", "explanation", "provenance"):
        if not isinstance(patch[field], str) or not patch[field].strip():
            raise ValueError(f"{field} trebuie sa fie text nevid.")
    if not isinstance(patch["line"], int) or isinstance(patch["line"], bool) or patch["line"] < 1:
        raise ValueError("line trebuie sa fie un intreg pozitiv.")
    if not isinstance(patch["new_lines"], list) or not patch["new_lines"]:
        raise ValueError("new_lines trebuie sa fie o lista nevida.")
    if any(not isinstance(line, str) for line in patch["new_lines"]):
        raise ValueError("Fiecare linie noua trebuie sa fie text.")
    return patch


                                                                           
def target_is_allowed(case, patch):
    expected = case["patch"]
    return patch["file"] == expected["file"] and patch["line"] == expected["line"]


                                                                                                         
def patch_has_valid_python(case, patch):
    lines = case["initial_code"].splitlines()
    index = patch["line"] - 1
    if index >= len(lines) or lines[index].strip() != patch["old_line"].strip():
        return False
    candidate = lines[:index] + patch["new_lines"] + lines[index + 1 :]
    try:
        compile("\n".join(candidate) + "\n", patch["file"], "exec")
    except SyntaxError:
        return False
    return True


                                                                                       
def patch_is_exact(case, patch):
    expected = case["patch"]
    return all(
        patch[field] == expected[field]
        for field in ("file", "line", "old_line", "new_lines")
    )


                                                           
def training_pairs(cases):
    return [(format_prompt(case), format_target(case)) for case in cases]


                                                                     
def main():
    from .pipeline import describe_role

    print(describe_role(ROLE))


if __name__ == "__main__":
    main()
