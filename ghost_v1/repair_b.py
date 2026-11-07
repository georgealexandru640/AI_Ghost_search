"""Ghost Repair B: cauta independent varianta de reparatie cea mai sigura."""

from .repair_a import (
    expected_repair,
    format_prompt as _format_prompt,
    format_target as _format_target,
    parse_output,
    patch_has_valid_python,
    patch_is_exact,
    target_is_allowed,
)


ROLE = "repair_b"
STRATEGY = "B"


                                                                                             
def format_prompt(case):
    return _format_prompt(case, STRATEGY)


                                                                             
def format_target(case):
    return _format_target(case, STRATEGY)


                                                           
def training_pairs(cases):
    return [(format_prompt(case), format_target(case)) for case in cases]


                                                                     
def main():
    from .pipeline import describe_role

    print(describe_role(ROLE))


if __name__ == "__main__":
    main()
