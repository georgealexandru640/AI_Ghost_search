"""Toate testele Ghost V1, tinute intr-un singur fisier ca proiectul sa fie simplu."""

import torch

from . import coordinator, critic, diagnostic, ghost_test, repair_a, repair_b
from .pipeline import (
    ROLE_LABELS,
    deterministic_demo,
    load_cases,
    load_model_bundle,
    load_tokenizer,
    patch_is_safe,
    validate_project,
)
from .tinyghost import (
    GhostBPETokenizer,
    TinyGhost,
    TinyGhostConfig,
    build_causal_mask,
)


                                                                              
def test_projectul_are_datele_si_modelele_corecte():
    result = validate_project()
    assert result["ok"], result["problems"]
    assert result["counts"] == {"train": 10, "dev": 4, "holdout": 30}
    assert set(result["models"]) == set(ROLE_LABELS)


                                                                    
def test_tokenizerul_pastreaza_textul_si_hashul_original():
    tokenizer = load_tokenizer()
    text = "print('Ghost știe Python')\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.fingerprint == "07ac51f0f9eb003e09df61bfe02c1160b1a9653fd6c6260314b787cf73a1cbc2"


                                                                                         
def test_tokenizer_nou_se_poate_antrena_local():
    tokenizer = GhostBPETokenizer().train(["ana are mere", "ana are pere"], max_merges=8)
    assert tokenizer.decode(tokenizer.encode("ana are mere")) == "ana are mere"


                                                   
def test_atentia_nu_poate_vedea_tokenii_viitori():
    mask = build_causal_mask(4)
    assert mask.tolist() == [
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False],
    ]


                                                                                  
def test_modelul_mic_face_forward_si_backward():
    model = TinyGhost(TinyGhostConfig(vocab_size=32, context_length=8, embed_dim=16, num_heads=2, num_layers=1))
    token_ids = torch.randint(0, 32, (2, 6), dtype=torch.long)
    logits, loss = model(token_ids, token_ids)
    assert logits.shape == (2, 6, 32)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


                                                                                      
def test_toate_checkpointurile_se_potrivesc_cu_arhitectura():
    bundle = load_model_bundle()
    for role, checkpoint in bundle["models"].items():
        model = TinyGhost(TinyGhostConfig.from_dict(checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        assert model.config.vocab_size == 326, role


                                                                               
def test_contractele_celor_sase_roluri_specializate():
    case = load_cases("train")[0]
    assert diagnostic.parse_output(diagnostic.format_target(case))["type"]
    assert repair_a.parse_output(repair_a.format_target(case))["provenance"] == "ghost-repair-a"
    assert repair_b.parse_output(repair_b.format_target(case))["provenance"] == "ghost-repair-b"
    assert ghost_test.parse_output(ghost_test.format_target(case))["tests"]
    critic_example = critic.build_examples([case])[0]
    assert critic.parse_output(critic.format_target(critic_example))["recommendation"] == "accept"
    coordinator_example = coordinator.build_examples([case])[0]
    assert coordinator.parse_output(coordinator.format_target(coordinator_example))["decision"] == "A"


                                                                    
def test_patchul_corect_trece_verificarile_deterministe():
    case = load_cases("train")[0]
    patch = repair_a.expected_repair(case)
    assert patch_is_safe(case, patch) == {
        "path_allowed": True,
        "target_allowed": True,
        "python_valid": True,
    }


                                                                            
def test_confirmarea_umana_ramane_obligatorie():
    without_confirmation = deterministic_demo("case-001", human_confirmed=False)
    with_confirmation = deterministic_demo("case-001", human_confirmed=True)
    assert without_confirmation["apply_allowed"] is False
    assert with_confirmation["apply_allowed"] is True
