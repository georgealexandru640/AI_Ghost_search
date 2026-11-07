"""Legatura comuna dintre modelele Ghost V1.

Fisierul incarca datele, tokenizerul si greutatile, antreneaza un rol la cerere,
ruleaza inferenta si pastreaza regulile de siguranta. Astfel, cele sapte fisiere
de model raman simple si nu repeta acelasi cod tehnic.
"""

import argparse
import asyncio
import importlib
import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

import torch
from torch.utils.data import DataLoader, Dataset

from .tinyghost import GhostBPETokenizer, TinyGhost, TinyGhostConfig, generate_text


ROOT = Path(__file__).resolve().parent
PROJECT_PATH = ROOT / "project.json"
MODELS_PATH = ROOT / "models.pt"

ROLE_ALIASES = {
    "tinyghost": "base",
    "repair-a": "repair_a",
    "repair-b": "repair_b",
    "test": "ghost_test",
    "critic": "ghost_critic",
    "coordinator": "ghost_coordinator",
}
ROLE_MODULES = {
    "diagnostic": "diagnostic",
    "repair_a": "repair_a",
    "repair_b": "repair_b",
    "ghost_test": "ghost_test",
    "ghost_critic": "critic",
    "ghost_coordinator": "coordinator",
}
ROLE_LABELS = {
    "base": "TinyGhost",
    "diagnostic": "Ghost Diagnostic",
    "repair_a": "Ghost Repair A",
    "repair_b": "Ghost Repair B",
    "ghost_test": "Ghost Test",
    "ghost_critic": "Ghost Critic",
    "ghost_coordinator": "Ghost Coordinator",
}


                                                                                                           
def normalize_role(role):
    role = ROLE_ALIASES.get(role, role)
    if role not in ROLE_LABELS:
        raise ValueError(f"Rol necunoscut: {role}")
    return role


                                                                     
def load_project():
    return json.loads(PROJECT_PATH.read_text(encoding="utf-8"))


                                                                         
def load_tokenizer(project=None):
    project = project or load_project()
    return GhostBPETokenizer.from_document(project["tokenizer"])


                                                                                   
def load_cases(split=None, project=None):
    project = project or load_project()
    if split is None:
        return list(project["cases"].values())
    if split not in project["splits"]:
        raise ValueError(f"Split necunoscut: {split}")
    return [project["cases"][case_id] for case_id in project["splits"][split]]


                                                                                       
def load_model_bundle():
    bundle = torch.load(MODELS_PATH, map_location="cpu", weights_only=False)
    if bundle.get("format_version") != 1 or set(bundle.get("models", {})) != set(ROLE_LABELS):
        raise ValueError("models.pt nu contine exact cele sapte modele Ghost V1.")
    return bundle


                                                                                              
def load_role(role, device="cpu"):
    role = normalize_role(role)
    project = load_project()
    tokenizer = load_tokenizer(project)
    checkpoint = load_model_bundle()["models"][role]
    if checkpoint["tokenizer_hash"] != tokenizer.fingerprint:
        raise ValueError("Checkpointul si tokenizerul nu sunt compatibile.")
    config = TinyGhostConfig.from_dict(checkpoint["model_config"])
    model = TinyGhost(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, checkpoint


                                                                        
def role_module(role):
    role = normalize_role(role)
    if role == "base":
        raise ValueError("TinyGhost foloseste preantrenare, nu un contract de rol.")
    return importlib.import_module(f"{__package__}.{ROLE_MODULES[role]}")


                                                                                       
def describe_role(role):
    role = normalize_role(role)
    checkpoint = load_model_bundle()["models"][role]
    descriptions = {
        "base": "invata limbajul de baza si furnizeaza punctul de pornire",
        "diagnostic": "gaseste tipul si locul erorii",
        "repair_a": "propune patchul minim",
        "repair_b": "propune independent reparatia sigura",
        "ghost_test": "propune teste pentru patch",
        "ghost_critic": "evalueaza riscul si poate respinge",
        "ghost_coordinator": "compara candidatii si alege rezultatul final",
    }
    return (
        f"{ROLE_LABELS[role]}: {descriptions[role]}. "
        f"Checkpoint local, pasul {checkpoint.get('step', '?')}, "
        f"{sum(value.numel() for value in checkpoint['model_state_dict'].values()):,} valori invatate."
    )


                                                                                                                 
def validate_project():
    project = load_project()
    tokenizer = load_tokenizer(project)
    problems = []
    seen_ids = set()
    seen_groups = {}
    for split, ids in project["splits"].items():
        for case_id in ids:
            if case_id in seen_ids:
                problems.append(f"{case_id} apare in mai multe splituri")
            seen_ids.add(case_id)
            case = project["cases"].get(case_id)
            if case is None:
                problems.append(f"{case_id} lipseste din cases")
                continue
            group = (case["program"], case["template_id"])
            previous_split = seen_groups.setdefault(group, split)
            if previous_split != split:
                problems.append(f"grupul {group} apare in {previous_split} si {split}")
            required = {
                "id", "program", "template_id", "family", "initial_code", "error",
                "log", "diagnosis", "patch", "verdict", "provenance",
            }
            if set(case) != required:
                problems.append(f"{case_id} nu respecta schema")
    if seen_ids != set(project["cases"]):
        problems.append("unele cazuri nu apar in niciun split")
    bundle = load_model_bundle()
    for role, checkpoint in bundle["models"].items():
        if checkpoint["tokenizer_hash"] != tokenizer.fingerprint:
            problems.append(f"tokenizer incompatibil cu {role}")
    return {
        "ok": not problems,
        "problems": problems,
        "counts": {name: len(ids) for name, ids in project["splits"].items()},
        "models": list(bundle["models"]),
        "tokenizer_hash": tokenizer.fingerprint,
    }


                                                                                          
def infer(role, item, max_new_tokens=None):
    role = normalize_role(role)
    if role == "base":
        raise ValueError("Pentru TinyGhost foloseste generate_text direct.")
    module = role_module(role)
    model, tokenizer, checkpoint = load_role(role)
    prompt = module.format_prompt(item)
    limit = max_new_tokens or checkpoint.get("training_config", {}).get("max_new_tokens", 128)
    raw = generate_text(model, tokenizer, prompt, max_new_tokens=limit, temperature=0.0)
    return {"role": role, "prompt": prompt, "raw": raw, "parsed": module.parse_output(raw)}


@dataclass
class Message:
    sender: str
    recipient: str
    kind: str
    payload: dict
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MessageBus:
    """Cutia postala interna prin care modelele isi trimit rezultate."""

                                                                         
    def __init__(self):
        self.queues = {}
        self.history = []

                                                                                         
    async def send(self, message):
        self.history.append(message)
        await self.queues.setdefault(message.recipient, asyncio.Queue()).put(message)

                                                                       
    async def receive(self, recipient):
        return await self.queues.setdefault(recipient, asyncio.Queue()).get()


                                                                                          
def patch_is_safe(case, patch):
    from .repair_a import patch_has_valid_python, target_is_allowed

    path = PurePosixPath(patch.get("file", ""))
    path_allowed = bool(path.parts) and not path.is_absolute() and ".." not in path.parts
    return {
        "path_allowed": path_allowed,
        "target_allowed": path_allowed and target_is_allowed(case, patch),
        "python_valid": path_allowed and patch_has_valid_python(case, patch),
    }


                                                                                                               
def deterministic_demo(case_id="case-001", human_confirmed=False):
    """Arata fluxul folosind raspunsurile etichetate, fara a pretinde ca sunt inferenta."""
    project = load_project()
    case = project["cases"][case_id]
    from . import coordinator, critic, diagnostic, ghost_test, repair_a, repair_b

    diagnosis = diagnostic.parse_output(diagnostic.format_target(case))
    candidate_a = repair_a.parse_output(repair_a.format_target(case))
    candidate_b = repair_b.parse_output(repair_b.format_target(case))
    tests = ghost_test.parse_output(ghost_test.format_target(case))
    safety_a = patch_is_safe(case, candidate_a)
    safety_b = patch_is_safe(case, candidate_b)
    critic_example = critic.build_examples([case])[0]
    critic_result = critic.parse_output(critic.format_target(critic_example))
    coordinator_example = coordinator.build_examples([case])[0]
    decision = coordinator.parse_output(coordinator.format_target(coordinator_example))
    allowed = coordinator.final_selection_allowed(
        decision["decision"],
        {"qa": safety_a["python_valid"], "security": safety_a["target_allowed"]},
        {"qa": safety_b["python_valid"], "security": safety_b["target_allowed"]},
        human_confirmed,
    )
    return {
        "mode": "demonstratie cu raspunsurile corecte din dataset",
        "case_id": case_id,
        "diagnostic": diagnosis,
        "repair_a": candidate_a,
        "repair_b": candidate_b,
        "tests": tests,
        "safety": {"A": safety_a, "B": safety_b},
        "critic": critic_result,
        "coordinator": decision,
        "human_confirmed": human_confirmed,
        "apply_allowed": allowed,
    }


                                                                                  
def case_as_pretraining_text(case):
    return (
        f"<case id={case['id']}>\nCOD:\n{case['initial_code']}"
        f"EROARE:\n{case['error']}\nLOG:\n{case['log']}\n"
        f"DIAGNOSTIC:{json.dumps(case['diagnosis'], ensure_ascii=False)}\n"
        f"PATCH:{json.dumps(case['patch'], ensure_ascii=False)}\n</case>\n"
    )


class NextTokenDataset(Dataset):
                                                                                                
    def __init__(self, token_ids, context_length):
        self.examples = []
        for start in range(0, max(0, len(token_ids) - 1), context_length):
            chunk = token_ids[start : start + context_length + 1]
            if len(chunk) > 1:
                self.examples.append(
                    (torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long))
                )

                                                                  
    def __len__(self):
        return len(self.examples)

                                                       
    def __getitem__(self, index):
        return self.examples[index]


class RoleDataset(Dataset):
                                                                                                        
    def __init__(self, pairs, tokenizer, context_length):
        self.examples = []
        eos = tokenizer.special_tokens["<|eos|>"]
        for prompt, target in pairs:
            prompt_ids = tokenizer.encode(prompt, add_bos=True, role="user")
            answer_ids = tokenizer.encode(target, role="assistant") + [eos]
            combined = (prompt_ids + answer_ids)[-context_length:]
            prompt_kept = max(0, len(combined) - len(answer_ids))
            inputs = torch.tensor(combined[:-1], dtype=torch.long)
            targets = torch.tensor(combined[1:], dtype=torch.long)
            targets[: max(0, prompt_kept - 1)] = -100
            self.examples.append((inputs, targets))

                                                 
    def __len__(self):
        return len(self.examples)

                                                              
    def __getitem__(self, index):
        return self.examples[index]


                                                                                                   
def _collate(batch):
    max_length = max(len(inputs) for inputs, _ in batch)
    input_batch = torch.zeros((len(batch), max_length), dtype=torch.long)
    target_batch = torch.full((len(batch), max_length), -100, dtype=torch.long)
    for row, (inputs, targets) in enumerate(batch):
        input_batch[row, : len(inputs)] = inputs
        target_batch[row, : len(targets)] = targets
    return input_batch, target_batch


                                                                  
def _training_config(role, project):
    if role == "base":
        return project["configs"]["base"]
    if role == "diagnostic":
        return project["configs"]["diagnostic"]
    return project["configs"]["roles"][role]


                                                                                                         
def train(role, max_steps=None):
    """Continua local antrenarea rolului si actualizeaza numai intrarea lui din models.pt."""
    role = normalize_role(role)
    project = load_project()
    tokenizer = load_tokenizer(project)
    bundle = load_model_bundle()
    checkpoint = bundle["models"][role]
    config = TinyGhostConfig.from_dict(checkpoint["model_config"])
    model = TinyGhost(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    settings = _training_config(role, project)
    training = settings["training"]
    steps = int(max_steps or training["max_steps"])
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])

    if role == "base":
        text = "".join(case_as_pretraining_text(case) for case in load_cases("train", project))
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        dataset = NextTokenDataset(ids, config.context_length)
    else:
        pairs = role_module(role).training_pairs(load_cases("train", project))
        dataset = RoleDataset(pairs, tokenizer, config.context_length)
    if not len(dataset):
        raise ValueError("Datasetul nu contine exemple antrenabile.")

    loader = DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=True,
        collate_fn=_collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"])
    model.train()
    iterator = iter(loader)
    losses = []
    for _ in range(steps):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, targets = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
        optimizer.step()
        losses.append(float(loss.detach()))

    updated = deepcopy(checkpoint)
    updated["model_state_dict"] = model.state_dict()
    updated["optimizer_state_dict"] = optimizer.state_dict()
    updated["step"] = int(checkpoint.get("step", 0)) + steps
    updated["training_config"] = training
    bundle["models"][role] = updated
    temporary = MODELS_PATH.with_suffix(".tmp")
    torch.save(bundle, temporary)
    temporary.replace(MODELS_PATH)
    return {"role": role, "steps_added": steps, "final_loss": losses[-1]}


                                                                           
def parse_args():
    parser = argparse.ArgumentParser(description="Ghost V1 compact")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("info")
    subparsers.add_parser("validate")
    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--case", default="case-001")
    demo_parser.add_argument("--confirm", action="store_true")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("role", choices=sorted(ROLE_LABELS))
    train_parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


                                                                                
def main():
    args = parse_args()
    if args.command == "info":
        result = {role: describe_role(role) for role in ROLE_LABELS}
    elif args.command == "validate":
        result = validate_project()
    elif args.command == "demo":
        result = deterministic_demo(args.case, args.confirm)
    else:
        result = train(args.role, args.max_steps)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
