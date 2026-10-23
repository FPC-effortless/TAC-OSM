"""CI coverage audit: supported PyTorch fitting must use shared guard adapters."""

import ast
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
STEP_NAME = re.compile(r"^(?:opt\d*|optimizer\d*|.*_optimizer)$")


def dotted(node, aliases):
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return dotted(node.value, aliases) + "." + node.attr
    return ""


def inspect_file(path):
    tree = ast.parse(path.read_text())
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for x in node.names:
                aliases[x.asname or x.name.split(".")[0]] = (
                    x.name if x.asname else x.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            for x in node.names:
                aliases[x.asname or x.name] = (node.module or "") + "." + x.name
    optimizer_names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and dotted(node.value.func, aliases).endswith("budgeted_optimizer")
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            optimizer_names.update(dotted(x, aliases) for x in targets)
    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func, aliases)
        raw_optimizer = name.startswith("torch.optim.") and not any(
            name.startswith(p)
            for p in ("torch.optim.lr_scheduler.", "torch.optim.swa_utils.")
        )
        dynamic = (
            isinstance(node.func, ast.Call)
            and dotted(node.func.func, aliases) == "getattr"
            and node.func.args
            and dotted(node.func.args[0], aliases) == "torch.optim"
        )
        if raw_optimizer or dynamic:
            errors.append(
                (
                    node.lineno,
                    "raw optimizer constructor; use budgeted_optimizer(whole_model, optimizer_type, ...)",
                )
            )
        if name in ("torch.save", "self.torch.save"):
            errors.append(
                (
                    node.lineno,
                    "raw checkpoint serialization; use save_training_checkpoint",
                )
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "step":
            owner = dotted(node.func.value, aliases)
            if owner in optimizer_names or STEP_NAME.fullmatch(
                owner.rsplit(".", 1)[-1]
            ):
                errors.append(
                    (
                        node.lineno,
                        "unguarded optimizer step; use checked_optimizer_step",
                    )
                )
    return errors


def audit(root=ROOT):
    failures = []
    files = []
    for folder in ("scripts", "src"):
        for path in sorted((root / folder).rglob("*.py")):
            rel = path.relative_to(root)
            if str(rel) in (
                "src/tac_osm/parameter_budget.py",
                "scripts/audit_training_budget.py",
            ):
                continue
            errors = inspect_file(path)
            files.append(str(rel))
            failures.extend(f"{rel}:{line}: {message}" for line, message in errors)
    return files, failures


if __name__ == "__main__":
    import json

    files, failures = audit()
    if failures:
        print("\n".join(failures))
        sys.exit(2)
    print(
        json.dumps(
            {
                "status": "PASS",
                "audited_Python_files": len(files),
                "scope": "supported PyTorch constructors, optimizer updates and checkpoint writes in scripts/src; software fixtures excluded",
            }
        )
    )
