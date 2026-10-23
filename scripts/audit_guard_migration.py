"""Recognize a no-fit adapter migration; never qualify new model training.

This narrow CI lane proves model/runner ASTs differ only by guarded optimizer,
step and checkpoint adapters. Other changes fall back to prospective study gates.
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "166d60d8d897f1f8be5f48410e7d90b8af46048c"
INFRASTRUCTURE = {
    "scripts/validate_research_parity.py",
    "scripts/audit_training_budget.py",
    "scripts/guarded_research.py",
    "scripts/verify_frozen_replay.py",
    "scripts/replay_frozen_research.py",
    "scripts/audit_guard_migration.py",
    "src/tac_osm/parameter_budget.py",
}


class StripAdapters(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == "tac_osm.parameter_budget":
            return None
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id == "budgeted_optimizer":
                if len(node.args) < 2:
                    raise ValueError("whole-model optimizer adapter required")
                return ast.Call(
                    func=node.args[1], args=node.args[2:], keywords=node.keywords
                )
            if node.func.id == "checked_optimizer_step":
                return ast.Call(
                    func=ast.Attribute(value=node.args[0], attr="step", ctx=ast.Load()),
                    args=node.args[1:],
                    keywords=node.keywords,
                )
            if node.func.id == "save_training_checkpoint":
                node.func = ast.Attribute(
                    value=ast.Name(id="torch", ctx=ast.Load()),
                    attr="save",
                    ctx=ast.Load(),
                )
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "save"
            and ast.unparse(node.func.value) == "self.torch"
        ):
            node.func.value = ast.Name(id="torch", ctx=ast.Load())
        return node


def normalized(source):
    # The frozen ME066 runner contains this syntax defect. Only repair its
    # literal newline transcription for comparison; no dynamics are changed.
    source = source.replace(
        '}:\\n        raise RuntimeError("four registered architectures required; no added or dropped arm")\\n    manifest=',
        '}:\n        raise RuntimeError("four registered architectures required; no added or dropped arm")\n    manifest=',
    )
    return ast.dump(StripAdapters().visit(ast.parse(source)), include_attributes=False)


def verify_migration(base_head, root=ROOT):
    paths = subprocess.check_output(
        ["git", "diff", "--name-only", base_head], cwd=root, text=True
    ).splitlines()
    verified = []
    for rel in paths:
        if (
            rel.startswith("research/")
            and rel != "research/REAL-WORLD-EVIDENCE-RULES.md"
            and not rel.startswith("research/governance/")
        ):
            raise ValueError("scientific evidence/registration changed: " + rel)
        if rel.startswith(("scripts/", "src/")) and rel not in INFRASTRUCTURE:
            original = subprocess.run(
                ["git", "show", SNAPSHOT + ":" + rel],
                cwd=root,
                capture_output=True,
                text=True,
            )
            if original.returncode or normalized(original.stdout) != normalized(
                (root / rel).read_text()
            ):
                raise ValueError("model mathematics or training recipe changed: " + rel)
            verified.append(rel)
        elif not (
            rel in INFRASTRUCTURE
            or rel
            in (
                "AGENTS.md",
                "docs/TACOSM-BASELINE-001.md",
                "research/REAL-WORLD-EVIDENCE-RULES.md",
            )
            or rel.startswith(("tests/", ".github/workflows/", "research/governance/"))
        ):
            raise ValueError("outside no-fit enforcement migration: " + rel)
    return {
        "status": "PASS_NO_FIT_ADAPTER_MIGRATION",
        "snapshot": SNAPSHOT,
        "verified_unchanged_model_or_runner_ASTs": verified,
        "training_authorized": False,
        "scope": "CI source migration only; supported optimizer updates still require universal live gates",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    args = p.parse_args()
    try:
        result = verify_migration(args.base)
    except (ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        p.exit(2, str(error) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
