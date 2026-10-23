from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

SCRIPT_RE = re.compile(
    r"(?:python(?:\s+-m)?|bash|sh)\s+((?:scripts|tools)/[A-Za-z0-9_./-]+\.(?:py|sh|bash))"
)


def test_all_workflow_direct_local_script_references_exist():
    missing = []
    for path in WORKFLOWS.glob("*"):
        if path.suffix not in {".yml", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8")
        for script in SCRIPT_RE.findall(text):
            if not (ROOT / script).is_file():
                missing.append(f"{path.relative_to(ROOT)} -> {script}")
    assert not missing, "workflow references missing local scripts: " + "; ".join(missing)

def test_historical_training_jobs_require_explicit_manual_opt_in():
    path = WORKFLOWS / "integrated-e2e-005.yml"
    workflow_text = path.read_text(encoding="utf-8")
    assert "if: ${{ github.event_name == 'workflow_dispatch'" in workflow_text
    assert "inputs.allow_new_fits == true" in workflow_text
    assert "default: false" in workflow_text
    assert "|| (github.event_name == 'push'" not in workflow_text
    assert "toJSON(github.event.commits)" not in workflow_text
    assert "if: \\${{" not in workflow_text
