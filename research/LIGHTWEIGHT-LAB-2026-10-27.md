# TAC-OSM Lightweight Research Lab (2026-10-27)

**Default: GitHub + your existing Python research scripts + GitHub Actions.**
No account, daemon, database, subscription, model-framework dependency, GPU
cluster, or vendor account is needed. Original checkpoints and experiment
scripts remain unchanged. This is an evidence INDEX, not an evaluator.

## Minimal daily workflow

Run an existing authorized script exactly as you currently do. After it
produces its own report, index the report:

    python scripts/lab.py record \
      --run-id example-seed-101 --experiment MY-EXPERIMENT \
      --variant "System 1" --stage analysis \
      --evidence historical_unverified \
      --source results/my-existing-report.json \
      --metric loss=0.12 --param model_parameters=100000 --seed 101

These values are illustrative, NOT results from an actual experiment.

Then view the run ledger and offline dashboard:

    python scripts/lab.py status
    python scripts/lab.py report
    # open runs/lab/dashboard.html in a browser

One immutable receipt per ID is saved as runs/lab/RUN-ID/receipt.json.
Repeated identical commands are safe, but changing evidence under an
existing ID is rejected. The ledger records a Git revision, hashes of the
original source report and optional dataset/checkpoint files, exact
parameters supplied, and finite numeric metrics. The ledger never copies
large data, reads private episode contents for claims, or alters training.

## All category choices, simple default

| Need | Available by default | Optional provider choice |
|---|---|---|
| Experiments, metrics, curves | Run JSON receipts and offline HTML | MLflow, W&B, Comet |
| Dataset and checkpoint lineage | Streaming SHA256 and run/source linkage | DVC or private Hugging Face Hub |
| Run scheduling | Existing Python + GitHub Actions | Local Docker, Slurm, Ray, Kubernetes plans |
| Exploratory work | Read-only example notebook | JupyterLab or Colab |
| Collaboration and lab knowledge | GitHub Issues, PRs and research Markdown | Notion, Confluence, Slack, Discord |

The optional providers are deliberately NOT installed, connected, enabled,
paid for or deployed. There is no default data upload.

## Read-only setup recipes for every provider

    python scripts/lab.py plan --provider mlflow
    python scripts/lab.py plan --provider wandb
    python scripts/lab.py plan --provider comet
    python scripts/lab.py plan --provider dvc
    python scripts/lab.py plan --provider huggingface
    python scripts/lab.py plan --provider docker
    python scripts/lab.py plan --provider slurm --gpus 0 --cpus 2
    python scripts/lab.py plan --provider ray
    python scripts/lab.py plan --provider kubernetes
    python scripts/lab.py plan --provider github
    python scripts/lab.py plan --provider jupyter
    python scripts/lab.py plan --provider colab
    python scripts/lab.py plan --provider notion
    python scripts/lab.py plan --provider confluence
    python scripts/lab.py plan --provider slack
    python scripts/lab.py plan --provider discord

Each recipe explicitly reports enabled=false, executed=false, and
external_data_sent=false. The plans describe safe configuration, not
actual cloud access. Example scalars-only MLflow mapping:

    python scripts/lab.py plan --provider mlflow \
      --receipt runs/lab/example-seed-101/receipt.json

To actually enable one of these remote services later requires a separately
authorized installation and credentials, data-license/privacy review,
operator-approved compute budget and trust boundaries. For large file
sharing, use DVC or private Hugging Face only when publisher rights allow.
For Slack or Discord notifications, prefer native GitHub notifications
until messaging integrations have an actual team.

## Automatic CI and notebooks

Existing research-observatory and anti-cheat GitHub Actions now record the
constructed fixture result AFTER their existing no-fit tests and archive
both the per-run receipt and offline HTML summary beside the existing
JUnit and JSON evidence. No extra training or external service is started.
Notebook: notebooks/research_lab_quicklook.ipynb opens local receipts
read-only, without package dependencies or uploading protected data.

An experiment issue template under .github/ISSUE_TEMPLATE tracks
hypotheses, primary authentic datasets and their licences, original
splits, independent labels, fair comparators, checkpoint replay, the PNDS
mathematics link, and the scientific promotion gates.

## Boundaries that never change

Every lab receipt has scientific_qualification=NOT_EVALUATED_BY_LAB,
capability_claim_authorized=false, comparison_parity_verified=false,
and reward_integrity_verified=false. Logging a 0.99 score does not
make it true or establish generalization. Actual experiments must still
pass the real-data/source rules, fair parameter/data/compute controls,
checkpoints and replay, prospective reward-hacking gates, and, where
applicable, PNDS Python and Lean formalization.

The existing .gitignore ignores runs/ and results/ so raw outputs, private
data and large checkpoints are not silently committed. This means local
lab receipts are NOT an offsite backup. Use approved access-controlled
artifact storage and proper run/checkpoint retention when needed.

**Recommended upgrade order:** keep GitHub first. Add MLflow if offline
search/plotting becomes insufficient. Add DVC/HF when large artifact
versioning matters. Move to Slurm/Ray/Kubernetes only after shared compute
contention becomes real, not to look more professional.

## References

MLflow https://mlflow.org/docs/latest/ml/tracking/
W&B https://docs.wandb.ai/
Comet https://www.comet.com/docs/
DVC https://dvc.org/doc
Hugging Face https://huggingface.co/docs/hub/
Docker https://docs.docker.com/
Ray https://docs.ray.io/
Slurm https://slurm.schedmd.com/documentation.html
Kubernetes https://kubernetes.io/docs/
