# Mechanisms, sources and scope

Sources checked on 17 October 2026. These references motivate experiments;
their results are not evidence of TAC-OSM performance.

| Source | Transfer tested here | Boundary |
| --- | --- | --- |
| [Retinal dendritic calcium computation](https://www.nature.com/articles/nature00931), Euler et al., 2002; [space–time wiring](https://www.nature.com/articles/nature13240), 2014 | Local temporal-order encoder, DEV-047A | Neural event ordering is an analogy, not a reconstruction of retinal circuitry. |
| [A heritable iron memory enables decision-making in E. coli](https://pubmed.ncbi.nlm.nih.gov/37988472/), 2023, DOI 10.1073/pnas.2309082120 | Compact persistent multi-timescale state | Heritable intracellular iron effects do not imply a small neural state is sufficient for maintenance. |
| [Mamba](https://arxiv.org/abs/2312.00752), Gu and Dao | Input-dependent retention control in a diagonal recurrent control | No optimized Mamba implementation, hardware-throughput result or guaranteed causal memory is claimed. |
| [ActionOPD authors' project](https://thu-nics.github.io/ActionOPD/), arXiv 2610.19820 | Supervise consequences at one-step student-encountered intervention/inspection states, DEV-046B | Cost supervision in a deterministic simulator differs from black-box language-agent semantic-action distillation. |
| [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM/blob/main/README.md) | Cost-sensitive hard-negative ranking and state/action prediction | Reusable state/action representations motivate the interface; this is not a CLM model reproduction or speed comparison. |
| [GEPA, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e9e708b6f48e14fd0ac29e167413f76-Abstract-Conference.html) | Development-only trace archive and candidate/Pareto control | LLM reflection is not run; sealed outcomes must never guide candidate revision. |
| [Predictive maintenance using digital twins: a systematic literature review](https://doi.org/10.1016/j.infsof.2022.107008), 2022 | Independent intervention replay and outcome verification | A deterministic synthetic world is not an industrial digital twin; data variety, calibration and simulator validity need external evidence. |

The architectural thesis remains a research question: can a compact persistent
state preserve decision-relevant distinctions under interventions more cheaply
than appropriately tuned simpler systems? Ranking, calibration, physical cost,
failure count and computation must be reported separately. A lower average cost
with additional failures does not pass the failure guard. A cryptographic digest
binds artifact bytes; it cannot establish causal validity or generalization.
