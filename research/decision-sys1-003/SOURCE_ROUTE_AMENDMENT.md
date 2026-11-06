# Source-route amendment — before authentic TRAIN acquisition

The initial run42257704781 at e02565e failed 404/401 on legacy query routes; its artifact13522344533/digest98e2cd89a565f574b02bf42f0229cd154f42d8c491f60966c0bb3af862696357 is immutable. No source/labels were acquired.

Public-catalogue-only run42259412779 at605d06f4d90992a21c08a7581d15f0905a1970bb succeeded in fetching metadata, NOT data. Original artifact13524496067/digest94c5c7cef6761339249f4119142da113dff5e2ebfa6746fde30150b463008011 contains public links and SHA256 declarations. Catalogue body SHA256 ea815a273a8240125c3c5279d2b6b39a2d575dfd9d6f9822ac89a96bf534a854. No protected data bodies were opened.

The publisher now supplies exact path-based routes:
- https://api.researchdata.se/dataset/2024-34/3/file/data/train_operational_readouts.csv
- https://api.researchdata.se/dataset/2024-34/3/file/data/train_tte.csv

Freeze both source checksums before this new attempt:
- readouts,1219209878 bytes: e01cb0bd87dfab4c9dbad215d51d81282fc0d413be96d6f819ea872fb7a3c715 (also matches historical receipt)
- outcomes,345412 bytes: d8c2379ed7c95a575dd869730b2b3b96d660317f49e57de300518ff3b08d53a5 (new publisher declaration; not independent evaluator attestation)

Only these two exact paths are allowed; remove query-route/other-host alternatives. The same size/time limits and unchanged original #719 auditor apply. Both hashes must match. No fitting, model scoring, validation/test access, archive or source upload. Labels requested only after readout authentication. Seven local route/fixture tests passed before this source attempt; generated fixtures are not empirical capability. Workflow now triggers only relevant source-file pushes on this named research branch, avoiding re-downloading data for unrelated model/reference/report changes. Existing exposures and failures remain. This route amendment does not authorize training or change scientific targets.
