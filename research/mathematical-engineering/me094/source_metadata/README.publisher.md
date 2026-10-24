---

pretty_name: RW-RL-HIL-Dataset
language:

- en
tags:
- robotics
- reinforcement-learning
- real-world-robotics
- embodied-ai
- human-intervention
- multimodal
- lerobot
license: cc-by-4.0

---

# RW-RL-HIL-Dataset: Real-World Reinforcement Learning Dataset With Human Intervention

![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-RW--RL%20Human%20Intervention-yellow) ![BodenAI](https://img.shields.io/badge/Released%20by-BodenAI-blue)![Data scale](https://img.shields.io/badge/Data-82.23%20real--robot%20hours-green)![Robot](https://img.shields.io/badge/Robot-R1Lite-orange)[

This open-source release from [BodenAI](https://huggingface.co/BodenAI) is a human-intervention subset
of the [RW-RL-Dataset](https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset). It contains **82.23
hours** of R1Lite household manipulation data collected during real-world policy deployment. When the
policy entered a state it could not handle, an operator took over, corrected the behavior, and handed
control back whenever recovery was completed.

The nine task directories preserve the policy's own trajectories together with human corrections,
providing data for failure detection, recovery learning, value estimation, and real-world policy
improvement. The measured duration below is computed from retained frames, replacing the earlier
rounded 80-hour description.

Of **3,347 episodes**, **3,068 contain at least one human intervention** and **279 have no intervention
frames** in the saved annotations. The latter remain in the supplied directories and are included in
all statistics below. Use `has_intervention == true` for a strictly intervention-containing subset.
This is deployment data rather than staged teleoperation demonstrations recorded for their own sake.

> This subset focuses on the RW-RL **human intervention** regime. It is designed
> to be combined with the teleoperation and autonomous-rollout data in the main RW-RL repository rather
> than to replace it.



## Dataset Coverage and Demo

![RW-RL scenario and embodiment coverage](https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset/resolve/main/assets/scene_embodiment_coverage_en.png)  
Scenario and embodiment coverage of the broader RW-RL data program. This release focuses on R1Lite human-intervention data.

RW-RL real-robot task demonstration.

## Why Human Intervention Data?

An imitation-learning corpus tells a policy what the ideal trajectory looks like. It does not tell the
policy what happens at the edge of its own competence, because that edge only exists once a specific
policy is actually deployed. Intervention data is generated at that edge by construction.

Three properties make this kind of data hard to synthesize any other way:

- **The takeover moment is a free failure label.** The frame at which an experienced operator decides
that the robot can no longer recover is a human judgment about impending failure, recorded without a
separate annotation pass. The distribution of those moments is itself a training signal for failure
detection, uncertainty estimation, and value learning.
- **The correction is a recovery demonstration from an off-distribution state.** The human does not
start from a tidy reset pose. They start from wherever the policy stranded the robot, which is
precisely the state distribution that demonstration-only datasets never cover.
- **The hand-back is a re-entry point.** After the correction, control returns to the policy from a
state the policy did not create, which gives a natural evaluation and continuation boundary.



## Subset at a Glance


| Dimension           | Description                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------ |
| Dataset             | [RW-RL-HIL-Dataset](https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset)             |
| Released by         | [BodenAI](https://huggingface.co/BodenAI)                                                  |
| Parent dataset      | [RW-RL-Dataset](https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset)                   |
| Total duration      | 82.23 hours (measured from retained trajectories)                                          |
| Tasks               | 9 household manipulation tasks                                                             |
| Robot               | R1Lite                                                                                     |
| Data regime         | Policy deployment with human intervention; includes 279 zero-intervention episodes         |
| Scene domain        | Household and daily-living environments                                                    |
| Episodes            | 3,347 (3,068 with intervention; 279 without)                                               |
| Total frames        | 4,440,155                                                                                  |
| Total size          | 188.57 GB (175.62 GiB; logical file sizes across all nine directories)                     |
| Format              | LeRobot v2.1: per-episode parquet and per-camera per-episode video, JSONL episode metadata |
| Video               | 15 FPS H.264, 640 × 360 (metadata specification)                                           |
| Camera streams      | 3: head, left wrist, right wrist                                                           |
| Intervention labels | Per-frame control-authority flag and intervention segment boundaries                       |




## Position in the RW-RL Data Regimes

The RW-RL program defines three data regimes. This subset focuses on intervention during deployment,
on a single embodiment; the supplied directories also retain episodes without a recorded takeover.


| Regime              | Covered here  | Notes                                                                                                                                            |
| ------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Human teleoperation | No            | Available in the main RW-RL repository.                                                                                                          |
| Human intervention  | Primary focus | 3,068 episodes contain one or more contiguous intervention segments.                                                                             |
| Autonomous rollout  | Present       | Policy-controlled segments occur within episodes; 279 episodes have no intervention frames. Absence of intervention does not imply task success. |




## Anatomy of an Intervention Episode

An episode containing intervention can follow the control-authority structure below. The per-frame
flag and segment boundaries let you slice these phases directly.


| Phase             | Control authority | What it captures                                                                                              |
| ----------------- | ----------------- | ------------------------------------------------------------------------------------------------------------- |
| Autonomous prefix | Policy            | The policy executing on the real robot up to the point of trouble, including the drift that leads to failure. |
| Takeover          | Human             | The moment the operator judges that the policy cannot recover on its own.                                     |
| Correction        | Human             | The recovery trajectory, starting from an off-distribution state the policy produced.                         |
| Hand-back         | Policy            | Control returns to the policy from a state it did not generate.                                               |


An episode may contain multiple takeover / correction / hand-back cycles. In the current data,
114 episodes end while intervention is still active, so a hand-back is not always recorded. The
per-frame `intervention` flag is the authoritative source for control authority at any timestep.

## Task Coverage

Nine household manipulation tasks, **82.23 measured hours** total. The task identifiers below follow
the RW-RL naming convention with the `r1lite_` prefix. Statistics were computed on 2026-09-14 from the
nine supplied task directories. Hours count synchronized trajectory time once, not once per camera.


| Task folder                        | Description                                                | Hours     | Episodes  | Video files |
| ---------------------------------- | ---------------------------------------------------------- | --------- | --------- | ----------- |
| `r1lite_wipe_table_and_furniture`  | Clear and wipe a cluttered tabletop and furniture surfaces | 10.21     | 384       | 1,152       |
| `r1lite_spray_bottle_head_install` | Install a spray head onto a bottle neck                    | 10.63     | 424       | 1,272       |
| `r1lite_trash_bag_install`         | Fit and secure a new trash bag into a bin                  | 10.25     | 461       | 1,383       |
| `r1lite_hang_clothes_on_rack`      | Hang garments on a drying rack or hanger                   | 10.50     | 291       | 873         |
| `r1lite_fold_clothes`              | Fold garments on a flat surface                            | 10.23     | 321       | 963         |
| `r1lite_plug_into_power_strip`     | Insert a plug into a power strip socket                    | 4.13      | 275       | 825         |
| `r1lite_make_tea_and_drinks`       | Brew tea and prepare simple beverages                      | 10.07     | 177       | 531         |
| `r1lite_tissue_box_replace`        | Replace tissues and perform routine desktop upkeep         | 6.02      | 509       | 1,527       |
| `r1lite_scoop_cat_litter`          | Scoop and dispose of cat litter clumps                     | 10.18     | 505       | 1,515       |
| **Total**                          |                                                            | **82.23** | **3,347** | **10,041**  |


The policy that was running during collection is an internal VLA policy deployed on R1Lite. Intervention
statistics therefore describe the competence boundary of that specific policy on these tasks, not an
absolute difficulty ranking of the tasks themselves.

### Intervention Duration and Time Share

The supplied [analysis script](scripts/analyze_lerobot_intervention.py) was run on the server without
filtering or removing episodes. It defines human control as `intervention == true` and segments as
maximal contiguous runs of that flag, in `frame_index` order. The saved `timestamp` grid is 15 FPS
in all nine directories; this is retained trajectory time, not raw capture wall-clock time.

The script computes episode time as `last timestamp - first timestamp + median frame interval` and
sums frame intervals under human control for intervention time (using the median interval for the
last frame). Frame-weighted share is `sum(intervention frames) / sum(all frames)`; its difference
from timestamp-weighted share is below 0.000001 percentage points in every task. Episode percentiles
and histogram bins use `intervention frames / episode length`, with every episode weighted equally.
Totals pool all frames, episodes, or segments as appropriate; they do not average task percentages.

Across the collection, **506,832 frames / 9.39 hours (11.41%)** are under human control. There are
**6,408 contiguous intervention segments**, averaging **1.91 per episode** and **5.27 seconds per
segment**. The median segment lasts **4.33 s**, and the 90th percentile is **10.40 s**. Per-episode
intervention time (summing all segments, including zero-intervention episodes) has median **8.07 s**
and 90th percentile **22.00 s**.


| Task                    | Human time (h) | Human frame share | Episode share P50 | Episode share P90 | Zero-intervention episodes |
| ----------------------- | -------------- | ----------------- | ----------------- | ----------------- | -------------------------- |
| Surface wiping          | 1.84           | 18.01%            | 16.04%            | 35.05%            | 12                         |
| Spray head installation | 1.04           | 9.80%             | 9.06%             | 16.33%            | 0                          |
| Trash bag installation  | 1.58           | 15.41%            | 13.15%            | 24.94%            | 11                         |
| Garment hanging         | 1.55           | 14.74%            | 13.93%            | 22.20%            | 0                          |
| Garment folding         | 1.02           | 9.96%             | 9.29%             | 16.85%            | 0                          |
| Plug into power strip   | 0.61           | 14.73%            | 14.87%            | 19.08%            | 0                          |
| Tea preparation         | 0.80           | 7.97%             | 7.39%             | 14.36%            | 0                          |
| Tissue replacement      | 0.52           | 8.71%             | 8.58%             | 18.06%            | 159                        |
| Cat litter scooping     | 0.42           | 4.16%             | 2.63%             | 9.90%             | 97                         |
| **All tasks**           | **9.39**       | **11.41%**        | **10.56%**        | **19.91%**        | **279**                    |


**Episode intervention-share distribution.** Cells are episode counts. Intervals include the lower
bound and exclude the upper bound; the last bin includes 100%. The first bin includes zero-intervention
episodes. For readability, the script's bins at or above 30% are combined here; the full histogram,
P10/P25/P50/P75/P90/P95 values, and threshold counts are in the
[machine-readable summary](analysis/intervention_summary.json).


| Task                    | [0, 5)% | [5, 10)% | [10, 15)% | [15, 20)% | [20, 25)% | [25, 30)% | [30, 100]% |
| ----------------------- | ------- | -------- | --------- | --------- | --------- | --------- | ---------- |
| Surface wiping          | 23      | 49       | 107       | 65        | 45        | 26        | 69         |
| Spray head installation | 69      | 159      | 125       | 71        | 0         | 0         | 0          |
| Trash bag installation  | 37      | 105      | 131       | 68        | 75        | 28        | 17         |
| Garment hanging         | 17      | 53       | 92        | 79        | 39        | 11        | 0          |
| Garment folding         | 57      | 117      | 82        | 65        | 0         | 0         | 0          |
| Plug into power strip   | 1       | 20       | 121       | 133       | 0         | 0         | 0          |
| Tea preparation         | 37      | 96       | 28        | 16        | 0         | 0         | 0          |
| Tissue replacement      | 200     | 88       | 123       | 98        | 0         | 0         | 0          |
| Cat litter scooping     | 367     | 90       | 29        | 19        | 0         | 0         | 0          |
| **All tasks**           | **808** | **777**  | **838**   | **614**   | **159**   | **65**    | **86**     |


**Continuous intervention segment duration.** Duration is `(end_frame - start_frame + 1) / 15`
seconds. Percentiles use linear interpolation across all segments in the indicated task. Episodes
without intervention contribute no segments to this table; segments still active at an episode's end
are included at their observed length, so these durations do not imply a completed recovery.


| Task                    | Segments  | Mean (s) | P50 (s)  | P90 (s)   | P95 (s)   | Max (s)   |
| ----------------------- | --------- | -------- | -------- | --------- | --------- | --------- |
| Surface wiping          | 628       | 10.54    | 10.07    | 15.53     | 18.00     | 47.20     |
| Spray head installation | 915       | 4.10     | 3.80     | 7.04      | 8.53      | 17.60     |
| Trash bag installation  | 701       | 8.11     | 7.53     | 13.07     | 15.47     | 26.93     |
| Garment hanging         | 1,515     | 3.68     | 3.00     | 6.53      | 8.22      | 21.80     |
| Garment folding         | 515       | 7.12     | 6.67     | 11.31     | 13.40     | 24.33     |
| Plug into power strip   | 422       | 5.19     | 5.77     | 9.46      | 10.33     | 14.07     |
| Tea preparation         | 597       | 4.84     | 4.20     | 8.36      | 10.41     | 29.13     |
| Tissue replacement      | 525       | 3.60     | 3.20     | 6.67      | 7.72      | 14.67     |
| Cat litter scooping     | 590       | 2.59     | 1.93     | 5.00      | 6.64      | 14.93     |
| **All tasks**           | **6,408** | **5.27** | **4.33** | **10.40** | **12.80** | **47.20** |


To reproduce the per-task time-share analysis on a machine with the data and `numpy`, `pandas`, and
`pyarrow` installed:

```bash
python scripts/analyze_lerobot_intervention.py /path/to/task_dataset \
  --thresholds 15,20,25,30
```

All 10,041 expected video paths (three per episode) exist. Counts above come from the filesystem:
`meta/info.json` has stale `total_videos` values for trash bag installation (423 versus 1,383 actual),
spray head installation (276 versus 1,272), and surface wiping (90 versus 1,152). These server files
were left unchanged. Video existence and metadata specifications were checked; videos were not decoded
for this analysis.

## Intervention Annotations

Beyond the standard RW-RL video, state, and action fields, this subset adds the control-authority
labels that make the data usable for human-in-the-loop learning.


| Annotation                   | Location                         | Description                                                                                                                                                                                     |
| ---------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Per-frame control authority  | `data/chunk-*/episode_*.parquet` | Boolean `intervention`: `true` means human control; `false` means policy control. `control_source` is respectively `"intervention"` or `"policy"`.                                              |
| Source segment index         | Same parquet                     | `intervention_segment_index` is the recorded segment label. It can differ from the contiguous-run numbering reconstructed in episode metadata; use the boolean flag to slice control authority. |
| Episode intervention summary | `meta/episodes.jsonl`            | `has_intervention`, `intervention_segments`, `intervention_frames`, and `intervention_ranges`, rebuilt from the parquet flag.                                                                   |


Each `meta/episodes.jsonl` line has exactly these fields, matching the target sample:

```json
{"episode_index": 1, "tasks": ["scoop_cat_litter_into_trash_can"], "length": 981, "has_intervention": true, "intervention_segments": 1, "intervention_frames": 125, "intervention_ranges": [{"segment_index": 0, "start_frame": 702, "end_frame": 826}]}
```

Frame indices are episode-local, zero-based, and **both endpoints are inclusive**. Segment indices
start at 0 within each episode. The invariants are
`intervention_segments == len(intervention_ranges)` and
`intervention_frames == sum(end_frame - start_frame + 1)`.
An episode without intervention has `has_intervention: false`, both counts set to `0`, and an empty
`intervention_ranges` list. Existing episode order, `episode_index`, `tasks`, and `length` are preserved;
other collection-time metadata fields are omitted from this normalized file.

The boolean flag and `control_source` agree on all 4,440,155 frames. The original parquet
`intervention_segment_index` differs from reconstructed contiguous-run labels on **4,567 frames**;
parquet files remain unchanged. Rebuilding also corrected pre-existing intervention summaries in
**42 episodes**. Consumers should use the normalized ranges or derive runs directly from `intervention`,
rather than assuming source segment labels equal the new `segment_index` values.

This release does **not** ship scalar `reward` values, `done` flags, structured intervention reasons, or
episode-level success labels. The control-authority signal is intended to be used directly as an
intervention-based supervision signal, in the style of HG-DAgger, interactive imitation learning, and
RLPD-style off-policy methods that treat human actions as high-value transitions.

## Quality Control

The collection workflow uses automated checks for motion, physical ranges, synchronization, and
episode completeness. This metadata reconstruction did not rerun those collection-time quality checks
or certify every episode against a timing-error threshold. Collection-time quality fields are omitted
from the normalized `episodes.jsonl` schema.

For this update, all 3,347 episode parquet files were checked against their metadata: episode IDs and
lengths agree, frame indices are contiguous from zero, timestamps are finite and increasing, and
intervention flags agree with `control_source`. Reconstructed ranges cover exactly the intervention
frames and respect episode boundaries. All expected camera video paths exist. Zero-intervention
episodes, source segment-label differences, and intervention-active episode endings are reported above.

## File Layout

Each task folder is organized as follows:

```text
<task_name>/
  data/
    chunk-000/
      episode_000000.parquet
      episode_000001.parquet
      ...
  videos/
    chunk-000/
      observation.images.cam_high/
        episode_000000.mp4
        ...
      observation.images.cam_left_wrist/
        episode_000000.mp4
        ...
      observation.images.cam_right_wrist/
        episode_000000.mp4
        ...
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl  # present in some tasks
    stats.json           # present in some tasks
```

The LeRobot v2.1 path templates in `meta/info.json` are:

```text
data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
```

With `chunks_size = 1000`, `episode_chunk = episode_index // 1000`. Filenames use the global
`episode_index`, not an index modulo the chunk size. `meta/episodes.jsonl` stores one JSON object per
episode and retains the standard `episode_index`, `tasks`, and `length` fields.

## Modalities and Fields

The inspected files expose the following synchronized fields. State and action vectors each have
14 components; component names are not provided in the current feature metadata.


| Group                     | Fields                                                                                                   |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| Multi-view video          | `observation.images.cam_high`, `observation.images.cam_left_wrist`, `observation.images.cam_right_wrist` |
| Robot state               | `observations.state.qpos` (14-dimensional)                                                               |
| Robot action              | `action.qpos` (14-dimensional)                                                                           |
| Control authority         | `intervention`, `control_source`, `intervention_segment_index`                                           |
| Episode indexing and time | `timestamp`, `capture_time`, `frame_index`, `episode_index`, `index`, `task_index`                       |




## Downloading a Subset

Use `huggingface_hub` to download a single task folder without pulling the entire repository:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BodenAI/RW-RL-HIL-Dataset",
    repo_type="dataset",
    allow_patterns=[
        "r1lite_scoop_cat_litter/**",
    ],
    local_dir="RW-RL-HIL",
)
```

To inspect only metadata first:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BodenAI/RW-RL-HIL-Dataset",
    repo_type="dataset",
    allow_patterns=[
        "*/meta/**",
    ],
    local_dir="RW-RL-HIL-meta",
)
```



## How This Subset Complements the RW-RL Releases

Several of the task templates here also appear in the main RW-RL repository. That overlap is intentional:
the same task template can be paired across regimes, so a policy can be pretrained on the demonstration
data and then corrected with intervention data from the same task family.


| Task template             | Main RW-RL repository                                                                                   | This subset                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Spray bottle head install | `r1lite_spray_bottle_head_install`, `so101_insert_spray_pump_bottle_neck` — teleoperated demonstrations | `r1lite_spray_bottle_head_install` — intervention traces during deployment |
| Tissue box replacement    | `r1lite_tissue_box_replace` — teleoperated demonstrations                                               | `r1lite_tissue_box_replace` — intervention traces during deployment        |
| Plug into power strip     | `r1lite_plug_charger_to_power_strip`, `so101_plug_power_cord_socket` — teleoperated demonstrations      | `r1lite_plug_into_power_strip` — intervention traces during deployment     |
| Fold clothes              | `so101_fold_clothes_left_stack_right` — teleoperated demonstrations, SO101 embodiment                   | `r1lite_fold_clothes` — intervention traces on R1Lite                      |


The remaining five tasks — surface wiping, trash bag installation, garment hanging, beverage
preparation, and cat litter scooping — extend the RW-RL program into long-tail household chores that
are not covered by the industrial, logistics, and retail domains in the main repository.

## Intended Use and Limitations

This dataset is built for a specific research setting, and reading it as a general-purpose imitation
corpus will produce misleading results.

- **Selection bias is substantial.** These directories come from intervention-oriented deployment
collection and processing. They include 279 episodes without recorded intervention, but are not an
unbiased sample of policy rollouts and have no episode-level success labels. Intervention absence
cannot be used as a success label or to estimate the deployed policy's success rate.
- **The action stream is mixed-authority.** Within a single episode, some actions come from the policy
and some from the human. Training a behavior-cloning policy on the concatenated stream without reading
the control-authority flag will fit a controller that does not exist. Filter or weight by the flag.
- **Intervention timing reflects operator judgment.** Different operators take over at different
thresholds of perceived risk. The takeover distribution encodes human caution as well as robot
failure.
- **Statistics are policy-specific.** Intervention rates characterize the deployed policy at collection
time. They will not transfer to a different policy or a later checkpoint.



## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You may share
and adapt the material, including for commercial purposes, provided that you give appropriate credit
and indicate whether changes were made.

## Project Links

- Parent dataset: [https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset](https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset)
- RW-RL GitHub: [https://github.com/MINT-SJTU/Evo-RL](https://github.com/MINT-SJTU/Evo-RL)
- This dataset: [https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset](https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset)
- BodenAI: [https://huggingface.co/BodenAI](https://huggingface.co/BodenAI) 
  - Offical Website: [https://www.boden.ai/](https://www.boden.ai/)
  - Github: [https://github.com/BodenAI](https://github.com/BodenAI)
  - Linkedin: [https://www.linkedin.com/company/bodenai](https://www.linkedin.com/company/bodenai)
  - X: [https://x.com/BodenAI](https://x.com/BodenAI)



## Citation

If you use this subset, please cite both this release and the parent RW-RL Dataset.

```bibtex
@misc{rw_rl_human_intervention_2026,
  title        = {RW-RL-HIL-Dataset: Real-World Reinforcement Learning Dataset With Human Intervention},
  author       = {BodenAI},
  year         = {2026},
  howpublished = {https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset}
}

@misc{rw_rl_dataset_2026,
  title        = {RW-RL Dataset: Real-World Reinforcement Learning Dataset},
  author       = {Boden Intelligence and Junpu Innovation Center and MINT Lab, Shanghai Jiao Tong University},
  year         = {2026},
  howpublished = {https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset}
}
```

