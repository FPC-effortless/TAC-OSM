# Source semantics — no change to ME094 inputs, scores or registration

Re-read the exact public publisher README from the registered source revision
9290f5a96dc91ea58256e81d9ce2dd9621890b51. Archived unmodified in
source_metadata/README.publisher.md, SHA256
946b7f53833942f5e65d9f2ed0900c9458f00d97f2487e496c2266d16c430049.
Source: https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset/resolve/9290f5a96dc91ea58256e81d9ce2dd9621890b51/README.md
Attribution BodenAI/RW-RL, CC-BY-4.0. No additional episode records were opened.

The publisher lists synchronized state/action/video fields and retained indexing.
It does not establish within-row command issue versus sensor measurement order,
independent sub-frame camera synchronization or a causal intervention effect.
ME094 therefore keeps the exact registered reference-through-t-1 input. A same-row
action is not silently admitted as an easy shortcut. A future action-input change
requires separately verified forecast-time availability and a new registration;
it cannot retroactively qualify this experiment or claim equal-information gains.

Publisher feature metadata does not name the14individual components. Earlier
frozen diagnostic field labels called coordinates6/13 'grippers'; that anatomical
mapping is an inference, not independently verified from these feature metadata.
The arithmetic and original records are unchanged; models treat indexed channels.
Report aggregate error as source qpos-space units, not a certified SI joint-safety
metric or a validated control consequence. Recorded frame horizon is not guaranteed
fixed physical duration when capture gaps occur.

The publisher also warns of intervention-focused selection, mixed control
authority and absent episode-level success labels. Registered autonomous-segment
forecast errors cannot establish deployment success, causal repair/action benefit,
safe closed-loop control or unbiased rollout performance. No authoritative workflow
state is modified by this research model.
