# Preserved002 execution attempts

First registered source6e699e9cfa758ba242cc4ee5c98b080c46d4eb7c, run41000254179,
failed standalone CLI import (`ModuleNotFoundError: scripts`) BEFORE its first
cohort/model evaluation. Original checkpoint replay and prerequisite checks
passed in that job. No002scientific result was produced or selected.

Correct standalone root path setup; recipe/checkpoints/cohort/thresholds unchanged.
Also correct the summary's unavailable count to include every registered query
with no matching history, without dropping or changing any cohort cells.
Preserve this failed run and subsequent successful original artifact separately.
