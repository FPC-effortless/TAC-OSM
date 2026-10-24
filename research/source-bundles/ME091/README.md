# Exact observed image inputs for ME091

Derived from **BodenAI / RW-RL**, RW-RL-HIL-Dataset, **CC BY4.0**:
https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset
Pinned revision9290f5a96dc91ea58256e81d9ce2dd9621890b51.
License: https://creativecommons.org/licenses/by/4.0/ . Attribution retained;
no endorsement by the original authors is implied.

Modification: overhead video decoded and area-resized to32x32 grayscale uint8
using recorded ffmpeg7.1.5. These are observed inputs, not learned parameters,
frozen-encoder features, targets or synthetic samples. Only the existing120FIT/
40exposedREPORT episodes appear; CAL/TEST/tissue-box are excluded. Full image
episodes are retained for reproducibility; model batching supplies only frames
through the observed cutoff, never future images.

SOURCE_INDEX lists original publisher LFS SHA256, frame count, decoded pixel
SHA256 and artifact SHA256 per episode. Initial acquisition verified all160
publisher videos (4,633,584,032bytes); temporary MP4s were removed. The exact
114,832,136-byte decoded input cache is Git-owned to avoid decoding4.6GB every
replay or changing observations with decoder versions.

Restore: `python scripts/me091_restore_pixels.py`.
Re-derive: `PYTHONPATH=src:. python scripts/me091_visual_source.py --cache /fresh/cache`.
Different decoding pixels do not permit changing the original input contract or
replay tolerance. Hash/replay failures remain failures. Numeric observations
are restored from pinned publisher parquet by the existing loader. Neither
input restoration operation performs fitting. This is research-only public
licensed data; no proprietary customer records or independently verified pilot.
