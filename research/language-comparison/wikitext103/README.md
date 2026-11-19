# Larger FIT schedule

We staged both WikiText-103 raw training shards from Salesforce/wikitext, revision b08601e04326c79dfdd32d625aee71d232d685c3. The upstream dataset is derived from Wikipedia and lists CC BY-SA 3.0 and GFDL. The release receipt records SHA-256 digests of both original shards and all three generated orders. This directory contains coordinates and provenance instructions. Redistribution of the source text is unnecessary. [Dataset](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1)

Each seed has 16,384 planned updates, eight row-local windows per update and 256 next-byte targets per window. Each of the 33,554,432 target positions occurs once in its seed's order. Repeated text at different source positions and overlap across seeds are possible. Rows shorter than 257 bytes and trailing fragments are excluded. The seed files store nine-byte little-endian records consisting of a one-byte shard index, a four-byte row index and a four-byte UTF-8 byte offset. Read 257 bytes at that offset, give bytes 0–255 to the model, and score bytes 1–256. This schedule resets state at window start and does not evaluate long-document memory.

Use CPython 3.12 and PyArrow 26.0.1. The following regenerates the orders and compares the receipt, including the hashes of the order files.

```sh
mkdir -p shards
for shard in train-00000-of-00002 train-00001-of-00002; do
  curl -fL "https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/$shard.parquet" -o "shards/$shard.parquet"
done
python prepare.py shards orders > regenerated.json
diff -u FIT_RECEIPT.json regenerated.json
```

The schedule covers 25.6 times the byte presentations and 6.4 times the updates per fit of the 10M LANGSCALE001 campaign. These are planned exposures. No model has trained on this new schedule in this work. WikiText-103 validation has the same published SHA-256 as the WikiText-2 validation file already used elsewhere in this research. Use a separately frozen, access-separated evaluation source for a final quality claim.
