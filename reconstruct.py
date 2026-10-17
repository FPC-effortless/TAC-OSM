from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
for name, spec in json.loads((root / "manifest.json").read_text()).items():
    chunks = []
    for part in spec["parts"]:
        data = (root / part["path"]).read_bytes()
        assert len(data) == part["bytes"]
        assert hashlib.sha256(data).hexdigest() == part["sha256"]
        chunks.append(data)
    data = b"".join(chunks)
    assert len(data) == spec["bytes"]
    assert hashlib.sha256(data).hexdigest() == spec["sha256"]
    (root / name).write_bytes(data)
    print(name, len(data), spec["sha256"])
