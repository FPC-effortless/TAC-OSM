"""Record local child CPU/wall/RSS without a platform-specific time binary."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


if __name__ == '__main__':
    destination = Path(sys.argv[1]); command = sys.argv[2:]
    if not command:
        raise SystemExit('Expected usage report followed by command arguments')
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter(); result = subprocess.run(command, check=False)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(command=command, exit_code=result.returncode,
        user_seconds=after.ru_utime-before.ru_utime, system_seconds=after.ru_stime-before.ru_stime,
        wall_seconds=time.perf_counter()-start, max_child_rss_kib=after.ru_maxrss), indent=2)+'\n')
    raise SystemExit(result.returncode)
