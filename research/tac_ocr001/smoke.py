"""Controlled repeated-image throughput/memory smoke; NOT long-document OCR validation.

Uses an attributed upstream Tesseract fixture, repeated across 1/10/50/100
pages. Identical pages do not test layout diversity, handwriting, semantics,
cross-page reasoning, or genuine 100-page transcription quality.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import resource

from PIL import Image

from .pipeline import PipelineConfig,file_sha256,run_document


def process_tree_rss(pid):
    """Sample Linux resident memory, including Tesseract children (not VSZ)."""
    root=Path("/proc")/str(pid)
    try:
        rss=int((root/"statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
    except (OSError,IndexError,ValueError):
        return 0,0
    children=set()
    for task in (root/"task").glob("*"):
        try:
            children.update(int(x) for x in (task/"children").read_text().split())
        except OSError:
            pass
    descendants=[process_tree_rss(child) for child in children]
    return rss+sum(x[0] for x in descendants),len(children)+sum(x[1] for x in descendants)


def worker(source,output,workers):
    stop=threading.Event()
    peak=[0]
    observed_children=[0]
    # Some managed runtimes virtualize getpid while exposing host /proc.
    # Resolve this process from self/status; never sample an unrelated PID.
    host_pid=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('Pid:')))
    def watch():
        while not stop.is_set():
            sampled,children=process_tree_rss(host_pid)
            peak[0]=max(peak[0],sampled)
            observed_children[0]=max(observed_children[0],children)
            stop.wait(.02)
    thread=threading.Thread(target=watch,daemon=True)
    thread.start()
    try:
        result=run_document(source,output,PipelineConfig(workers=workers))
    finally:
        stop.set()
        thread.join()
    texts=[]
    # The *smoke checker* collects only page hashes, not page images/text.
    with (output/"pages.jsonl").open() as stream:
        next(stream)
        import hashlib
        for index,line in enumerate(stream,1):
            row=json.loads(line)
            assert row["page"]==index
            texts.append(hashlib.sha256(row["text"].encode()).hexdigest())
    result.update({"sampled_peak_process_tree_rss_bytes":peak[0] if observed_children[0] else None,
        "observed_max_child_count":observed_children[0],
        "total_memory_status":"SAMPLED_PARTIAL_PROCESS_TREE" if observed_children[0] else "INCONCLUSIVE_NO_CHILD_PROCESS_VISIBILITY",
        "peak_main_rss_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        "peak_single_child_rss_bytes":int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024),
        "memory_measurement":"Linux /proc every 20ms; total withheld if OCR children unobservable. Main and single-child HWM from getrusage, not a simultaneous total.",
        "all_page_records_present":len(texts)==result["pages"],
        "repeated_image_transcripts_identical":len(set(texts))==1,
        "pages_per_second":result["pages"]/result["elapsed_seconds"]})
    return result


def run_smoke(output):
    import fitz
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    fixture=Path(__file__).parent/"fixtures"/"phototest.tif"
    with Image.open(fixture) as image:
        width,height=image.size
        import io
        encoded=io.BytesIO()
        image.convert("L").save(encoded,format="PNG")
    rows=[]
    for pages,workers in ((1,2),(10,2),(50,2),(100,2),(50,1)):
        source=output/f"repeated-{pages}.pdf"
        if not source.exists():
            with fitz.open() as document:
                xref=0
                for _ in range(pages):
                    page=document.new_page(width=width*72/200,height=height*72/200)
                    xref=page.insert_image(page.rect,stream=encoded.getvalue() if not xref else None,xref=xref)
                document.save(source)
        destination=output/f"pages-{pages}-workers-{workers}"
        command=[sys.executable,"-m","research.tac_ocr001.smoke","--worker",str(source),
                 "--output",str(destination),"--workers",str(workers)]
        process=subprocess.run(command,capture_output=True,text=True,check=True,timeout=180)
        rows.append(json.loads(process.stdout))
    report={"schema":"tac-ocr001.repeated-fixture-smoke.v1","source":"https://github.com/tesseract-ocr/test/blob/main/testing/phototest.tif",
        "source_sha256":file_sha256(fixture),"new_optimizer_updates":0,
        "qualification":"controlled_software_stress_only_not_real_document_benchmark",
        "limitations":["Repeated one source image; no document diversity or cross-page task", "Tesseract pretrained backend; custom TAC model not used", "Linux process-tree RSS is sampled, not GPU VRAM or exact allocator peak"],
        "source_code_sha256":{name:file_sha256(Path(__file__).with_name(name)) for name in ("smoke.py","pipeline.py")},
        "runs":rows}
    (output/"SMOKE.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    print(json.dumps(worker(args.worker,args.output,args.workers) if args.worker else run_smoke(args.output)))
