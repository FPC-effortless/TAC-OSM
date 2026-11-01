"""Bounded-page OCR. All document-length output and search state live on disk.

Tesseract is an unmatched pretrained practical backend, not a TAC result.
No hidden language-model correction, training, cloud API, or remote upload.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time

from PIL import Image, ImageOps


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PipelineConfig:
    dpi: int = 200
    workers: int = 2
    max_pixels: int = 8_000_000
    language: str = "eng"
    psm: int = 3
    mode: str = "ocr"
    timeout_seconds: int = 120
    review_confidence: float = 65.
    max_page_output_bytes: int = 8_000_000

    def __post_init__(self):
        for key, low, high in (("dpi",72,600),("workers",1,8),("max_pixels",100_000,40_000_000),
                               ("psm",1,13),("timeout_seconds",1,1800),
                               ("max_page_output_bytes",1024,64_000_000)):
            if type(getattr(self,key)) is not int or not low <= getattr(self,key) <= high:
                raise ValueError(f"invalid {key}")
        if self.mode not in ("ocr", "auto") or not re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*",self.language):
            raise ValueError("explicit OCR/auto mode and installed language names required")
        if not math.isfinite(self.review_confidence) or not 0 <= self.review_confidence <= 100:
            raise ValueError("invalid review threshold")


class ReferenceIndex:
    """Exact page text on disk; FTS retrieval returns bounded text bytes.

    This is a usable search API. It does NOT feed extra text to the neural
    candidate or pretend that lexical search is learned visual routing.
    """
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA cache_size=-1024")
        self.db.execute("PRAGMA mmap_size=0")
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS pages USING fts5(page UNINDEXED, text)")

    def add(self, page, text):
        self.db.execute("DELETE FROM pages WHERE page=?", (page,))
        self.db.execute("INSERT INTO pages(page,text) VALUES(?,?)", (page,text))
        self.db.commit()

    def search(self, query, *, limit=8, max_bytes=8192):
        if not 1 <= limit <= 16 or not 1 <= max_bytes <= 65536:
            raise ValueError("bounded reference budget required")
        terms = re.findall(r"\w+",query, re.UNICODE)[:32]
        if not terms:
            return []
        expression = " OR ".join('"'+term+'"' for term in terms)
        rows = self.db.execute("SELECT page,snippet(pages,1,'','',' … ',24) FROM pages WHERE pages MATCH ? ORDER BY bm25(pages) LIMIT ?", (expression,limit))
        result, remaining = [], max_bytes
        for page,text in rows:
            bounded = text.encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
            if bounded:
                result.append({"page":int(page), "text":bounded})
                remaining -= len(bounded.encode("utf-8"))
            if not remaining:
                break
        return result

    def close(self):
        self.db.close()


def _shrink(image, max_pixels):
    if image.width*image.height > max_pixels:
        scale = math.sqrt(max_pixels/(image.width*image.height))
        image = image.resize((max(1,int(image.width*scale)), max(1,int(image.height*scale))), Image.Resampling.LANCZOS)
    return image


def iter_pages(path, config, *, skip=0):
    """Lazy PDF/TIFF reader. No list of all pages, images or recognized text."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import fitz
        with fitz.open(path) as document:
            if document.needs_pass:
                raise ValueError("encrypted PDF requires an already decrypted input")
            for number in range(skip,len(document)):
                started = time.perf_counter()
                page = document[number]
                native = None
                if config.mode == "auto":
                    words = page.get_text("words",sort=True)
                    # Conservative: a mixed image/text page still takes OCR.
                    if words and not page.get_images() and all("\ufffd" not in w[4] for w in words):
                        native = [{"text":w[4], "confidence":None,
                                   "bbox":[float(v) for v in w[:4]],
                                   "line":[int(w[5]),0,int(w[6])]} for w in words]
                if native is not None:
                    yield {"page":number+1,"native_words":native,"size":[page.rect.width,page.rect.height],
                           "coordinate_unit":"pdf_point","prepare_seconds":time.perf_counter()-started}
                else:
                    scale = min(config.dpi/72,math.sqrt(config.max_pixels/max(1,page.rect.width*page.rect.height)))
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale,scale),colorspace=fitz.csGRAY,alpha=False)
                    yield {"page":number+1,"image":Image.frombytes("L",(pixmap.width,pixmap.height),pixmap.samples),
                           "coordinate_unit":"pixel","prepare_seconds":time.perf_counter()-started}
    else:
        with Image.open(path) as original:
            for number in range(skip,getattr(original,"n_frames",1)):
                started = time.perf_counter()
                original.seek(number)
                image = _shrink(ImageOps.exif_transpose(original).convert("L"),config.max_pixels)
                yield {"page":number+1,"image":image,"coordinate_unit":"pixel",
                       "prepare_seconds":time.perf_counter()-started}


def _assemble(page,words,size,unit,backend,elapsed,prepare,threshold):
    lines, current, last = [], [], None
    for word in words:
        line = tuple(word.pop("line"))
        if line != last and current:
            lines.append(" ".join(current))
            current=[]
        current.append(word["text"])
        last=line
    if current:
        lines.append(" ".join(current))
    measured=[w["confidence"] for w in words if w["confidence"] is not None]
    confidence=sum(measured)/len(measured) if measured else None
    return {"schema":"tac-ocr001.page.v1","page":page,"backend":backend,
            "width":size[0],"height":size[1],"coordinate_unit":unit,
            "text":"\n".join(lines),"words":words,
            "mean_engine_confidence":confidence,
            "review_required":not words or (confidence is not None and confidence < threshold),
            "confidence_is_calibrated":False,
            "recognition_seconds":elapsed,"prepare_seconds":prepare}


def recognize_page(payload, config):
    if "native_words" in payload:
        return _assemble(payload["page"],payload["native_words"],payload["size"],payload["coordinate_unit"],
                         "pdf-text-layer",0.,payload["prepare_seconds"],config.review_confidence)
    started=time.perf_counter()
    image=payload["image"]
    # PGM avoids PNG compression and file round trips. Tesseract reads stdin.
    data=f"P5\n{image.width} {image.height}\n255\n".encode()+image.tobytes()
    env=dict(os.environ,OMP_THREAD_LIMIT="1",OMP_NUM_THREADS="1")
    command=["tesseract","stdin","stdout","-l",config.language,"--psm",str(config.psm),"tsv"]
    completed=subprocess.run(command,input=data,capture_output=True,env=env,
                             timeout=config.timeout_seconds,check=False)
    if completed.returncode:
        raise RuntimeError("Tesseract failed: "+completed.stderr.decode(errors="replace")[-500:])
    if len(completed.stdout)>config.max_page_output_bytes:
        raise ValueError("page OCR output exceeds configured limit")
    words=[]
    for row in csv.DictReader(io.StringIO(completed.stdout.decode("utf-8")),delimiter="\t",quoting=csv.QUOTE_NONE):
        if row["level"]!="5" or not row.get("text","").strip():
            continue
        x,y,w,h=(int(row[key]) for key in ("left","top","width","height"))
        words.append({"text":row["text"],"confidence":float(row["conf"]),"bbox":[x,y,x+w,y+h],
                      "line":[int(row[key]) for key in ("block_num","par_num","line_num")]})
    return _assemble(payload["page"],words,[image.width,image.height],payload["coordinate_unit"],
                     "tesseract-pretrained",time.perf_counter()-started,payload["prepare_seconds"],config.review_confidence)


def bounded_map(pages, config, recognizer=recognize_page):
    """Ordered output and at most workers rendered/submitted pages in flight."""
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        pending=deque()
        for _ in range(config.workers):
            try:
                pending.append(executor.submit(recognizer,next(pages),config))
            except StopIteration:
                break
        while pending:
            future=pending.popleft()
            result=future.result()
            # Yield before rendering/submitting more: strict bounded pressure.
            yield result
            try:
                pending.append(executor.submit(recognizer,next(pages),config))
            except StopIteration:
                pass


def _recover_pages(path,header,index):
    """Stream recovery; discard a torn trailing line, reject wrong ownership."""
    if not path.exists():
        return 0
    count,good=0,0
    with path.open("rb") as stream:
        first=stream.readline()
        if not first.endswith(b"\n") or json.loads(first)!=header:
            raise ValueError("resume source/config/backend mismatch")
        good=stream.tell()
        while raw:=stream.readline():
            if not raw.endswith(b"\n"):
                break
            page=json.loads(raw)
            if page.get("page")!=count+1 or page.get("schema")!="tac-ocr001.page.v1":
                raise ValueError("invalid or noncontiguous OCR journal")
            index.add(page["page"],page["text"])
            count+=1
            good=stream.tell()
    with path.open("r+b") as stream:
        stream.truncate(good)
    return count


def render_markdown(journal,target):
    temporary=target.with_suffix(".md.tmp")
    with journal.open() as source,temporary.open("w",encoding="utf-8") as output:
        next(source)
        for line in source:
            page=json.loads(line)
            output.write(f"<!-- page {page['page']}; source {page['backend']} -->\n\n")
            output.write(page["text"]+"\n\n")
    temporary.replace(target)


def run_document(source, output, config=None, *, resume=False):
    config=config or PipelineConfig()
    source,output=Path(source).resolve(),Path(output).resolve()
    if not source.is_file() or (source==output or output in source.parents):
        raise ValueError("existing input file and separate output directory required")
    if not shutil.which("tesseract"):
        raise RuntimeError("install Tesseract and requested language models first")
    output.mkdir(parents=True,exist_ok=True)
    journal=output/"pages.jsonl"
    if journal.exists() and not resume:
        raise FileExistsError("output already exists; choose a new directory or --resume")
    version=subprocess.run(["tesseract","--version"],capture_output=True,check=True).stdout.decode().splitlines()[0]
    listing=subprocess.run(["tesseract","--list-langs"],capture_output=True,text=True,check=True)
    match=re.search(r'List of available languages in "([^"]+)"',listing.stdout+listing.stderr)
    if not match:
        raise RuntimeError("cannot identify actual OCR language files for reproducible resume")
    language_hashes={name:file_sha256(Path(match.group(1))/(name+".traineddata")) for name in config.language.split("+")}
    header={"schema":"tac-ocr001.document.v1","source_sha256":file_sha256(source),
            "config":asdict(config),"engine_version":version,"language_weights_sha256":language_hashes}
    started=time.perf_counter()
    index=ReferenceIndex(output/"references.sqlite")
    try:
        # Rebuild from the authoritative journal, clearing any entry committed
        # just before a previous process crashed while writing that page.
        index.db.execute("DELETE FROM pages")
        index.db.commit()
        completed=_recover_pages(journal,header,index) if resume else 0
        recovered=completed
        if not journal.exists():
            with journal.open("w",encoding="utf-8") as stream:
                stream.write(json.dumps(header,ensure_ascii=False)+"\n")
        with journal.open("a",encoding="utf-8") as stream:
            for page in bounded_map(iter_pages(source,config,skip=completed),config):
                index.add(page["page"],page["text"])
                stream.write(json.dumps(page,ensure_ascii=False,separators=(",",":"))+"\n")
                stream.flush()
                os.fsync(stream.fileno())
                completed+=1
        render_markdown(journal,output/"document.md")
        result={"schema":"tac-ocr001.pipeline-run.v1","pages":completed,"resumed_pages":recovered,
                "new_pages":completed-recovered,"elapsed_seconds":time.perf_counter()-started,
                "max_inflight_pages":config.workers,"max_rendered_pixels_per_page":config.max_pixels,
                "backend":"tesseract-pretrained-or-explicit-pdf-text-layer",
                "new_optimizer_updates":0,"custom_neural_ocr_trained":False,
                "qualification":"software_run_not_benchmark_qualification",
                "memory_scope":"bounded rendered pages; disk outputs/index and PDF metadata grow with document length",
                "source_sha256":header["source_sha256"],"config":asdict(config)}
        (output/"run.json").write_text(json.dumps(result,indent=2)+"\n")
        return result
    finally:
        index.close()
