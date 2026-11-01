from dataclasses import asdict
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from .pipeline import (PipelineConfig,ReferenceIndex,bounded_map,iter_pages,run_document)


def test_bounded_backpressure_and_order():
    produced=[]
    def source():
        for i in range(100):
            produced.append(i)
            yield {"page":i}
    config=PipelineConfig(workers=3)
    output=bounded_map(source(),config,lambda x,c:x)
    for expected in range(100):
        assert next(output)["page"]==expected
        assert len(produced)<=expected+3
    assert list(output)==[]


def test_search_budget_unicode_and_sql_injection(tmp_path):
    index=ReferenceIndex(tmp_path/"index.sqlite")
    try:
        for p in range(20):
            index.add(p,"invoice ₦100 Ibadan "+"中文 "*50)
        result=index.search('invoice"; DROP TABLE pages; --',limit=16,max_bytes=70)
        assert len(result)<=16
        assert sum(len(r["text"].encode()) for r in result)<=70
        assert index.search("Ibadan")
        assert index.search("'")==[]
        with pytest.raises(ValueError):
            index.search("invoice",max_bytes=1_000_000)
    finally:
        index.close()


def test_pdf_lazy_render_native_and_pixel_limit(tmp_path):
    import fitz
    path=tmp_path/"input.pdf"
    with fitz.open() as doc:
        for i in range(3):
            page=doc.new_page(width=1000,height=1000)
            page.insert_text((20,50),f"Page {i+1}")
        doc.save(path)
    first=next(iter_pages(path,PipelineConfig(max_pixels=100_000)))
    assert first["image"].width*first["image"].height<=101_000
    assert first["coordinate_unit"]=="pixel"
    native=list(iter_pages(path,PipelineConfig(mode="auto"),skip=1))
    assert len(native)==2 and native[0]["page"]==2
    assert all("native_words" in row and "image" not in row for row in native)


@pytest.mark.skipif(not shutil.which("tesseract"),reason="system OCR binary not installed")
def test_real_image_transcription_resume_and_torn_journal(tmp_path):
    source=Path(__file__).parent/"fixtures"/"phototest.tif"
    config=PipelineConfig(workers=1)
    output=tmp_path/"ocr"
    run=run_document(source,output,config)
    assert run["pages"]==1 and run["new_optimizer_updates"]==0
    lines=(output/"pages.jsonl").read_text().splitlines()
    page=json.loads(lines[1])
    assert "text" in page["text"] and len(page["words"])>10
    assert page["backend"]=="tesseract-pretrained"
    assert not page["confidence_is_calibrated"]
    before=(output/"document.md").read_bytes()
    with (output/"pages.jsonl").open("ab") as stream:
        stream.write(b'{"page":2')
    resumed=run_document(source,output,config,resume=True)
    assert resumed["new_pages"]==0 and resumed["resumed_pages"]==1
    assert (output/"document.md").read_bytes()==before
    assert len((output/"pages.jsonl").read_text().splitlines())==2
    with pytest.raises(FileExistsError):
        run_document(source,output,config)
    with pytest.raises(ValueError,match="mismatch"):
        run_document(source,output,PipelineConfig(workers=2),resume=True)


@pytest.mark.skipif(not shutil.which("tesseract"),reason="system OCR binary not installed")
def test_multipage_tiff_and_no_silent_page_dropping(tmp_path):
    source=Path(__file__).parent/"fixtures"/"phototest.tif"
    dest=tmp_path/"three.tif"
    with Image.open(source) as image:
        image.save(dest,save_all=True,append_images=[image.copy(),image.copy()])
    output=tmp_path/"out"
    run=run_document(dest,output,PipelineConfig(workers=2))
    assert run["pages"]==3
    with (output/"pages.jsonl").open() as stream:
        next(stream)
        assert [json.loads(line)["page"] for line in stream]==[1,2,3]


def test_worker_error_propagates():
    def fail(payload,c):
        raise RuntimeError("fixture OCR failure")
    with pytest.raises(RuntimeError,match="fixture"):
        list(bounded_map(iter([{"page":1}]),PipelineConfig(workers=1),fail))


@pytest.mark.parametrize("overrides",[{"workers":0},{"max_pixels":0},{"language":"eng; echo test"},{"mode":"unknown"}])
def test_config_rejects_unsafe_or_unbounded(overrides):
    with pytest.raises(ValueError):
        PipelineConfig(**overrides)
