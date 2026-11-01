from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import PipelineConfig, ReferenceIndex, run_document


def main(argv=None):
    parser=argparse.ArgumentParser(description="Streaming OCR; custom TAC candidate remains untrained")
    sub=parser.add_subparsers(dest="command",required=True)
    run=sub.add_parser("transcribe")
    run.add_argument("input",type=Path)
    run.add_argument("--output",type=Path,required=True)
    run.add_argument("--workers",type=int,default=2)
    run.add_argument("--dpi",type=int,default=200)
    run.add_argument("--max-pixels",type=int,default=8_000_000)
    run.add_argument("--language",default="eng")
    run.add_argument("--psm",type=int,default=3)
    run.add_argument("--mode",choices=["ocr","auto"],default="ocr")
    run.add_argument("--resume",action="store_true")
    search=sub.add_parser("search")
    search.add_argument("output",type=Path)
    search.add_argument("query")
    search.add_argument("--max-bytes",type=int,default=8192)
    profile=sub.add_parser("profile")
    profile.add_argument("--output",type=Path,required=True)
    profile.add_argument("--steps",type=int,default=32)
    profile.add_argument("--threads",type=int,default=1)
    args=parser.parse_args(argv)
    if args.command=="transcribe":
        config=PipelineConfig(workers=args.workers,dpi=args.dpi,max_pixels=args.max_pixels,
                              language=args.language,psm=args.psm,mode=args.mode)
        result=run_document(args.input,args.output,config,resume=args.resume)
    elif args.command=="search":
        path=args.output/"references.sqlite"
        if not path.is_file():
            parser.error("an existing OCR output directory is required")
        index=ReferenceIndex(path)
        try:
            result=index.search(args.query,max_bytes=args.max_bytes)
        finally:
            index.close()
    else:
        from .profile import profile_candidates
        result=profile_candidates(args.output,steps=args.steps,threads=args.threads)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
