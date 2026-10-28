#!/usr/bin/env python3
"""Preserve/restore original archive bytes; this command never executes models."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.original_archive_shards import split_original,restore_original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    preserve=commands.add_parser('preserve')
    preserve.add_argument('--archive',type=Path,required=True)
    preserve.add_argument('--destination',type=Path,required=True)
    restore=commands.add_parser('restore')
    restore.add_argument('--directory',type=Path,required=True)
    restore.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='preserve':result=split_original(args.archive,args.destination)
    else:result=restore_original(args.directory,args.output)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
