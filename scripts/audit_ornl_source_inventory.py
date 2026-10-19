"""Inventory public measured-fault files; read headers only, never score outcomes."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile


def run(directory, output):
    root = Path(directory)
    expected = {
        'dataset.zip': '1e6cd3df991abf865f7567e6cc0ac246968299eb02dfe9464d887727f930dde0',
        'curateddatasetdetaildescription.pdf': 'fca4ed23265c244434ee40ba45127fe8a9d166a246fc86e1d90b6781521755a4'}
    files = {}
    for name, sha in expected.items():
        with (root/name).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert actual == sha, name
        files[name] = dict(sha256=actual, bytes=(root/name).stat().st_size,
            url='https://data.openei.org/files/909/'+name)
    members, schemas, kinds = [], {}, Counter()
    with ZipFile(root/'dataset.zip') as archive:
        assert archive.testzip() is None
        for member in archive.infolist():
            assert member.filename.endswith('.csv') and '/' not in member.filename
            kind = 'weather' if member.filename.startswith('Weather_') else 'equipment'
            kinds[kind] += 1
            with archive.open(member) as stream:
                header = stream.readline().decode('utf-8-sig').strip()
            columns = next(csv.reader(io.StringIO(header)))
            sha = hashlib.sha256(header.encode()).hexdigest()
            schemas.setdefault(sha, dict(first_line_columns=columns, files=[]))['files'].append(member.filename)
            members.append(dict(name=member.filename, uncompressed_bytes=member.file_size,
                                kind=kind, first_line_sha256=sha))
    result = dict(status='SOURCE_INVENTORY_ONLY_NO_MODEL',
        official_source='https://data.openei.org/submissions/909',
        doi='10.25984/1862682', license='CC-BY-4.0',
        license_url='https://creativecommons.org/licenses/by/4.0/', files=files,
        member_counts=dict(kinds), members=members, first_line_schemas=schemas,
        sensor_data_rows_parsed=False, csv_first_lines_read=True,
        limits='Measured ORNL FRP fault-injection experiments, not naturally occurring factory repair logs. Publisher description says49 scenarios; downloaded archive contains63 equipment files plus5 weather files. Reconcile this before selecting an experimental cohort. Weather files begin with logger metadata, not measurement headers. No actual repair/inspection prices or independent field repair verification established. No model, heldout score, causal effect or commercial qualification.')
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], member_counts=result['member_counts'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    run(args.directory, args.output)
