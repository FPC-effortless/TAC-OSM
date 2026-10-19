"""Public repair-source coverage only; never inspect outcome distributions or fit."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path


def run(directory, output):
    root = Path(directory)
    names = ('floodchlorinationsurvey.csv', 'data-raw_dictionary.csv', 'DESCRIPTION', 'LICENSE.md', 'data_processing.R')
    hashes = {}
    for name in names:
        with (root/name).open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert 'License: CC BY 4.0' in (root/'DESCRIPTION').read_text()
    with (root/'floodchlorinationsurvey.csv').open(newline='') as stream:
        reader = csv.DictReader(stream); rows = list(reader); columns = reader.fieldnames
    missing = lambda value: value is None or value.strip() in ('', 'NA')
    required = ('water_point_name', 'chlorination_rehab_date', 'functional_status_before',
                'functionality_after', 'time_to_fill_20l_bucket', 'strokes_to_yield_water')
    assert all(name in columns for name in required)
    coverage = {name: sum(not missing(row[name]) for row in rows) for name in required}
    replacements = [name for name in columns if name.startswith('replaced_')]
    assets = Counter(row['water_point_name'].strip().casefold() for row in rows
                     if not missing(row['water_point_name']))
    result = dict(status='SOURCE_COVERAGE_ONLY_NO_MODEL',
        source='https://github.com/openwashdata/floodchlorinationsurvey',
        source_commit='a2755213c841fdfc7b65aaaff0b91016296f2985',
        source_data_path='inst/extdata/floodchlorinationsurvey.csv',
        source_doi='10.5281/zenodo.15846578', license='CC-BY-4.0',
        attribution='Emmanuel Mhango and contributors, floodchlorinationsurvey0.1.0, USAID flood response survey, Mulanje Malawi2019–2020',
        files_sha256=hashes, rows=len(rows), columns=columns, present_counts=coverage,
        replacement_columns=len(replacements), distinct_normalized_waterpoint_names=len(assets),
        names_with_multiple_rows=sum(count > 1 for count in assets.values()),
        rows_with_any_recorded_replacement_field=sum(any(not missing(row[name]) for name in replacements) for row in rows),
        cost_like_columns=[name for name in columns if any(term in name.lower() for term in ('cost', 'price', 'payment', 'expense'))],
        outcomes_distribution_inspected=False, model_fitted=False,
        public_source_page_preview_examples_seen=True,
        future_sealed_evaluation_requires_excluding_preview_asset_groups=True,
        limits='Actual field-rehabilitation survey with reported component actions and measured postrepair flow proxies. Missing prices remain unknown; no actual repair/inspection cost contract. Collector independence and followup timing are not established. A submission date is not an independently measured outcome date. Replacement counts may be missing rather than zero. Repeated names are not reliable unique asset IDs. Completion text is not binary proof of physical restoration. Not industrial factory equipment. No causal repair benefit, profitable policy or model accuracy inferred. Any fit needs a new source-frozen protocol, asset/time holdout and leakage controls; do not use postrepair fields to predict pre-action choices. No linked photos requested.')
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('status','rows','replacement_columns','present_counts','cost_like_columns')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    run(args.directory, args.output)
