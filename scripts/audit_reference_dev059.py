"""Evaluate the public and privileged reference on DEV-059's shared test worlds."""
import argparse
from audit_evidence_gaps_dev059 import reference_cells, summarize_rows
from run_learned_stack_dev055 import write

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cells = reference_cells(5591003)
    write(args.output, {'seed': 5591003, 'cells': cells,
                      'summary': summarize_rows([r for c in cells for r in c['rows']])})
