from pathlib import Path

def test_unified_governance_documents_exist():
    root = Path(__file__).parents[1]
    assert (root / 'docs' / 'RESEARCH_GOVERNANCE.md').is_file()
    assert (root / 'docs' / 'RESEARCH_LANE_REGISTRY.md').is_file()
    assert (root / 'docs' / 'RESEARCH_RUN_GATES_V2.md').is_file()
    assert (root / '.github' / 'workflows' / 'unified-research-gates.yml').is_file()

def test_unified_governance_contains_continuity_and_integrity_rules():
    root = Path(__file__).parents[1]
    c = (root / 'docs' / 'RESEARCH_GOVERNANCE.md').read_text()
    required = ['Continuity before novelty','Historical evidence','Universal gate','Security','Leakage','Verification','Linting','Ablation','Invalid runs','Universal future rule']
    assert all(x in c for x in required)

def test_lane_registry_keeps_current_portfolio_visible():
    root = Path(__file__).parents[1]
    c = (root / 'docs' / 'RESEARCH_LANE_REGISTRY.md').read_text()
    for lane in ['TACOSM-M1-RELEVANCE','TACOSM-M2-SELECTIVE-COMPUTE','TACOSM-PLM-ACTIVE-EVIDENCE','TACOSM-PLM-INTEGRATED-E2E','TACOSM-MTSK-BACTERIAL']:
        assert lane in c