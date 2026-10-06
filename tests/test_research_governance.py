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
    required_groups = [
        ('TACOSM-M1-RELEVANCE',),
        ('TACOSM-M2-SELECTIVE-COMPUTE',),
        ('TACOSM-PLM-ACTIVE-EVIDENCE',),
        ('TACOSM-E2E-001', 'TACOSM-PLM-INTEGRATED-E2E'),
        ('TACOSM-MTSK-BACTERIAL',),
    ]
    for aliases in required_groups:
        assert any(alias in c for alias in aliases)

def test_retrospective_audit_is_portfolio_wide():
    root = Path(__file__).parents[1]
    audit = (root / 'docs' / 'RETROSPECTIVE_SCIENCE_AUDIT_001.md').read_text()
    registry = (root / 'docs' / 'RESEARCH_LANE_REGISTRY.md').read_text()
    for anchor in [
        'TACOSM-REP-006', 'TACOSM-REP-007', 'TACOSM-REP-009',
        'TACOSM-C5-COVERAGE-FRONTIER-004',
        'TACOSM-C5-PERSISTENT-RELATIONAL-LOOP-001',
        'TACOSM-AXON-STRUCTMEANS-SECA-050-054',
        'TACOSM-VRS-001', 'TACOSM-E2E-003',
        'TACOSM-GCASM-015', 'TACOSM-GCASM-016C-R2',
    ]:
        assert anchor in audit or anchor in registry

def test_claim_ledger_has_unique_claim_ids_and_authoritative_e2e_disposition():
    root = Path(__file__).parents[1]
    c = (root / 'docs' / 'CLAIMS.md').read_text()
    import re
    ids = re.findall(r'^## (C\d+)\s+—', c, flags=re.MULTILINE)
    assert len(ids) == len(set(ids))
    assert ids.count('C12') == 1
    c12 = c[c.index('## C12'):c.index('## C13')]
    assert 'SUPERSEDED — VOID FOR BENCHMARK VALIDITY' in c12
    assert 'ignored the query entity argument' in c12

def test_current_docs_do_not_claim_temporal_measurement_is_unrun():
    root = Path(__file__).parents[1]
    roadmap = (root / 'docs' / 'ROADMAP.md').read_text()
    architecture = (root / 'docs' / 'ARCHITECTURE.md').read_text()
    assert 'TACOSM-TEMPORAL-001 preregistered, no result' not in roadmap
    assert '**new — not implemented anywhere**' not in architecture
    assert 'implemented in the hardened runtime' in architecture

def test_invalid_evidence_cannot_be_registered_as_a_positive_baseline():
    root = Path(__file__).parents[1]
    registry = (root / 'docs' / 'RESEARCH_LANE_REGISTRY.md').read_text()
    assert 'TACOSM-E2E-001 | INVALIDATED / VOID' in registry
    assert 'TACOSM-GCASM-016C-R1 | INVALIDATED' in registry
    assert 'Frontier-001/002/003 are **INVALIDATED**' in (root / 'docs' / 'RETROSPECTIVE_SCIENCE_AUDIT_001.md').read_text()
