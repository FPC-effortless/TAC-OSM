from pathlib import Path

from tac_osm.contract import load_contract


def test_bacterial_research_docs_and_contract_are_present():
    root = Path(__file__).parents[1]
    assert (root / "docs" / "BACTERIAL_ADAPTIVE_STATE_RESEARCH_PROGRAM.md").exists()
    assert (root / "docs" / "RESEARCH_RUN_GATES_V2.md").exists()
    assert (root / "contracts" / "TACOSM-BACTERIAL-ADAPTIVE-STATE-001.json").exists()


def test_bacterial_contract_is_well_formed_and_preregistered():
    contract = load_contract("TACOSM-BACTERIAL-ADAPTIVE-STATE-001")
    assert contract.status == "pre-registered"
    assert contract.primary_endpoint() == "exact_class_balanced_success"
    assert tuple(contract.seeds) == (0, 1, 2, 3, 4)
    assert tuple(contract.h_levels) == (8, 64, 256)
    assert [a.name for a in contract.arms] == [
        "no_state", "single_timescale", "mtsk"
    ]
    assert contract.check_consistency() == []


def test_bacterial_protocol_keeps_one_novel_block_boundary():
    text = (
        Path(__file__).parents[1]
        / "docs"
        / "BACTERIAL_ADAPTIVE_STATE_RESEARCH_PROGRAM.md"
    ).read_text()
    assert "Multi-Timescale Adaptive State Kernel (MTSK)" in text
    assert "No custom general-purpose matrix multiplication" in text
    assert "The surrounding system remains battle-tested" in text
