from tac_osm.unified import Budget, IdentifiabilityReport, KnowledgeStore, MemoryWrite, Observation

def test_budget_defaults_are_fixed():
    b = Budget()
    assert (b.proposal_k, b.route_k, b.planning_depth) == (8, 1, 2)

def test_verified_commit_and_retraction():
    store = KnowledgeStore()
    store.commit([MemoryWrite("add", "op", (1.0, 2.0), 1.0, "verified")])
    assert "op" in store.knowledge.structures
    store.commit([MemoryWrite("invalidate", "op", confidence=1.0, reason="contradiction")])
    assert "op" not in store.knowledge.structures
    assert "op" in store.knowledge.invalidated

def test_observation_has_no_hidden_truth_fields():
    obs = Observation("language", (1.0, 2.0), "train:0", 0)
    assert not hasattr(obs, "gold_index")
    assert not hasattr(obs, "target_action")

def test_identifiability_is_first_class():
    report = IdentifiabilityReport(True, 0, 32)
    assert report.identifiable and report.collisions == 0
