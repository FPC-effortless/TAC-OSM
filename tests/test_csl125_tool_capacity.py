"""Vocab alternatives are independent *untrained* 20-layer architectures."""
from scripts.profile_csl125_r4_tool_vocab import capacity_screen


def test_vocabulary_capacity_counts_every_parameter_and_2bit_storage_lower_bound():
    x=capacity_screen()["vocabularies"]
    assert {k:v["params"] for k,v in x.items()}=={
        "8192":104_998_895,
        "16384":111_290_351,
        "32768":123_873_263,
    }
    assert x["32768"]["ideal_2bit_weight_bytes"]==30_968_316
    assert x["16384"]["ideal_2bit_weight_bytes"]==27_822_588
    assert x["8192"]["ideal_2bit_weight_bytes"]==26_249_724
    assert all(v["params"] <= 125_000_000 for v in x.values())
