def test_address_baseline_contract():
    import json
    from pathlib import Path
    c=json.loads(Path("contracts/TACOSM-ADDRESS-BASELINE-001.json").read_text())
    assert c["status"]=="pre-registered"
    assert c["protocol"]["primary_dimension"]==16
    assert c["protocol"]["memory_sizes"]==[3,8,16,32,64,128,256]
