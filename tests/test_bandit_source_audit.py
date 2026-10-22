"""Source integrity negatives, with generated files used only as unit fixtures."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('bandit_source_audit',Path(__file__).parents[1]/'scripts/audit_bandit_source.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_plausible_columns_cannot_replace_authentic_publisher_hash(tmp_path):
    source = tmp_path/'invented.csv'
    source.write_text('timestamp,item_id,position,click,propensity_score,user_feature_0\n0,1,1,0,.5,2\n')
    result = module.audit(source)
    assert result['required_bandit_fields_present']
    assert not result['pinned_publisher_bytes_verified']
    assert not result['one_step_ope_schema_eligible']
    assert not result['persistent_memory_study_eligible']


def test_injected_entity_id_does_not_make_publisher_data_longitudinal(tmp_path):
    source = tmp_path/'invented.csv'
    source.write_text('timestamp,item_id,position,click,propensity_score,user_id\n0,1,1,0,.5,fake\n')
    result = module.audit(source)
    assert not result['persistent_memory_study_eligible']
    assert not result['pinned_publisher_bytes_verified']
