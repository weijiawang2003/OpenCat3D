import pytest

from opencat3d.config import config_hash, load_config
from opencat3d.errors import LicenseGateError
from opencat3d.registry import assert_profile_allowed, load_model_registry


def test_default_config_loads_and_hashes() -> None:
    cfg = load_config(overrides=["project.project_id=test_cat"])

    assert cfg.project.project_id == "test_cat"
    assert len(config_hash(cfg)) == 64


def test_registry_license_gate() -> None:
    registry = load_model_registry()

    assert assert_profile_allowed("manual", "product", registry).provider == "opencat3d/manual"
    with pytest.raises(LicenseGateError):
        assert_profile_allowed("prima", "product", registry)

