from datetime import UTC, datetime

from opencat3d.io.cache import StageCache, compute_stage_key, hash_dict
from opencat3d.schemas import StageManifest, StageStatus
from opencat3d.versioning import __version__


def test_hash_dict_is_order_independent() -> None:
    assert hash_dict({"a": 1, "b": 2}) == hash_dict({"b": 2, "a": 1})


def test_stage_cache_hit_round_trip(tmp_path) -> None:
    input_hash = hash_dict({"input": "a"})
    cfg_hash = hash_dict({"lr": 1e-3})
    cache_key = compute_stage_key("ingest", input_hash, cfg_hash, __version__)
    cache = StageCache(tmp_path)
    manifest = StageManifest(
        stage_name="ingest",
        status=StageStatus.COMPLETE,
        input_hash=input_hash,
        config_hash=cfg_hash,
        code_version=__version__,
        random_seed=20260625,
        completed_at=datetime.now(UTC),
    )

    path = cache.write_manifest(manifest)

    assert path.exists()
    assert manifest.cache_key == cache_key
    assert cache.is_hit("ingest", input_hash, cfg_hash, __version__)
    assert not cache.is_hit("ingest", hash_dict({"input": "b"}), cfg_hash, __version__)

