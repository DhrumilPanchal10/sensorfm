"""Task 1.3: LOTSA reader, cleaning, decontamination, sharding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from sensorfm.data.build import BuildStats, build_records, fake_lotsa_source, synthetic_source
from sensorfm.data.dedup import WindowIndex, name_is_blocked
from sensorfm.data.filter import DropStats, clean_series
from sensorfm.data.lotsa import iter_arrow_file, load_selection
from sensorfm.data.shard import ShardReader, ShardRecord, load_manifest, split_of, write_corpus


def _write_stream(path: Path, table: pa.Table) -> None:
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table, max_chunksize=2)


def test_iter_arrow_univariate(tmp_path: Path) -> None:
    t = pa.table({
        "item_id": ["a", "b", "c"],
        "start": pa.array([0, 0, 0], pa.timestamp("s")),
        "freq": ["H", "H", "H"],
        "target": pa.array([[1.0, 2.0], [3.0, None, 5.0], [6.0]], pa.list_(pa.float32())),
    })  # fmt: skip
    p = tmp_path / "u.arrow"
    _write_stream(p, t)
    out = list(iter_arrow_file(p, "ds", "energy"))
    assert [s.item_id for s in out] == ["a", "b", "c"]
    assert out[1].values.dtype == np.float32
    assert np.isnan(out[1].values[1])
    assert out[0].domain == "energy"


def test_iter_arrow_multivariate_splits_variates(tmp_path: Path) -> None:
    t = pa.table({
        "item_id": ["m"],
        "start": pa.array([0], pa.timestamp("s")),
        "freq": ["5T"],
        "target": pa.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], pa.list_(pa.list_(pa.float32()))),
    })  # fmt: skip
    p = tmp_path / "m.arrow"
    _write_stream(p, t)
    out = list(iter_arrow_file(p, "ds"))
    assert [s.item_id for s in out] == ["m#v0", "m#v1"]
    np.testing.assert_array_equal(out[1].values, [4.0, 5.0, 6.0])


def test_selection_config_loads_and_is_sane() -> None:
    cfg = load_selection("configs/data/lotsa_selection.yaml")
    assert cfg.repo_id == "Salesforce/lotsa_data"
    assert len(cfg.subsets) > 20
    assert not any(name_is_blocked(s.name) for s in cfg.subsets)
    assert {s.domain for s in cfg.subsets} >= {"energy", "transport", "climate", "industrial"}


# --- cleaning ---------------------------------------------------------------------------------


def test_clean_drops_short_constant_missing() -> None:
    st = DropStats()
    assert clean_series(np.ones(500, np.float32) * 2, 128, 0.2, 32, st) == []
    assert clean_series(np.arange(50, dtype=np.float32), 128, 0.2, 32, st) == []
    x = np.arange(500, dtype=np.float32)
    x[::2] = np.nan
    assert clean_series(x, 128, 0.2, 32, st) == []
    assert (
        np.full(10, np.nan).size
        and clean_series(np.full(300, np.nan, np.float32), 128, 0.2, 32, st) == []
    )
    assert st.reasons == {"constant": 1, "too_short": 1, "too_missing": 1, "all_missing": 1}


def test_clean_imputes_short_gaps_linearly_and_trims_edges() -> None:
    x = np.arange(300, dtype=np.float32)
    x[:5] = np.nan
    x[100:110] = np.nan
    x[-3:] = np.nan
    st = DropStats()
    (seg,) = clean_series(x, 128, 0.2, 32, st)
    np.testing.assert_allclose(seg, np.arange(5, 297, dtype=np.float32))
    assert st.imputed_points == 10


def test_clean_splits_at_long_gaps() -> None:
    x = np.sin(np.arange(1000, dtype=np.float32) / 7)
    x[400:450] = np.nan  # 50 > max_gap
    segs = clean_series(x, 128, 0.2, 32)
    assert [len(s) for s in segs] == [400, 550]
    assert all(np.isfinite(s).all() for s in segs)


def test_clean_treats_inf_as_missing() -> None:
    x = np.sin(np.arange(400, dtype=np.float32))
    x[10] = np.inf
    (seg,) = clean_series(x, 128, 0.2, 32)
    assert np.isfinite(seg).all()


# --- decontamination --------------------------------------------------------------------------


def test_name_blocklist() -> None:
    for n in ("ETTh1", "ettm2", "CMAPSS_FD001", "ucr_anomaly", "cwru_bearing"):
        assert name_is_blocked(n)
    for n in ("PEMS03", "buildings_900k", "weather", "kdd2022", "lcl"):
        assert not name_is_blocked(n)


def test_window_index_catches_shifted_scaled_copy() -> None:
    rng = np.random.default_rng(0)
    ref = np.cumsum(rng.standard_normal(5000))
    idx = WindowIndex(window=64)
    idx.add_reference(ref)
    leaked = np.concatenate([rng.standard_normal(37), 3.5 * ref[1000:1300] + 10.0])
    assert idx.overlaps(leaked)
    unrelated = np.cumsum(rng.standard_normal(5000))
    assert not idx.overlaps(unrelated)


def test_window_index_ignores_constant_windows() -> None:
    idx = WindowIndex(window=64)
    idx.add_reference(np.zeros(500))
    assert len(idx) == 0
    assert not idx.overlaps(np.zeros(500))


# --- sharding ---------------------------------------------------------------------------------


def test_split_is_stable_and_by_key() -> None:
    keys = [f"ds/item{i}" for i in range(20000)]
    splits = [split_of(k, 0.05) for k in keys]
    assert splits == [split_of(k, 0.05) for k in keys]
    assert 0.04 < splits.count("val") / len(keys) < 0.06


def test_shard_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    recs = [
        ShardRecord(f"d{i % 3}/s{i}", f"d{i % 3}", "energy" if i % 2 else "climate", "H",
                    rng.standard_normal(int(rng.integers(100, 2000))).astype(np.float32))
        for i in range(300)
    ]  # fmt: skip
    m = write_corpus(recs, tmp_path, shard_bytes=64_000, val_frac=0.1, seed=0)
    assert load_manifest(tmp_path)["manifest_sha256"] == m["manifest_sha256"]
    tr, va = m["splits"]["train"], m["splits"]["val"]
    assert tr["n_series"] + va["n_series"] == 300
    assert len(tr["shards"]) > 1
    by_key = {r.key: r for r in recs}
    seen = 0
    for split in ("train", "val"):
        for meta in m["splits"][split]["shards"]:
            rd = ShardReader(tmp_path / split, meta)
            for i in range(len(rd)):
                s = rd.series(i)
                ds = m["datasets"][rd.dataset[i]]
                matches = [r for r in by_key.values() if r.dataset == ds and len(r.values) == len(s)
                           and np.array_equal(r.values, s)]  # fmt: skip
                assert len(matches) == 1
                assert split_of(matches[0].key, 0.1) == split
                seen += 1
    assert seen == 300
    assert tr["n_obs"] + va["n_obs"] == sum(len(r.values) for r in recs)


# --- pipeline ---------------------------------------------------------------------------------


def test_build_pipeline_caps_blocks_and_decontaminates() -> None:
    rng = np.random.default_rng(3)
    secret = np.cumsum(rng.standard_normal(3000)).astype(np.float32)
    ref = WindowIndex()
    ref.add_reference(secret)

    from sensorfm.data.lotsa import RawSeries

    def leaky() -> list[RawSeries]:
        clean = list(fake_lotsa_source("weather", "climate", 20, 0))
        return [*clean, RawSeries("weather", "copy", "H", secret[500:2000] * 2, "climate")]

    st = BuildStats()
    recs = list(build_records(
        [("ETTh1", None, fake_lotsa_source("ETTh1", "energy", 5, 0)),
         ("weather", None, leaky()),
         ("PEMS03", 2000, fake_lotsa_source("PEMS03", "transport", 50, 1))],
        min_length=128, max_missing_frac=0.2, max_gap=32, reference=ref, stats=st,
    ))  # fmt: skip
    assert st.blocked_datasets == ["ETTh1"]
    assert st.decontaminated["weather"] == 1
    assert "PEMS03" in st.capped_datasets
    assert all(r.dataset != "ETTh1" for r in recs)
    assert all(len(r.values) >= 128 and np.isfinite(r.values).all() for r in recs)
    assert st.per_dataset["weather"].reasons["constant"] >= 1


def test_synthetic_source_records() -> None:
    recs = list(build_records([("synthetic", None, synthetic_source(8, 512, 0))], 128, 1.0, 32))
    assert len(recs) == 8
    assert {r.domain for r in recs} == {"synthetic"}


@pytest.mark.parametrize("n", [0])
def test_empty_corpus_writes_manifest(tmp_path: Path, n: int) -> None:
    m = write_corpus([], tmp_path, 1000, 0.1, 0)
    assert m["splits"]["train"]["n_series"] == n


def test_write_corpus_refuses_existing_shards(tmp_path: Path) -> None:
    rec = ShardRecord("d/s", "d", "x", "H", np.arange(200, dtype=np.float32))
    write_corpus([rec], tmp_path, 1000, 0.0, 0)
    with pytest.raises(FileExistsError):
        write_corpus([rec], tmp_path, 1000, 0.0, 0)


def test_window_index_ignores_low_information_spikes() -> None:
    """Regression: flat-with-one-spike windows matched across unrelated real datasets."""
    ref = np.zeros(1000)
    ref[500] = 7.0
    idx = WindowIndex(window=64)
    idx.add_reference(
        np.concatenate([ref, np.cumsum(np.random.default_rng(0).standard_normal(500))])
    )
    other = np.full(640, 3.0)
    other[100] = 50.0
    assert not idx.overlaps(other)
