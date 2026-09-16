"""Search modes and metadata filtering."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tenantq.search import search


def test_all_modes_return_hits(ingested, settings, embedder):
    for mode in ("dense", "sparse", "hybrid"):
        hits = search(ingested, settings, embedder, "neural network embedding", tenant_id="acme", mode=mode, limit=10)
        assert len(hits) > 0
        # scores are sorted descending
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)


def test_category_filter_restricts_results(ingested, settings, embedder, dataset):
    category = "networking"
    hits = search(
        ingested, settings, embedder, "router packet latency", tenant_id="acme",
        mode="hybrid", limit=20, category=category,
    )
    assert hits
    assert all(h.category == category for h in hits)


def test_created_at_range_filter(ingested, settings, embedder, dataset):
    from tenantq.search import build_filter  # noqa: F401  (exercised indirectly)

    acme = sorted([d for d in dataset.documents if d.tenant_id == "acme"], key=lambda d: d.created_at)
    lo = acme[2].created_at
    hi = acme[6].created_at
    hits = search(
        ingested, settings, embedder, "database index query", tenant_id="acme",
        mode="dense", limit=50, category=None, created_after=lo, created_before=hi,
    )
    # every returned doc must fall inside the requested time window
    by_id = {d.id: d for d in dataset.documents}
    for h in hits:
        assert lo <= by_id[h.id].created_at <= hi


@pytest.mark.parametrize("bad", ["", " ", "\t", "\n"])
def test_search_rejects_empty_or_whitespace_query(settings, bad):
    client = MagicMock()
    embedder = MagicMock()
    with pytest.raises(ValueError, match="query is required"):
        search(client, settings, embedder, bad, tenant_id="acme", mode="hybrid")
    embedder.embed_dense.assert_not_called()
    embedder.embed_sparse.assert_not_called()
    client.query_points.assert_not_called()


def test_search_accepts_query_with_internal_spaces(ingested, settings, embedder):
    hits = search(
        ingested,
        settings,
        embedder,
        " neural network ",
        tenant_id="acme",
        mode="dense",
        limit=5,
    )
    assert len(hits) > 0


@pytest.mark.parametrize("bad", ["", "   ", "\t"])
def test_build_filter_rejects_empty_tenant_id(bad):
    from tenantq.search import build_filter
    with pytest.raises(ValueError, match="not scoped"):
        build_filter(bad)


def test_build_filter_accepts_tenant_with_internal_spaces():
    from tenantq.search import build_filter
    f = build_filter("acme corp")
    assert f.must


def test_build_filter_rejects_impossible_range():
    from tenantq.search import build_filter
    with pytest.raises(ValueError, match="must be"):
        build_filter("acme", created_after=200, created_before=100)


def test_build_filter_allows_wide_open_range():
    from tenantq.search import build_filter
    f = build_filter("acme", created_after=100, created_before=200)
    assert f.must


@pytest.mark.parametrize("limit", [-1, 0, 10_000])
def test_search_rejects_out_of_range_limit(settings, limit):
    client = MagicMock()
    embedder = MagicMock()
    with pytest.raises(ValueError, match="limit must be within"):
        search(client, settings, embedder, "dense query", tenant_id="acme", mode="dense", limit=limit)
    client.query_points.assert_not_called()


@pytest.mark.parametrize("prefetch_limit", [-1, 0, 100_000])
def test_search_rejects_out_of_range_prefetch_limit(settings, prefetch_limit):
    client = MagicMock()
    embedder = MagicMock()
    with pytest.raises(ValueError, match="prefetch_limit must be within"):
        search(
            client, settings, embedder, "hybrid query",
            tenant_id="acme", mode="hybrid", prefetch_limit=prefetch_limit,
        )
    client.query_points.assert_not_called()
