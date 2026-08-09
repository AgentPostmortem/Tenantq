"""Search modes and metadata filtering."""

from __future__ import annotations

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


def test_build_filter_rejects_empty_tenant_id():
    import pytest
    from tenantq.search import UnscopedTenantError, build_filter

    with pytest.raises(UnscopedTenantError, match="not scoped"):
        build_filter("")
    with pytest.raises(UnscopedTenantError, match="not scoped"):
        build_filter("   ")


def test_build_filter_allows_tenant_id_with_internal_spaces():
    from tenantq.search import build_filter

    # Valid id that happens to contain spaces must not be stripped to empty
    f = build_filter("acme corp")
    assert f.must is not None
    assert f.must[0].match.value == "acme corp"


def test_search_rejects_empty_tenant_id(ingested, settings, embedder):
    import pytest
    from tenantq.search import UnscopedTenantError, search

    with pytest.raises(UnscopedTenantError, match="not scoped"):
        search(ingested, settings, embedder, "query", tenant_id="")
    with pytest.raises(UnscopedTenantError, match="not scoped"):
        search(ingested, settings, embedder, "query", tenant_id="  \t")
