"""Ingestion boundary checks."""

from __future__ import annotations

import pytest

from tenantq.embeddings import batched
from tenantq.ingest import ingest_documents


def test_batched_rejects_non_positive_size():
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        list(batched([1, 2, 3], 0))
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        list(batched([1, 2, 3], -1))


def test_ingest_documents_rejects_zero_batch_size(settings, embedder, dataset):
    from tenantq.client import make_client
    from tenantq.collection import recreate_collection

    client = make_client(settings)
    recreate_collection(client, settings)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        ingest_documents(client, settings, embedder, dataset.documents[:3], batch_size=0)
