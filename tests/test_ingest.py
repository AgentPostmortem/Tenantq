"""Ingestion input validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tenantq.embeddings import batched
from tenantq.ingest import ingest_documents


@pytest.mark.parametrize("batch_size", [0, -1])
def test_ingest_documents_rejects_batch_size_below_one(settings, batch_size):
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        ingest_documents(
            MagicMock(),
            settings,
            MagicMock(),
            documents=[],
            batch_size=batch_size,
        )


@pytest.mark.parametrize("size", [0, -3])
def test_batched_rejects_non_positive_size(size):
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        list(batched([1, 2, 3], size))
