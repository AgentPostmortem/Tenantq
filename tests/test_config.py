"""Settings.from_env() validation of numeric environment variables."""

from __future__ import annotations

import pytest

from tenantq.config import Settings


@pytest.mark.parametrize("name", ["TENANTQ_HNSW_M", "TENANTQ_HNSW_EF", "TENANTQ_DENSE_DIM"])
def test_from_env_rejects_malformed_int(monkeypatch, name):
    monkeypatch.setenv(name, "abc")
    with pytest.raises(ValueError, match=name):
        Settings.from_env()
