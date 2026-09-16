"""CLI behaviour: option validation surfaces clean usage errors."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tenantq.cli import app

runner = CliRunner()


@pytest.mark.parametrize("bad_mode", ["hybird", "HYBRID", "dense "])
def test_search_rejects_unknown_mode(bad_mode):
    result = runner.invoke(app, ["search", "neural network", "--tenant", "acme", "--mode", bad_mode])
    assert result.exit_code != 0
    assert result.exception is not None
    message = str(result.exception)
    assert "hybrid" in message
    assert "is not one of" in message
