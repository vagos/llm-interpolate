import json
import click
import pytest
import sqlite_utils
import numpy as np
from llm.cli import cli
from llm import Collection
from click.testing import CliRunner


@pytest.fixture
def db_path(tmpdir):
    db_path = tmpdir / "data.db"
    db = sqlite_utils.Database(str(db_path))
    collection = Collection("entries", db, model_id="simple-embeddings")
    collection.embed_multi(
        [
            (1, "one word"),
            (2, "two words"),
            (3, "three thing"),
            (4, "fourth thing"),
            (5, "fifth thing"),
            (6, "sixth thing"),
            (7, "seventh thing"),
            (8, "eighth thing"),
            (9, "ninth thing"),
            (10, "tenth thing"),
        ],
        store=True,
    )
    return db_path


@pytest.mark.parametrize("n", (2, 5, 10))
def test_interpolate_linear(db_path, n):
    db = sqlite_utils.Database(str(db_path))
    assert db["embeddings"].count == 10
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "interpolate",
            "entries",
            "1",
            "10",
            "-n",
            str(n),
            "--method",
            "linear",
            "--database",
            str(db_path),
        ],
    )
    assert result.exit_code == 0, result.output
    points = json.loads(result.output)
    assert len(points) == n
    assert points[-1] == "10"


def test_interpolate_linear_no_db_env(monkeypatch, tmpdir):
    db_path = tmpdir / "data.db"
    db = sqlite_utils.Database(str(db_path))
    collection = Collection("entries", db, model_id="simple-embeddings")
    collection.embed_multi(
        [
            (1, "one word"),
            (2, "two words"),
            (3, "three thing"),
            (4, "fourth thing"),
            (5, "fifth thing"),
            (6, "sixth thing"),
            (7, "seventh thing"),
            (8, "eighth thing"),
            (9, "ninth thing"),
            (10, "tenth thing"),
        ],
        store=True,
    )
    monkeypatch.setenv("LLM_EMBEDDINGS_DB", str(db_path))
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "interpolate",
            "entries",
            "1",
            "10",
            "-n",
            "5",
            "--method",
            "linear",
        ],
    )
    assert result.exit_code == 0, result.output
    points = json.loads(result.output)
    assert len(points) == 5
    assert points[-1] == "10"
