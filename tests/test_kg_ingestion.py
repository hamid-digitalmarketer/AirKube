import sys
import os
from unittest.mock import MagicMock

sys.path.append(os.getcwd())

import pytest
from ml import kg_ingestion
from ml.kg_schemas import ExtractionResult, Relationship


@pytest.mark.parametrize("bad", [
    "FOO}] DETACH DELETE t //", "belongs_to", "BELONGS_TO`", "", "UNKNOWN", "A B",
])
def test_safe_rel_type_rejects(bad):
    with pytest.raises(ValueError):
        kg_ingestion.safe_rel_type(bad)


def test_safe_rel_type_accepts_and_escapes():
    assert kg_ingestion.safe_rel_type("SERVES") == "`SERVES`"


def test_ingest_skips_injected_relationship(monkeypatch):
    conn = MagicMock()
    monkeypatch.setattr(kg_ingestion, "get_connector", lambda: conn)
    result = ExtractionResult(relationships=[
        Relationship(source_id="a", target_id="b", type="FOO}] DETACH DELETE t //"),
        Relationship(source_id="a", target_id="b", type="SERVES"),
    ])
    kg_ingestion.ingest_graph(result)
    queries = [c.args[0] for c in conn.run_write_query.call_args_list]
    assert len(queries) == 1
    assert "`SERVES`" in queries[0]
    assert not any("DETACH" in q for q in queries)


@pytest.mark.parametrize("rel_type,src_label,tgt_label", [
    ("BELONGS_TO", "Run", "Experiment"),
    ("PRODUCED_BY", "Model", "Run"),
    ("SERVES", "Deployment", "Model"),
])
def test_ingest_relationship_scopes_match_by_label(monkeypatch, rel_type, src_label, tgt_label):
    """
    Regression test for #16: the MATCH for each relationship type must be
    scoped by the expected source/target labels, not a bare {id: ...} match
    that could span every node type sharing that id.
    """
    conn = MagicMock()
    monkeypatch.setattr(kg_ingestion, "get_connector", lambda: conn)
    result = ExtractionResult(relationships=[
        Relationship(source_id="a", target_id="b", type=rel_type),
    ])
    kg_ingestion.ingest_graph(result)
    queries = [c.args[0] for c in conn.run_write_query.call_args_list]
    assert len(queries) == 1
    query = queries[0]
    assert f"(s:{src_label} {{id: $source_id}})" in query
    assert f"(t:{tgt_label} {{id: $target_id}})" in query


def test_rel_type_labels_cover_all_allowed_types():
    """Every allowlisted relationship type must have a label mapping, or
    ingestion would silently skip valid relationships."""
    assert kg_ingestion.ALLOWED_REL_TYPES <= set(kg_ingestion.REL_TYPE_LABELS.keys())
