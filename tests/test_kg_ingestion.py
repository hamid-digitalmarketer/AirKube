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
