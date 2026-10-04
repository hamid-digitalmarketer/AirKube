import logging
import re
from ml.kg_schemas import ExtractionResult
from ml.kg_utils import get_connector

logger = logging.getLogger("kg_ingestion")

ALLOWED_REL_TYPES = frozenset({"BELONGS_TO", "PRODUCED_BY", "SERVES"})
_REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

# Expected (source label, target label) for each allowed relationship type.
# Entity.id is only unique per-type (see ml/kg_schemas.py), not across all
# node types, so the MATCH for a relationship must be scoped by label on
# both sides - otherwise an id collision between e.g. a Model and a
# Deployment can splice an edge onto the wrong node (#16).
REL_TYPE_LABELS = {
    "BELONGS_TO": ("Run", "Experiment"),
    "PRODUCED_BY": ("Model", "Run"),
    "SERVES": ("Deployment", "Model"),
}


def safe_rel_type(rel_type):
    """Return a backtick-escaped relationship type, or raise ValueError.

    Relationship types cannot be parameterized in Cypher, so the value is
    checked against a strict allowlist and regex before interpolation.
    """
    if (not isinstance(rel_type, str) or not _REL_TYPE_RE.fullmatch(rel_type)
            or rel_type not in ALLOWED_REL_TYPES):
        raise ValueError(f"Invalid relationship type: {rel_type!r}")
    return "`" + rel_type.replace("`", "``") + "`"

def ingest_graph(extraction: ExtractionResult, model_version: str = "unknown"):
    """
    Ingests the validated MLOps extraction result into Neo4j.
    """
    connector = get_connector()
    
    # 1. Ingest Nodes
    logger.info("Ingesting nodes...")
    
    # Models
    model_query = """
    UNWIND $batch AS mapped
    MERGE (m:Model {id: mapped.id})
    SET m.name = mapped.name,
        m.version = mapped.version,
        m.framework = mapped.framework,
        m.description = mapped.description,
        m.last_updated = datetime()
    """
    models_data = [d.dict() for d in extraction.models]
    if models_data:
        connector.run_write_query(model_query, {"batch": models_data})

    # Experiments
    experiment_query = """
    UNWIND $batch AS mapped
    MERGE (e:Experiment {id: mapped.id})
    SET e.name = mapped.name,
        e.status = mapped.status,
        e.created_at = datetime()
    """
    experiments_data = [e.dict() for e in extraction.experiments]
    if experiments_data:
        connector.run_write_query(experiment_query, {"batch": experiments_data})
        
    # Runs
    # Metrics and Parameters are stored as JSON strings or properties
    run_query = """
    UNWIND $batch AS mapped
    MERGE (r:Run {id: mapped.id})
    SET r.name = mapped.name,
        r.status = mapped.status,
        r.metrics = mapped.metrics,  
        r.parameters = mapped.parameters
    """
    # Note: Neo4j can store maps directly if using APOC, but standard cypher handles simple maps or needs serialization.
    # For now we assume standard driver map support.
    runs_data = [r.dict() for r in extraction.runs]
    if runs_data:
        connector.run_write_query(run_query, {"batch": runs_data})
        
    # Deployments
    deployment_query = """
    UNWIND $batch AS mapped
    MERGE (d:Deployment {id: mapped.id})
    SET d.name = mapped.name,
        d.cluster = mapped.cluster,
        d.image = mapped.image,
        d.replicas = mapped.replicas
    """
    deployments_data = [d.dict() for d in extraction.deployments]
    if deployments_data:
        connector.run_write_query(deployment_query, {"batch": deployments_data})

    # 2. Ingest Relationships
    logger.info("Ingesting relationships...")
    
    for rel in extraction.relationships:
        try:
            rel_type = safe_rel_type(rel.type)
        except ValueError:
            logger.warning("Skipping relationship with invalid type: %r", rel.type)
            continue

        # rel.type is validated against ALLOWED_REL_TYPES above, so it must
        # be present in REL_TYPE_LABELS too - but guard defensively anyway.
        labels = REL_TYPE_LABELS.get(rel.type)
        if labels is None:
            logger.warning("No label mapping for relationship type: %r", rel.type)
            continue
        src_label, tgt_label = labels

        query = f"""
        MATCH (s:{src_label} {{id: $source_id}}), (t:{tgt_label} {{id: $target_id}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r += $props,
            r.ingested_at = datetime()
        """
        connector.run_write_query(query, {
            "source_id": rel.source_id,
            "target_id": rel.target_id,
            "props": rel.properties
        })
    
    logger.info("Ingestion complete.")
    connector.close()

