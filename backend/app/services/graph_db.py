"""
Graph Database Service - KuzuDB Backend
Embedded graph database replacing remote graph APIs.
Provides node/edge CRUD, search, and graph management.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..config import Config
from ..utils.logger import get_logger

try:
    import kuzu
except ImportError:  # pragma: no cover - dependency availability is environment specific
    kuzu = None

logger = get_logger('mirofish.graph_db')


@dataclass
class GraphNode:
    """Node in the knowledge graph"""
    uuid_: str
    name: str
    labels: List[str] = field(default_factory=lambda: ["Entity"])
    summary: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid_,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "created_at": self.created_at,
        }


@dataclass
class GraphEdge:
    """Edge (relationship) in the knowledge graph"""
    uuid_: str
    name: str
    fact: str = ""
    fact_type: str = ""
    source_node_uuid: str = ""
    target_node_uuid: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    episodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid_,
            "name": self.name,
            "fact": self.fact,
            "fact_type": self.fact_type,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "attributes": self.attributes,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at,
            "episodes": self.episodes,
        }


@dataclass
class Episode:
    """Text episode added to the graph for processing"""
    uuid_: str
    data: str
    type: str = "text"
    processed: bool = False
    created_at: str = ""


class GraphDatabase:
    """
    KuzuDB-backed graph database.

    Nodes and edges are stored in KuzuDB, while graph metadata,
    ontology, and episode ingestion bookkeeping remain as JSON sidecars.
    """

    def __init__(self, base_path: Optional[str] = None):
        if kuzu is None:
            raise ImportError(
                "kuzu package is required for GraphDatabase. "
                "Install backend dependencies with `uv sync` or `pip install -r backend/requirements.txt`."
            )
        self.base_path = base_path or Config.GRAPH_DB_PATH
        os.makedirs(self.base_path, exist_ok=True)

    def _graph_dir(self, graph_id: str) -> str:
        return os.path.join(self.base_path, graph_id)

    def _db_dir(self, graph_id: str) -> str:
        return os.path.join(self._graph_dir(graph_id), "graph.kuzu")

    def _legacy_nodes_file(self, graph_id: str) -> str:
        return os.path.join(self._graph_dir(graph_id), "nodes.json")

    def _legacy_edges_file(self, graph_id: str) -> str:
        return os.path.join(self._graph_dir(graph_id), "edges.json")

    def _episodes_file(self, graph_id: str) -> str:
        return os.path.join(self._graph_dir(graph_id), "episodes.json")

    def _meta_file(self, graph_id: str) -> str:
        return os.path.join(self._graph_dir(graph_id), "meta.json")

    def _ontology_file(self, graph_id: str) -> str:
        return os.path.join(self._graph_dir(graph_id), "ontology.json")

    def _load_json(self, path: str, default: Any = None) -> Any:
        if not os.path.exists(path):
            return [] if default is None else default
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_json(self, path: str, data: Any):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _has_legacy_json_graph(self, graph_id: str) -> bool:
        return os.path.exists(self._legacy_nodes_file(graph_id)) or os.path.exists(self._legacy_edges_file(graph_id))

    def _migrate_legacy_json_graph(self, graph_id: str):
        """Import legacy JSON node/edge files into KuzuDB on first access."""
        if not self._has_legacy_json_graph(graph_id):
            return

        db_dir = self._db_dir(graph_id)
        if os.path.exists(db_dir):
            return

        logger.info(f"Migrating legacy JSON graph to KuzuDB: {graph_id}")
        conn = kuzu.Connection(kuzu.Database(db_dir))
        self._initialize_schema(conn)

        legacy_nodes = self._load_json(self._legacy_nodes_file(graph_id), default=[])
        legacy_edges = self._load_json(self._legacy_edges_file(graph_id), default=[])

        for node in legacy_nodes:
            conn.execute(
                """
                CREATE (n:Node {
                    uuid: $uuid,
                    name: $name,
                    labels: $labels,
                    summary: $summary,
                    attributes: $attributes,
                    created_at: $created_at
                })
                """,
                {
                    "uuid": node.get("uuid") or str(uuid.uuid4()),
                    "name": node.get("name", ""),
                    "labels": node.get("labels") or ["Entity"],
                    "summary": node.get("summary", ""),
                    "attributes": json.dumps(node.get("attributes") or {}, ensure_ascii=False),
                    "created_at": node.get("created_at", "") or datetime.now().isoformat(),
                },
            )

        for edge in legacy_edges:
            conn.execute(
                """
                MATCH (a:Node {uuid: $source_uuid}), (b:Node {uuid: $target_uuid})
                CREATE (a)-[:Edge {
                    uuid: $uuid,
                    name: $name,
                    fact: $fact,
                    fact_type: $fact_type,
                    attributes: $attributes,
                    created_at: $created_at,
                    valid_at: $valid_at,
                    invalid_at: $invalid_at,
                    expired_at: $expired_at,
                    episodes: $episodes
                }]->(b)
                """,
                {
                    "source_uuid": edge.get("source_node_uuid", ""),
                    "target_uuid": edge.get("target_node_uuid", ""),
                    "uuid": edge.get("uuid") or str(uuid.uuid4()),
                    "name": edge.get("name", ""),
                    "fact": edge.get("fact", ""),
                    "fact_type": edge.get("fact_type", "") or edge.get("name", ""),
                    "attributes": json.dumps(edge.get("attributes") or {}, ensure_ascii=False),
                    "created_at": edge.get("created_at", "") or datetime.now().isoformat(),
                    "valid_at": edge.get("valid_at") or "",
                    "invalid_at": edge.get("invalid_at") or "",
                    "expired_at": edge.get("expired_at") or "",
                    "episodes": edge.get("episodes") or [],
                },
            )

        logger.info(
            f"Legacy graph migration complete: {graph_id}, "
            f"nodes={len(legacy_nodes)}, edges={len(legacy_edges)}"
        )

    def _connect(self, graph_id: str):
        db_dir = self._db_dir(graph_id)
        if not os.path.exists(db_dir) and self._has_legacy_json_graph(graph_id):
            self._migrate_legacy_json_graph(graph_id)
        if not os.path.exists(db_dir):
            raise FileNotFoundError(f"Graph database not found: {graph_id}")
        db = kuzu.Database(db_dir)
        conn = kuzu.Connection(db)
        return conn

    def _initialize_schema(self, conn):
        conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Node(
                uuid STRING PRIMARY KEY,
                name STRING,
                labels STRING[],
                summary STRING,
                attributes STRING,
                created_at STRING
            )
            """
        )
        conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS Edge(
                FROM Node TO Node,
                uuid STRING,
                name STRING,
                fact STRING,
                fact_type STRING,
                attributes STRING,
                created_at STRING,
                valid_at STRING,
                invalid_at STRING,
                expired_at STRING,
                episodes STRING[]
            )
            """
        )

    def _decode_attributes(self, value: Any) -> Dict[str, Any]:
        if not value:
            return {}
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value)
        except Exception:
            return {}

    def _row_to_node(self, row: List[Any]) -> GraphNode:
        return GraphNode(
            uuid_=row[0],
            name=row[1],
            labels=row[2] or ["Entity"],
            summary=row[3] or "",
            attributes=self._decode_attributes(row[4]),
            created_at=row[5] or "",
        )

    def _row_to_edge(self, row: List[Any]) -> GraphEdge:
        return GraphEdge(
            uuid_=row[0],
            name=row[1],
            fact=row[2] or "",
            fact_type=row[3] or "",
            source_node_uuid=row[4] or "",
            target_node_uuid=row[5] or "",
            attributes=self._decode_attributes(row[6]),
            created_at=row[7] or "",
            valid_at=row[8] or None,
            invalid_at=row[9] or None,
            expired_at=row[10] or None,
            episodes=row[11] or [],
        )

    # ========== Graph Management ==========

    def create_graph(self, graph_id: str, name: str, description: str = "") -> str:
        """Create a new graph database"""
        graph_dir = self._graph_dir(graph_id)
        db_dir = self._db_dir(graph_id)
        os.makedirs(graph_dir, exist_ok=True)

        conn = kuzu.Connection(kuzu.Database(db_dir))
        self._initialize_schema(conn)

        meta = {
            "graph_id": graph_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
        }
        self._save_json(self._meta_file(graph_id), meta)
        self._save_json(self._episodes_file(graph_id), [])

        logger.info(f"Created graph: {graph_id} ({name})")
        return graph_id

    def delete_graph(self, graph_id: str):
        """Delete a graph and all its data"""
        graph_dir = self._graph_dir(graph_id)
        if os.path.exists(graph_dir):
            shutil.rmtree(graph_dir)
            logger.info(f"Deleted graph: {graph_id}")

    def graph_exists(self, graph_id: str) -> bool:
        return os.path.exists(self._meta_file(graph_id)) and (
            os.path.exists(self._db_dir(graph_id)) or self._has_legacy_json_graph(graph_id)
        )

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """Store ontology definition for the graph"""
        self._save_json(self._ontology_file(graph_id), ontology)
        logger.info(
            f"Set ontology for graph {graph_id}: "
            f"{len(ontology.get('entity_types', []))} entity types, "
            f"{len(ontology.get('edge_types', []))} edge types"
        )

    def get_ontology(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """Get the ontology definition for a graph"""
        path = self._ontology_file(graph_id)
        if os.path.exists(path):
            return self._load_json(path)
        return None

    # ========== Episode Management ==========

    def add_episode(self, graph_id: str, data: str, type: str = "text") -> Episode:
        """Add a text episode for processing. Returns the episode object."""
        ep = Episode(
            uuid_=str(uuid.uuid4()),
            data=data,
            type=type,
            processed=False,
            created_at=datetime.now().isoformat(),
        )
        episodes = self._load_json(self._episodes_file(graph_id), default=[])
        episodes.append({
            "uuid": ep.uuid_,
            "data": ep.data,
            "type": ep.type,
            "processed": ep.processed,
            "created_at": ep.created_at,
        })
        self._save_json(self._episodes_file(graph_id), episodes)
        return ep

    def add_episodes_batch(self, graph_id: str, texts: List[str]) -> List[Episode]:
        """Add multiple text episodes. Returns list of Episode objects."""
        episodes_data = self._load_json(self._episodes_file(graph_id), default=[])
        result = []
        now = datetime.now().isoformat()
        for text in texts:
            ep = Episode(
                uuid_=str(uuid.uuid4()),
                data=text,
                type="text",
                processed=False,
                created_at=now,
            )
            episodes_data.append({
                "uuid": ep.uuid_,
                "data": ep.data,
                "type": ep.type,
                "processed": ep.processed,
                "created_at": ep.created_at,
            })
            result.append(ep)
        self._save_json(self._episodes_file(graph_id), episodes_data)
        return result

    def mark_episode_processed(self, graph_id: str, episode_uuid: str):
        """Mark an episode as processed"""
        episodes = self._load_json(self._episodes_file(graph_id), default=[])
        for ep in episodes:
            if ep["uuid"] == episode_uuid:
                ep["processed"] = True
                break
        self._save_json(self._episodes_file(graph_id), episodes)

    def get_episode(self, graph_id: str, episode_uuid: str) -> Optional[Episode]:
        """Get episode by UUID"""
        episodes = self._load_json(self._episodes_file(graph_id), default=[])
        for ep in episodes:
            if ep["uuid"] == episode_uuid:
                return Episode(
                    uuid_=ep["uuid"],
                    data=ep["data"],
                    type=ep.get("type", "text"),
                    processed=ep.get("processed", False),
                    created_at=ep.get("created_at", ""),
                )
        return None

    # ========== Node Operations ==========

    def add_node(
        self,
        graph_id: str,
        name: str,
        labels: List[str],
        summary: str = "",
        attributes: Dict[str, Any] = None,
        node_uuid: Optional[str] = None,
    ) -> GraphNode:
        """Add a node. Deduplicates by name (case-insensitive)."""
        existing = self.get_node_by_name(graph_id, name)
        merged_labels = list(set(["Entity"] + (labels or [])))
        merged_attributes = attributes or {}

        if existing:
            if summary and len(summary) > len(existing.summary or ""):
                existing.summary = summary
            existing.labels = list(set(existing.labels + merged_labels))
            existing.attributes.update(merged_attributes)

            conn = self._connect(graph_id)
            conn.execute(
                """
                MATCH (n:Node {uuid: $uuid})
                SET n.labels = $labels,
                    n.summary = $summary,
                    n.attributes = $attributes
                RETURN n.uuid
                """,
                {
                    "uuid": existing.uuid_,
                    "labels": existing.labels,
                    "summary": existing.summary,
                    "attributes": json.dumps(existing.attributes, ensure_ascii=False),
                },
            )
            return existing

        now = datetime.now().isoformat()
        node = GraphNode(
            uuid_=node_uuid or str(uuid.uuid4()),
            name=name,
            labels=merged_labels,
            summary=summary,
            attributes=merged_attributes,
            created_at=now,
        )
        conn = self._connect(graph_id)
        conn.execute(
            """
            CREATE (n:Node {
                uuid: $uuid,
                name: $name,
                labels: $labels,
                summary: $summary,
                attributes: $attributes,
                created_at: $created_at
            })
            """,
            {
                "uuid": node.uuid_,
                "name": node.name,
                "labels": node.labels,
                "summary": node.summary,
                "attributes": json.dumps(node.attributes, ensure_ascii=False),
                "created_at": node.created_at,
            },
        )
        return node

    def get_node(self, graph_id: str, node_uuid: str) -> Optional[GraphNode]:
        """Get a node by UUID"""
        conn = self._connect(graph_id)
        result = conn.execute(
            """
            MATCH (n:Node {uuid: $uuid})
            RETURN n.uuid, n.name, n.labels, n.summary, n.attributes, n.created_at
            LIMIT 1
            """,
            {"uuid": node_uuid},
        )
        rows = result.get_all()
        return self._row_to_node(rows[0]) if rows else None

    def get_node_by_name(self, graph_id: str, name: str) -> Optional[GraphNode]:
        """Get a node by name (case-insensitive)"""
        conn = self._connect(graph_id)
        result = conn.execute(
            """
            MATCH (n:Node)
            WHERE lower(n.name) = lower($name)
            RETURN n.uuid, n.name, n.labels, n.summary, n.attributes, n.created_at
            LIMIT 1
            """,
            {"name": name.strip()},
        )
        rows = result.get_all()
        return self._row_to_node(rows[0]) if rows else None

    def get_all_nodes(self, graph_id: str) -> List[GraphNode]:
        """Get all nodes in a graph"""
        conn = self._connect(graph_id)
        result = conn.execute(
            """
            MATCH (n:Node)
            RETURN n.uuid, n.name, n.labels, n.summary, n.attributes, n.created_at
            """
        )
        return [self._row_to_node(row) for row in result.get_all()]

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[GraphEdge]:
        """Get all edges connected to a node"""
        conn = self._connect(graph_id)
        result = conn.execute(
            """
            MATCH (a:Node)-[e:Edge]->(b:Node)
            WHERE a.uuid = $uuid OR b.uuid = $uuid
            RETURN e.uuid, e.name, e.fact, e.fact_type, a.uuid, b.uuid,
                   e.attributes, e.created_at, e.valid_at, e.invalid_at, e.expired_at, e.episodes
            """,
            {"uuid": node_uuid},
        )
        return [self._row_to_edge(row) for row in result.get_all()]

    # ========== Edge Operations ==========

    def add_edge(
        self,
        graph_id: str,
        source_node_uuid: str,
        target_node_uuid: str,
        name: str,
        fact: str = "",
        fact_type: str = "",
        attributes: Dict[str, Any] = None,
        episode_uuid: Optional[str] = None,
    ) -> GraphEdge:
        """Add an edge between two nodes"""
        edge = GraphEdge(
            uuid_=str(uuid.uuid4()),
            name=name,
            fact=fact,
            fact_type=fact_type or name,
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
            attributes=attributes or {},
            created_at=datetime.now().isoformat(),
            episodes=[episode_uuid] if episode_uuid else [],
        )

        conn = self._connect(graph_id)
        conn.execute(
            """
            MATCH (a:Node {uuid: $source_uuid}), (b:Node {uuid: $target_uuid})
            CREATE (a)-[:Edge {
                uuid: $uuid,
                name: $name,
                fact: $fact,
                fact_type: $fact_type,
                attributes: $attributes,
                created_at: $created_at,
                valid_at: $valid_at,
                invalid_at: $invalid_at,
                expired_at: $expired_at,
                episodes: $episodes
            }]->(b)
            """,
            {
                "source_uuid": source_node_uuid,
                "target_uuid": target_node_uuid,
                "uuid": edge.uuid_,
                "name": edge.name,
                "fact": edge.fact,
                "fact_type": edge.fact_type,
                "attributes": json.dumps(edge.attributes, ensure_ascii=False),
                "created_at": edge.created_at,
                "valid_at": edge.valid_at or "",
                "invalid_at": edge.invalid_at or "",
                "expired_at": edge.expired_at or "",
                "episodes": edge.episodes,
            },
        )
        return edge

    def get_all_edges(self, graph_id: str) -> List[GraphEdge]:
        """Get all edges in a graph"""
        conn = self._connect(graph_id)
        result = conn.execute(
            """
            MATCH (a:Node)-[e:Edge]->(b:Node)
            RETURN e.uuid, e.name, e.fact, e.fact_type, a.uuid, b.uuid,
                   e.attributes, e.created_at, e.valid_at, e.invalid_at, e.expired_at, e.episodes
            """
        )
        return [self._row_to_edge(row) for row in result.get_all()]

    # ========== Search ==========

    def search(self, graph_id: str, query: str, limit: int = 10, scope: str = "edges") -> List[Dict[str, Any]]:
        """
        Text search across graph nodes and/or edges.
        Matches query terms against names, summaries, facts, and attributes.
        """
        query_lower = query.lower().strip()
        query_terms = query_lower.split()
        results = []

        if scope in ("edges", "both"):
            edges = self.get_all_edges(graph_id)
            nodes = self.get_all_nodes(graph_id)
            node_map = {n.uuid_: n.name for n in nodes}

            for e in edges:
                text = f"{e.name} {e.fact}".lower()
                score = sum(1 for term in query_terms if term in text)
                if score > 0:
                    results.append({
                        "type": "edge",
                        "uuid": e.uuid_,
                        "name": e.name,
                        "fact": e.fact,
                        "source_node_uuid": e.source_node_uuid,
                        "target_node_uuid": e.target_node_uuid,
                        "source_node_name": node_map.get(e.source_node_uuid, ""),
                        "target_node_name": node_map.get(e.target_node_uuid, ""),
                        "score": score / len(query_terms) if query_terms else 0,
                    })

        if scope in ("nodes", "both"):
            nodes = self.get_all_nodes(graph_id)
            for n in nodes:
                text = f"{n.name} {n.summary}".lower()
                attrs_text = json.dumps(n.attributes, ensure_ascii=False).lower()
                score = sum(1 for term in query_terms if term in text or term in attrs_text)
                if score > 0:
                    results.append({
                        "type": "node",
                        "uuid": n.uuid_,
                        "name": n.name,
                        "labels": n.labels,
                        "summary": n.summary,
                        "score": score / len(query_terms) if query_terms else 0,
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    # ========== Graph Data Export ==========

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """Get complete graph data (nodes + edges) for visualization"""
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        node_map = {n.uuid_: n.name for n in nodes}

        nodes_data = [n.to_dict() for n in nodes]
        edges_data = []
        for e in edges:
            data = e.to_dict()
            data["source_node_name"] = node_map.get(e.source_node_uuid, "")
            data["target_node_name"] = node_map.get(e.target_node_uuid, "")
            edges_data.append(data)

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """Get graph statistics"""
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        type_counts: Dict[str, int] = {}
        for n in nodes:
            for label in n.labels:
                if label not in ("Entity", "Node"):
                    type_counts[label] = type_counts.get(label, 0) + 1

        rel_counts: Dict[str, int] = {}
        for e in edges:
            rel_counts[e.name] = rel_counts.get(e.name, 0) + 1

        return {
            "graph_id": graph_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "entity_type_counts": type_counts,
            "relationship_type_counts": rel_counts,
        }
