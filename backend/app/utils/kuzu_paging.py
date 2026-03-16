"""Graph pagination utilities.

With KuzuDB (local embedded database), pagination is handled directly
by the GraphDatabase class. This module is kept as a stub for backward
compatibility with any imports.
"""

from __future__ import annotations
from typing import Any
from .logger import get_logger

logger = get_logger('mirofish.graph_paging')


def fetch_all_nodes(client: Any, graph_id: str, **kwargs) -> list:
    """Stub - pagination is now handled by GraphDatabase.get_all_nodes()"""
    logger.warning("fetch_all_nodes called on deprecated kuzu_paging module. "
                  "Use GraphDatabase.get_all_nodes() instead.")
    return []


def fetch_all_edges(client: Any, graph_id: str, **kwargs) -> list:
    """Stub - pagination is now handled by GraphDatabase.get_all_edges()"""
    logger.warning("fetch_all_edges called on deprecated kuzu_paging module. "
                  "Use GraphDatabase.get_all_edges() instead.")
    return []
