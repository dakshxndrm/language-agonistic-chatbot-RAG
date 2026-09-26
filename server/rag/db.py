"""Shared asyncpg pool. asyncpg speaks the wire protocol directly, so the
SQLAlchemy-style `+asyncpg` in DATABASE_URL has to be stripped first.
"""

from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from server.config import get_settings

_pool: asyncpg.Pool | None = None


def dsn() -> str:
    return get_settings().database_url.replace("postgresql+asyncpg://", "postgresql://")


async def _init(conn: asyncpg.Connection) -> None:
    await register_vector(conn)      # lets us pass/receive python lists as vector


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn(), min_size=1, max_size=5, init=_init)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
