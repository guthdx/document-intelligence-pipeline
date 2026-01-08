"""Storage layer for Document Intelligence Pipeline.

Provides database access via SQLAlchemy with PostgreSQL + pgvector.
"""

from .database import (
    Base,
    async_session_factory,
    close_db,
    engine,
    get_session,
    init_db,
)
from .orm_models import (
    BlockORM,
    ChunkORM,
    DocumentORM,
    EntityORM,
    PageORM,
    TableCellORM,
    TableORM,
)
from .repositories import (
    BlockRepository,
    ChunkRepository,
    DocumentRepository,
    PageRepository,
    TableRepository,
)

__all__ = [
    # Database
    "Base",
    "engine",
    "async_session_factory",
    "get_session",
    "init_db",
    "close_db",
    # ORM Models
    "DocumentORM",
    "PageORM",
    "BlockORM",
    "TableORM",
    "TableCellORM",
    "EntityORM",
    "ChunkORM",
    # Repositories
    "DocumentRepository",
    "PageRepository",
    "BlockRepository",
    "TableRepository",
    "ChunkRepository",
]
