from sqlalchemy.orm import DeclarativeBase

from MultiModal.db.meta import meta


class Base(DeclarativeBase):
    """Base for all models."""

    metadata = meta
