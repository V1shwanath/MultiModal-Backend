from typing import Awaitable, Callable

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from MultiModal.db.meta import meta
from MultiModal.db.models import load_all_models
from MultiModal.services.redis.lifetime import init_redis, shutdown_redis
from MultiModal.settings import settings

from MultiModal.static.vectordb import vector_store
import uvicorn


def _setup_db(app: FastAPI, vector_store) -> None:  # pragma: no cover
    """
    Creates connection to the database and initializes VectorStore.

    :param app: fastAPI application.
    :param vector_store: instance of VectorStore.
    """
    engine = create_async_engine(str(settings.db_url), echo=settings.db_echo)
    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )
    app.state.db_engine = engine
    app.state.db_session_factory = session_factory
    vector_store.create_collection("your_collection_name")


async def _create_tables() -> None:  # pragma: no cover
    """Populates tables in the database."""
    load_all_models()
    engine = create_async_engine(str(settings.db_url))
    async with engine.begin() as connection:
        await connection.run_sync(meta.create_all)
    await engine.dispose()


def register_startup_event(
    app: FastAPI,
    vector_store
) -> Callable[[], Awaitable[None]]:
    """
    Actions to run on application startup.

    This function uses fastAPI app to store data
    in the state, such as db_engine.

    :param app: the fastAPI application.
    :param vector_store: instance of VectorStore.
    :return: function that actually performs actions.
    """

    @app.on_event("startup")
    async def _startup() -> None:
        app.middleware_stack = None
        _setup_db(app,vector_store)
        await _create_tables()
        init_redis(app)

        app.middleware_stack = app.build_middleware_stack()

    return _startup




def register_shutdown_event(
    app: FastAPI,
    vector_store
) -> Callable[[], Awaitable[None]]:  # pragma: no cover
    """
    Actions to run on application's shutdown.

    :param app: fastAPI application.
    :param vector_store: instance of VectorStore.
    :return: function that actually performs actions.
    """

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # noqa: WPS430
        await app.state.db_engine.dispose()

        await shutdown_redis(app)
        
        if vector_store.collection is not None:
            print("====> Deleting collection <====")
            vector_store.delete_collection("your_collection_name")
        else:
            print("Collection Deleted Already")
        
        pass  # noqa: WPS420    

    return _shutdown
