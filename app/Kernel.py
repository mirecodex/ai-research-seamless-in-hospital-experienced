import json
import logging

from fastapi import FastAPI
from config.setting import env
from contextlib import asynccontextmanager
from fastapi_limiter import FastAPILimiter
from config.ratelimit import custom_callback, service_name_identifier, redis_connection
from config.phoenix import Phoenix
from core.scheduler import SchedulerManager
from core.queue import QueueManager
from core.navigation import GraphManager
from core.playwright import PlaywrightManager
from app.repositories.GraphRepository import graphRepository
from config.apm import apm
from config.mcp import mcpconfig

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # os.environ["LANGSMITH_API_KEY"] = env.langsmith_api_key
    # os.environ["LANGSMITH_ENDPOINT"] = env.langsmith_endpoint
    # os.environ["LANGSMITH_TRACING_V2"] = env.langsmith_tracing
    # os.environ["LANGSMITH_PROJECT"] = env.langsmith_project
    # async with contextlib.AsyncExitStack() as stack:
        # await stack.enter_async_context(mcp_server.session_manager.run())

    # Phoenix.init()
    if env.MCP_SESSION:
        app.state.mcp = mcpconfig
        await app.state.mcp.open()
        print("MCP session opened in lifespan.")

    from config.routes import setup_routes
    setup_routes(app)

    QueueManager.init()
    await SchedulerManager.init()
    await FastAPILimiter.init(
        redis_connection,
        identifier=service_name_identifier,
        http_callback=custom_callback,
    )

    GraphManager.set_repository(graphRepository)
    try:
        await GraphManager.load_all_buildings()
    except Exception:
        from core.navigation.graph import HospitalGraph
        fallback_path = f"{env.GRAPH_DATA_DIR}/{env.DEFAULT_BUILDING}.json"
        try:
            with open(fallback_path) as f:
                doc = json.load(f)
            graph = HospitalGraph.from_mongo_doc(doc)
            GraphManager.register(doc.get("_id", env.DEFAULT_BUILDING), graph)
            logger.info("Loaded graph from local JSON fallback: %s", fallback_path)
        except FileNotFoundError:
            logger.warning("No MongoDB and no local fallback at %s", fallback_path)

    try:
        await GraphManager.start_listener()
    except Exception:
        logger.warning("Redis pub/sub unavailable, graph sync disabled")

    if env.PLAYWRIGHT_POOL_SIZE > 0:
        try:
            await PlaywrightManager.start(pool_size=env.PLAYWRIGHT_POOL_SIZE)
        except Exception:
            logger.warning("Playwright unavailable, PNG rendering disabled")

    yield
    if env.PLAYWRIGHT_POOL_SIZE > 0:
        await PlaywrightManager.stop()
    await GraphManager.stop_listener()
    if hasattr(app.state, "mcp"):
        await app.state.mcp.close()
    await redis_connection.aclose() # FastAPILimiter redis connection close deprecating .close() into aclose()
    await SchedulerManager.close()
    await QueueManager.close()
    apm.close()

app = FastAPI(lifespan=lifespan)
