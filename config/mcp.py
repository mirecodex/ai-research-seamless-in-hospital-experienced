from __future__ import annotations

import asyncio
import base64
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import httpx
from fastmcp import Client
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools

from config.setting import env
from config.logger import logger

# Timeouts and retry variable
HEALTH_CHECK_TIMEOUT = 2.0
TOOL_CALL_TIMEOUT = 8.0
TOOL_CALL_RETRY_TIMEOUT = 4.0 
RECONNECT_SLEEP_AFTER_HEALTH = 7.0
SESSION_CREATE_RETRIES = 6
SESSION_CREATE_BACKOFF = 2.0


def _decode_cfg(raw: str) -> Dict[str, Any]:
    """Decode connection config from base64(json) or plain json string.

    Returns a dict representing decoded JSON.
    Raises ValueError on invalid input.
    """
    if not raw:
        return {}

    try:
        decoded = base64.b64decode(raw).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        try:
            return json.loads(raw)
        except Exception as exc:
            raise ValueError("Invalid MCP config (not JSON or base64-encoded JSON)") from exc


class MCPConfig:
    """Persistent MCP client manager.

    Usage pattern:
      - call open() once during application startup / kernel lifespan
      - call tool_call_mcp(...) many times
      - call close() once during shutdown
    """

    def __init__(self, env_configs: List[str]):
        self.connections: Dict[str, Dict[str, Any]] = self._merge_connections(env_configs)
        self.client: Client = Client(self.connections)
        self._all_tools: Optional[List[StructuredTool]] = None
        self._stack: Optional[AsyncExitStack] = None
        self._sessions: Dict[str, Any] = {}

        self._opened: bool = False
        self._sessions_created: bool = False
        self._lock = asyncio.Lock()
        self._disconnect_detected: Dict[str, bool] = {}

        logger.info("MCPConfig initialized with servers: %s", list(self.connections.keys()))

    def _merge_connections(self, env_configs: List[str]) -> Dict[str, Dict[str, Any]]:
        """Merge multiple environment configuration strings into single connections dict."""
        merged: Dict[str, Dict[str, Any]] = {}
        for raw in env_configs:
            if not raw:
                continue
            cfg = _decode_cfg(raw)
            if not isinstance(cfg, dict):
                continue
            merged.update(cfg)

        if not merged:
            raise ValueError("No valid MCP connections found from env_configs")
        return merged

    def _wrap_tool_with_persistent_session(
        self, tool: StructuredTool, server_name: str
    ) -> StructuredTool:
        """Return a StructuredTool that routes execution through this MCPConfig.

        The wrapper ensures calls use our persistent session management and
        centralized error handling (tool_call_mcp).
        """

        async def persistent_coroutine(**kwargs):
            if not self._sessions_created:
                return f"Error: MCP service is currently unavailable. Tool '{tool.name}' cannot be executed."

            # Resolve the tool's effective name: allow both prefixed and unprefixed
            tool_name = tool.name
            if "_" in tool_name:
                prefix, rest = tool_name.split("_", 1)
                tool_real_name = rest if prefix == server_name else tool_name
            else:
                tool_real_name = tool_name

            # Use our centralized caller
            full_tool_name = f"{server_name}_{tool_real_name}"
            return await self.tool_call_mcp(tool_name=full_tool_name, tool_input={"arguments": kwargs})

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            coroutine=persistent_coroutine,
            response_format="content",
            metadata=tool.metadata,
        )

    async def _check_mcp_health(self, timeout: float = HEALTH_CHECK_TIMEOUT, raise_on_failure: bool = False) -> bool:
        """Check MCP health endpoint. Returns True if healthy.

        When raise_on_failure=True, a RuntimeError is raised on failure (used during startup).
        """
        url = getattr(env, "MCP_HEALTH_CHECK_URL", None)
        if not url:
            msg = "MCP_HEALTH_CHECK_URL is not configured in env"
            logger.error(msg)
            if raise_on_failure:
                raise RuntimeError(msg)
            return False

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
                healthy = resp.status_code == 200
                if healthy:
                    logger.debug("MCP health check OK")
                else:
                    logger.warning("MCP health check returned status %s", resp.status_code)
                return healthy
        except Exception as exc:
            logger.warning("MCP health check failed: %s", exc)
            if raise_on_failure:
                raise RuntimeError("MCP health check failed") from exc
            return False

    async def open(self) -> None:
        """Open persistent client and create sessions + load tools.

        This method is safe to call multiple times; subsequent calls are no-ops.
        """
        async with self._lock:
            if self._opened:
                return
            
            # Ensure MCP server reachable during startup
            await self._check_mcp_health(raise_on_failure=True)

            try:
                await self.client.__aenter__()
                self._stack = AsyncExitStack()
                await self._stack.__aenter__()

                all_tools: List[StructuredTool] = []
                for server_name, connection in self.connections.items():
                    session = await self._stack.enter_async_context(create_session(connection))
                    await session.initialize()

                    tools = await load_mcp_tools(session=session, server_name=server_name)
                    wrapped = [self._wrap_tool_with_persistent_session(t, server_name) for t in tools]
                    all_tools.extend(wrapped)
                    self._sessions[server_name] = session

                self._all_tools = all_tools
                self._sessions_created = True
                self._opened = True

                # initialize disconnect flags
                for server in self.connections.keys():
                    self._disconnect_detected[server] = False

                logger.info("MCP persistent session opened; loaded %d tools", len(self._all_tools))
                for tool in self._all_tools:
                    logger.debug("  - %s", tool.name)

            except Exception as exc:
                logger.exception("Failed to initialize MCP sessions")
                await self.close()
                raise RuntimeError("Failed to initialize MCP sessions") from exc

    def open_sync(self) -> None:
        """Synchronous entrypoint for environments where event loop is not running.

        This wraps open() with asyncio.run. If an event loop is already running this
        will log a warning and skip (tools should be loaded by open() in that case).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # no running loop
            try:
                asyncio.run(self.open())
                logger.info("MCP persistent session opened (sync)")
            except Exception as exc:
                logger.exception("open_sync failed")
                raise
        else:
            logger.warning("open_sync called while an asyncio loop is running; use open() instead")

    async def close(self) -> None:
        """Close sessions and client; idempotent."""
        async with self._lock:
            if not self._opened and not self._sessions_created:
                return

            self._opened = False
            self._sessions_created = False

            if self._stack is not None:
                try:
                    await self._stack.__aexit__(None, None, None)
                except Exception:
                    logger.exception("Error closing AsyncExitStack")
                self._stack = None
                self._sessions.clear()

            try:
                await self.client.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error closing MCP client")

            logger.info("MCP persistent session closed")

    async def tool_call_mcp(self, tool_name: str, tool_input: dict) -> str:
        """Call a tool via MCP and always return a string result (never raise).

        tool_name may be prefixed with the server name like "ai-search_doctor".
        """
        logger.debug("Calling tool %s with input keys: %s", tool_name, list(tool_input.keys()))

        arguments = tool_input.get("arguments", {}) or {}

        server_name: Optional[str] = None
        tool_real_name: str = tool_name

        if "_" in tool_name:
            prefix, rest = tool_name.split("_", 1)
            if prefix in self.connections:
                server_name, tool_real_name = prefix, rest

        if not await self._check_mcp_health(raise_on_failure=False):
            logger.warning("MCP is not healthy - aborting tool call %s", tool_name)
            return "Error: MCP service is currently unavailable. Please try again later."

        #Ensure sessions exist or attempt reconnect
        sessions_need_reconnect = (
            not self._sessions_created 
            or not self._sessions 
            or any(self._disconnect_detected.values())
        )
        if sessions_need_reconnect:
            logger.debug(
                "Sessions require reconnect (created=%s, count=%d, disconnect=%s)",
                self._sessions_created,
                len(self._sessions),
                any(self._disconnect_detected.values())
            )
            try:
                await self._reconnect_all_sessions()
                for k in self._disconnect_detected:
                    self._disconnect_detected[k] = False
            except Exception:
                logger.exception("Reconnect failed")
                return "Error: MCP service is reconnecting. Please try again in a moment."

        async def _call_session(sess, name: str, timeout: float = TOOL_CALL_TIMEOUT) -> str:
            """Helper to call tool on session with timeout and safe content extraction."""
            res = await asyncio.wait_for(
                sess.call_tool(name=name, arguments=arguments), 
                timeout=timeout
            )
            if res.content and len(res.content) > 0 and hasattr(res.content[0], 'text'):
                return res.content[0].text
            return str(res)

        # If server prefixed and available, call directly
        if server_name and server_name in self._sessions:
            session = self._sessions[server_name]
            try:
                return await _call_session(session, tool_real_name)
            except asyncio.TimeoutError:
                logger.warning("Tool %s timeout on server %s - attempting reconnect", tool_real_name, server_name)
                self._disconnect_detected[server_name] = True
                try:
                    await self._reconnect_all_sessions()
                    self._disconnect_detected[server_name] = False

                    if server_name in self._sessions:
                        logger.debug("Retrying %s after reconnect", tool_real_name)
                        return await _call_session(
                            self._sessions[server_name], 
                            tool_real_name,
                            timeout=TOOL_CALL_RETRY_TIMEOUT
                        )
                except Exception:
                    logger.exception("Retry after reconnect failed for %s on %s", tool_real_name, server_name)
                
                return f"Error: Tool '{tool_real_name}' timed out. MCP service may be disconnecting."
            except Exception as exc:
                logger.exception("Tool '%s' failed on server '%s'", tool_real_name, server_name)
                self._disconnect_detected[server_name] = True

        # try all sessions as fallback
        for srv_name, session in list(self._sessions.items()):
            try:
                logger.debug("Trying tool %s on server %s (fallback)", tool_real_name, srv_name)
                return await _call_session(session, tool_real_name)
            except asyncio.TimeoutError:
                logger.warning("Tool %s timeout on server %s (fallback) - marking for reconnect", tool_real_name, srv_name)
                self._disconnect_detected[srv_name] = True
                continue
            except Exception as exc:
                error_msg = str(exc).lower()
                logger.debug("Tool %s failed on %s: %s", tool_real_name, srv_name, exc)
                # Detect HTTP errors (400, 4xx, 5xx) as session issues
                if (
                    '400' in error_msg 
                    or '4' in error_msg and ('status' in error_msg or 'bad request' in error_msg)
                    or '5' in error_msg and ('status' in error_msg or 'server error' in error_msg)
                    or 'connect' in error_msg
                    or not error_msg.strip()
                ):
                    logger.warning("Tool %s on %s returned HTTP error - marking for reconnect", tool_real_name, srv_name)
                    self._disconnect_detected[srv_name] = True
                continue

        logger.error("Tool '%s' not found or all sessions failed", tool_name)
        return f"Error: Tool '{tool_real_name}' is currently unavailable. Please try again later."

    def get_tools_for_bind(self, tool_names: List[str]) -> List[StructuredTool]:
        """Return prepared StructuredTool objects for the given names.
        
        Requires open() to be called first during application startup.
        """
        if self._all_tools is None:
            logger.warning("Tools not loaded yet. Call open() in lifespan first.")
            return []

        filtered = [t for t in self._all_tools if t.name in tool_names]
        logger.debug("get_tools_for_bind filtered %d tools from %s", len(filtered), tool_names)
        return filtered

    async def _reconnect_all_sessions(self) -> None:
        """Attempt to recreate sessions and reload tools when MCP becomes available."""
        async with self._lock:
            if not self._opened:
                logger.info("Skipping reconnect - system is shutting down")
                return

            # Ensure client/stack exist
            if self._stack is None:
                logger.debug("Creating client and stack for first connection")
                await self.client.__aenter__()
                self._stack = AsyncExitStack()
                await self._stack.__aenter__()

            # Give server time to finish starting
            logger.debug("Waiting %.1f seconds for MCP server to be fully ready...", RECONNECT_SLEEP_AFTER_HEALTH)
            await asyncio.sleep(RECONNECT_SLEEP_AFTER_HEALTH)

            self._sessions.clear()
            all_tools: List[StructuredTool] = []

            for server_name, connection in self.connections.items():
                session = None
                last_exc: Optional[Exception] = None
                
                for attempt in range(SESSION_CREATE_RETRIES):
                    try:
                        session = await self._stack.enter_async_context(create_session(connection))
                        await session.initialize()
                        tools = await load_mcp_tools(session=session, server_name=server_name)
                        wrapped = [self._wrap_tool_with_persistent_session(t, server_name) for t in tools]
                        all_tools.extend(wrapped)
                        self._sessions[server_name] = session
                        logger.info("Session restored for %s", server_name)
                        break
                    except Exception as exc:
                        last_exc = exc
                        logger.debug(
                            "Session creation failed for %s (attempt %d/%d): %s",
                            server_name, attempt + 1, SESSION_CREATE_RETRIES, exc
                        )
                        if attempt < SESSION_CREATE_RETRIES - 1:
                            await asyncio.sleep(SESSION_CREATE_BACKOFF)

                if session is None:
                    logger.error(
                        "Failed to restore session for %s after %d attempts: %s",
                        server_name, SESSION_CREATE_RETRIES, last_exc
                    )

            self._all_tools = all_tools
            self._sessions_created = bool(self._sessions)
            logger.info("Reconnection finished; loaded %d tools", len(self._all_tools or []))


if env.MCP_SESSION:
    mcpconfig = MCPConfig([env.MCP_CONFIG_AI_SEARCH, env.MCP_CONFIG_HOPE_RETRIEVER])
else:
    mcpconfig = None