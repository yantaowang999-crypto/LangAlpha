"""Shared utilities for agent tools."""

import functools
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import structlog

from ptc_agent.core.sandbox._shared import SANDBOX_TIMEOUT
from ptc_agent.core.sandbox.runtime import SandboxGoneError, SandboxTransientError
from src.observability import mark_tool_failed, mark_tool_unavailable, stamp_tool_provider

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def mark_sandbox_failed(failure: str) -> None:
    """Mark the enclosing ExecuteCode or Bash run failed for a sandbox fault.

    Only for the sandbox failing, never for the agent's code raising or
    exiting nonzero, which is the tool doing its job. A timeout is kept apart
    from the failures: code the agent wrote running long hits the limit far
    more often than a stuck sandbox does, and a dead one raises instead.
    """
    if failure == SANDBOX_TIMEOUT:
        mark_tool_unavailable(failure)
    else:
        mark_tool_failed(failure)


def mark_sandbox_raised(exc: BaseException) -> None:
    """Mark the enclosing run failed when what a tool caught is the sandbox failing.

    For a tool that catches everything to answer the agent in words, where
    most of what it catches is the agent's request refused (a read-only path,
    content the store will not hold). A sandbox gone or unreachable is the
    provider failing.
    """
    if isinstance(exc, (SandboxGoneError, SandboxTransientError)):
        mark_tool_failed(exc)


def stamp_sandbox_served(backend: Any, path: str) -> None:
    """Name the sandbox's provider on a file call the sandbox serves.

    On every such call, not only the failures, so a provider's error rate has
    the calls it served beneath it. A path under a store mount is served in
    process and names no provider.
    """
    route_for = getattr(backend, "route_for", None)
    provider = getattr(backend, "provider", None)
    if provider and (route_for is None or route_for(path) is None):
        stamp_tool_provider(provider)


def tool_error_handler(operation_name: str) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[str]]]:
    """Decorator for consistent tool error handling.

    Args:
        operation_name: Human-readable name of the operation for logging

    Usage:
        @tool_error_handler("file read")
        async def my_tool(...):
            ...
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[str]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> str:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"{operation_name} failed",
                    error=str(e),
                    exc_info=True
                )
                return f"ERROR: {operation_name} failed - {e!s}"
        return wrapper
    return decorator
