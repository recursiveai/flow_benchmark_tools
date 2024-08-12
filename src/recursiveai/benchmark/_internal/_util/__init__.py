# Copyright 2024 Recursive AI

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Awaitable, Callable, Optional

Function = Callable[..., Any]
AsyncFunction = Callable[..., Awaitable[Any]]


def retry(
    func: Optional[Function] = None,
    *,
    max_retries: int = 4,
    exc_tuple: tuple[Exception] = (Exception),
    delay: float = 1.0,
    backoff: float = 2.0,
):

    def decorator(func: Function):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _delay = delay
            logger = logging.getLogger(func.__module__)
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exc_tuple:
                    logger.warning(
                        "Caught exception while calling function. Trying again in %ss",
                        _delay,
                        exc_info=True,
                    )
                    time.sleep(_delay)
                    _delay *= backoff
            return func(*args, **kwargs)

        return wrapper

    if func:
        return decorator(func)
    return decorator


def async_retry(
    func: Optional[AsyncFunction] = None,
    *,
    max_retries: int = 4,
    exc_tuple: tuple[Exception] = (Exception),
    delay: float = 1.0,
    backoff: float = 2.0,
):

    def decorator(func: AsyncFunction):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            _delay = delay
            logger = logging.getLogger(func.__module__)
            for _ in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exc_tuple:
                    logger.warning(
                        "Caught exception while calling function. Trying again in %ss",
                        _delay,
                        exc_info=True,
                    )
                    await asyncio.sleep(_delay)
                    _delay *= backoff
            return await func(*args, **kwargs)

        return wrapper

    if func:
        return decorator(func)
    return decorator
