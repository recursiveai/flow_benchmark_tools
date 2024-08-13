import pytest

from recursiveai.benchmark._internal._util import async_retry, retry


@pytest.mark.asyncio
async def test_retry_success_wo_args():
    counter = 0

    @retry
    def increment() -> None:
        nonlocal counter
        counter += 1

    increment()
    assert counter == 1


@pytest.mark.asyncio
async def test_async_retry_success_wo_args():
    counter = 0

    @async_retry
    async def increment() -> None:
        nonlocal counter
        counter += 1

    await increment()
    assert counter == 1


@pytest.mark.asyncio
async def test_retry_return_on_success():
    @retry
    def func() -> int:
        return 5

    assert func() == 5


@pytest.mark.asyncio
async def test_async_retry_return_on_success():
    @async_retry
    async def func() -> int:
        return 5

    assert await func() == 5


@pytest.mark.asyncio
async def test_retry_success_w_args():
    counter = 0

    @retry(max_retries=3, exc_tuple=(Exception), delay=2.0, backoff=1.0)
    def increment() -> None:
        nonlocal counter
        counter += 1

    increment()
    assert counter == 1


@pytest.mark.asyncio
async def test_async_retry_success_w_args():
    counter = 0

    @async_retry(max_retries=3, exc_tuple=(Exception), delay=2.0, backoff=1.0)
    async def increment() -> None:
        nonlocal counter
        counter += 1

    await increment()
    assert counter == 1


@pytest.mark.asyncio
async def test_retry_fails_once():
    counter = 0

    class DummyException(Exception):
        pass

    @retry(delay=0.1, exc_tuple=(DummyException))
    def increment() -> None:
        nonlocal counter
        counter += 1
        if counter == 1:
            raise DummyException()

    increment()
    assert counter == 2


@pytest.mark.asyncio
async def test_async_retry_fails_once():
    counter = 0

    class DummyException(Exception):
        pass

    @async_retry(delay=0.1, exc_tuple=(DummyException))
    async def increment() -> None:
        nonlocal counter
        counter += 1
        if counter == 1:
            raise DummyException()

    await increment()
    assert counter == 2


@pytest.mark.asyncio
async def test_retry_fails_non_listed_exception():
    class DummyException(Exception):
        pass

    @retry(delay=0.1, exc_tuple=(DummyException))
    def increment() -> None:
        raise Exception()

    with pytest.raises(Exception):
        increment()


@pytest.mark.asyncio
async def test_async_retry_fails_non_listed_exception():
    class DummyException(Exception):
        pass

    @async_retry(delay=0.1, exc_tuple=(DummyException))
    async def increment() -> None:
        raise Exception()

    with pytest.raises(Exception):
        await increment()


@pytest.mark.asyncio
async def test_retry_fails_every_time():
    class DummyException(Exception):
        pass

    @retry(delay=0.1, backoff=1.0, exc_tuple=(DummyException))
    def increment() -> None:
        raise DummyException

    with pytest.raises(DummyException):
        increment()


@pytest.mark.asyncio
async def test_async_retry_fails_every_time():
    class DummyException(Exception):
        pass

    @async_retry(delay=0.1, backoff=1.0, exc_tuple=(DummyException))
    async def increment() -> None:
        raise DummyException

    with pytest.raises(DummyException):
        await increment()
