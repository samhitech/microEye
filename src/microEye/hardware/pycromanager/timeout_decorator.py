import functools
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def target():
                return func(*args, **kwargs)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(target)
                try:
                    return future.result(timeout=seconds)
                except TimeoutError as err:
                    raise TimeoutError(
                        f'Function {func.__name__} timed out after {seconds} seconds'
                    ) from err
                except Exception as err:
                    raise Exception(
                        f'Function {func.__name__} raised an error: {err}'
                    ) from err

        return wrapper

    return decorator
