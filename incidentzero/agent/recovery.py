from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from incidentzero.model.errors import TransientModelError

T = TypeVar("T")


class RetryPolicy:
    """Bounded retry for transient model failures (R5, R11, R12).

    Only :class:`TransientModelError` is retried. A permanent model error, a
    schema mistake, a budget exhaustion or an unsafe tool action is a decision
    for the controller, not something to re-attempt.

    ``max_attempts`` is a total attempt count, not a count of extra tries, so
    ``max_attempts=3`` means at most 3 provider requests and at most 2 sleeps.
    The sleeper is injected so unit tests stay fast (Task C).
    """

    def __init__(
        self,
        max_attempts: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        jitter: float = 0.0,
        on_retry: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = max_attempts
        self.sleeper = sleeper
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.on_retry = on_retry
        #: Retries performed during the most recent call_model, for the trace (R13).
        self.last_retry_count = 0
        #: Retries performed across the whole run.
        self.total_retries = 0

    def backoff_delay(self, attempt: int) -> float:
        """Capped exponential backoff for the failure of ``attempt`` (1-based).

        Jitter defaults to zero so that tests are deterministic; a live run can
        enable it to avoid synchronised retries against a rate-limited provider.
        """
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        if self.jitter:
            delay += random.uniform(0.0, self.jitter)
        return delay

    def call_model(self, fn: Callable[[], T]) -> T:
        """Call ``fn``, retrying only transient failures, at most ``max_attempts`` times.

        Budget accounting lives inside ``fn``: each attempt is a real provider
        request and must consume the LLM budget (Q23). Because a budget error is
        not a TransientModelError, exhausting the budget mid-retry stops the loop
        immediately instead of burning the remainder.
        """
        self.last_retry_count = 0
        last_error: TransientModelError | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn()
            except TransientModelError as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break  # bounded: never retry forever (R11)
                delay = self.backoff_delay(attempt)
                self.last_retry_count += 1
                self.total_retries += 1
                if self.on_retry is not None:
                    self.on_retry({
                        "event": "model_retry",
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "max_attempts": self.max_attempts,
                        "delay_seconds": delay,
                        "error": str(exc),
                    })
                self.sleeper(delay)

        assert last_error is not None
        raise last_error
