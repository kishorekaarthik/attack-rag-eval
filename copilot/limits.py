"""Usage limits for the public demo, so visitors cannot exhaust the API quota."""
import threading
from collections.abc import Callable
from datetime import date, datetime, timezone


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


class DailyLimit:
    """A count of uses per day, shared by every session in the app process.

    It lives in memory, so it restarts at zero when the app restarts; it is a
    guard against casual overuse, not an accounting system. The day is UTC.
    Gemini's free quota resets at midnight Pacific time, so the cap is set well
    below the quota to absorb the difference.
    """

    def __init__(self, limit: int, today: Callable[[], date] = utc_today):
        self._limit, self._today = limit, today
        self._lock = threading.Lock()
        self._day, self._used = None, 0

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self._used = day, 0

    def try_acquire(self) -> bool:
        with self._lock:
            self._roll()
            if self._used >= self._limit:
                return False
            self._used += 1
            return True

    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return self._limit - self._used
