import threading
from datetime import date

from copilot.limits import DailyLimit


def test_daily_limit_allows_up_to_the_limit_then_refuses():
    limit = DailyLimit(3, today=lambda: date(2026, 11, 19))
    assert [limit.try_acquire() for _ in range(4)] == [True, True, True, False]
    assert limit.remaining() == 0


def test_daily_limit_resets_on_a_new_day():
    day = {"d": date(2026, 11, 19)}
    limit = DailyLimit(1, today=lambda: day["d"])
    assert limit.try_acquire() and not limit.try_acquire()
    day["d"] = date(2026, 11, 20)
    assert limit.remaining() == 1 and limit.try_acquire()


def test_daily_limit_is_safe_across_threads():
    limit = DailyLimit(50, today=lambda: date(2026, 11, 19))
    granted = []

    def worker():
        for _ in range(20):
            granted.append(limit.try_acquire())

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(granted) == 50
