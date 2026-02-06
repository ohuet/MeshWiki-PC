"""Tests for meshwiki.rate_limiter — sliding window rate limiting."""

import threading
import time
from unittest.mock import patch

from meshwiki.rate_limiter import RateLimiter


def test_allows_up_to_max_requests():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        allowed, msg = limiter.check("user1")
        assert allowed is True
        assert msg is None


def test_blocks_after_max_requests():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        limiter.check("user1")
    allowed, msg = limiter.check("user1")
    assert allowed is False
    assert msg is not None
    assert "3 requêtes par 1 min" in msg


def test_denial_message_contains_wait_time_minutes():
    limiter = RateLimiter(max_requests=2, window_seconds=600)
    limiter.check("user1")
    limiter.check("user1")
    allowed, msg = limiter.check("user1")
    assert allowed is False
    assert "min" in msg
    assert "Réessayez dans" in msg


def test_denial_message_contains_wait_time_seconds():
    limiter = RateLimiter(max_requests=2, window_seconds=30)
    limiter.check("user1")
    limiter.check("user1")
    allowed, msg = limiter.check("user1")
    assert allowed is False
    assert "sec" in msg


def test_window_expiration_allows_again():
    """After the window expires, the user should be allowed again."""
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    # Simulate timestamps in the past
    past = time.time() - 120  # 2 minutes ago
    with limiter._lock:
        limiter._requests["user1"] = [past, past + 1]

    allowed, msg = limiter.check("user1")
    assert allowed is True
    assert msg is None


def test_independent_user_counters():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    limiter.check("user1")
    limiter.check("user1")

    # user1 is blocked
    allowed, _ = limiter.check("user1")
    assert allowed is False

    # user2 is still allowed
    allowed, _ = limiter.check("user2")
    assert allowed is True


def test_cleanup_removes_inactive_users():
    limiter = RateLimiter(max_requests=5, window_seconds=60)

    # Add an old entry (> 1 hour ago)
    old_time = time.time() - 7200
    with limiter._lock:
        limiter._requests["old_user"] = [old_time]
        limiter._requests["recent_user"] = [time.time()]

    limiter.cleanup()

    with limiter._lock:
        assert "old_user" not in limiter._requests
        assert "recent_user" in limiter._requests


def test_thread_safety():
    """Multiple threads calling check() concurrently should not cause errors."""
    limiter = RateLimiter(max_requests=100, window_seconds=60)
    errors = []

    def worker():
        try:
            for _ in range(50):
                limiter.check("shared_user")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Total requests recorded should be exactly 100 (max_requests), rest denied
    with limiter._lock:
        assert len(limiter._requests["shared_user"]) == 100


def test_custom_parameters_in_message():
    """Denial message should reflect custom parameters."""
    limiter = RateLimiter(max_requests=5, window_seconds=300)
    for _ in range(5):
        limiter.check("user1")
    allowed, msg = limiter.check("user1")
    assert allowed is False
    assert "5 requêtes par 5 min" in msg
