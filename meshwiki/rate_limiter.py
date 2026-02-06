"""Thread-safe sliding window rate limiter per Meshtastic node ID."""

import math
import threading
import time


class RateLimiter:
    """Sliding window rate limiter with French denial messages."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, user_id: str) -> tuple[bool, str | None]:
        """Check if a user is allowed to make a request.

        Returns (True, None) if allowed, or (False, denial_message) if rate limited.
        """
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            timestamps = self._requests.get(user_id, [])
            # Remove expired timestamps
            timestamps = [t for t in timestamps if t > cutoff]

            if len(timestamps) < self.max_requests:
                timestamps.append(now)
                self._requests[user_id] = timestamps
                return (True, None)

            # Rate limited — compute wait time
            oldest = timestamps[0]
            wait_seconds = self.window_seconds - (now - oldest)
            wait_str = self._format_wait_time(wait_seconds)
            window_minutes = self.window_seconds // 60

            message = (
                f"Limite atteinte ({self.max_requests} requêtes par "
                f"{window_minutes} min). Réessayez dans {wait_str}."
            )
            self._requests[user_id] = timestamps
            return (False, message)

    def cleanup(self) -> None:
        """Remove entries for users inactive for more than 1 hour."""
        cutoff = time.time() - 3600
        with self._lock:
            inactive = [
                uid for uid, timestamps in self._requests.items()
                if not timestamps or timestamps[-1] < cutoff
            ]
            for uid in inactive:
                del self._requests[uid]

    @staticmethod
    def _format_wait_time(seconds: float) -> str:
        """Format wait time in French: minutes if >= 60s, else seconds."""
        if seconds >= 60:
            minutes = math.ceil(seconds / 60)
            return f"{minutes} min"
        return f"{math.ceil(seconds)} sec"
