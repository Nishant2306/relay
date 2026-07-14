"""Redis Lua token-bucket rate limiting (SPEC C4, ADR-0005).

Two buckets per team, one atomic Lua round-trip per bucket:
  rl:{team}:rpm — capacity = RPM, refill RPM/60 tokens/sec, cost 1/request
  rl:{team}:tpm — capacity = TPM, refill TPM/60 tokens/sec,
                  cost = prompt_tokens + max_tokens (pre-charge), unused
                  remainder refunded after the response (reconcile)

Why token bucket: O(1) memory per team, natural burst allowance up to
capacity, and the whole take-or-reject decision happens atomically inside
Redis — no read-modify-write race under concurrent gateway workers.

On TPM rejection the RPM token already taken is refunded so a blocked request
costs nothing. Actual usage above the pre-charge is absorbed (documented
trade-off: overshoot is bounded by max_tokens accuracy).
"""

from __future__ import annotations

import time

from redis.asyncio import Redis

from gateway.models import TeamContext
from gateway.pipeline import LimitDecision

# KEYS[1] bucket | ARGV: capacity, refill_per_sec, now_ms, cost
# Returns {allowed(0/1), retry_after_seconds(str), tokens_left(str)}
TAKE_LUA = """
local tokens = redis.call('HGET', KEYS[1], 'tokens')
local ts = redis.call('HGET', KEYS[1], 'ts')
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

if tokens == false then
  tokens = capacity
  ts = now
else
  tokens = tonumber(tokens)
  ts = tonumber(ts)
end

local elapsed = math.max(0, now - ts) / 1000.0
tokens = math.min(capacity, tokens + elapsed * rate)

local allowed = 0
local retry_after = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after = (cost - tokens) / rate
end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', KEYS[1], math.ceil(capacity / rate * 2000))
return {allowed, tostring(retry_after), tostring(tokens)}
"""

# KEYS[1] bucket | ARGV: capacity, amount  — refund up to capacity
REFUND_LUA = """
local capacity = tonumber(ARGV[1])
local amount = tonumber(ARGV[2])
local tokens = redis.call('HGET', KEYS[1], 'tokens')
if tokens == false then
  tokens = capacity
else
  tokens = math.min(capacity, tonumber(tokens) + amount)
end
redis.call('HSET', KEYS[1], 'tokens', tokens)
return tostring(tokens)
"""


class RedisRateLimiter:
    def __init__(self, redis: Redis):
        self.redis = redis
        self._take = redis.register_script(TAKE_LUA)
        self._refund = redis.register_script(REFUND_LUA)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    async def _take_from(self, key: str, capacity: int, cost: float) -> tuple[bool, float]:
        rate = capacity / 60.0
        allowed, retry_after, _left = await self._take(
            keys=[key], args=[capacity, rate, self._now_ms(), cost]
        )
        return bool(int(allowed)), float(retry_after)

    async def _refund_to(self, key: str, capacity: int, amount: float) -> None:
        await self._refund(keys=[key], args=[capacity, amount])

    async def check(self, team: TeamContext, est_tokens: int) -> LimitDecision:
        rpm_key = f"rl:{team.team_id}:rpm"
        ok, retry_after = await self._take_from(rpm_key, team.rpm, 1)
        if not ok:
            return LimitDecision(allowed=False, kind="rpm", retry_after_s=retry_after)

        if team.tpm > 0:
            tpm_key = f"rl:{team.team_id}:tpm"
            ok, retry_after = await self._take_from(tpm_key, team.tpm, est_tokens)
            if not ok:
                # the request is rejected — give the RPM token back
                await self._refund_to(rpm_key, team.rpm, 1)
                return LimitDecision(allowed=False, kind="tpm", retry_after_s=retry_after)

        return LimitDecision(allowed=True)

    async def reconcile(self, team: TeamContext, est_tokens: int, actual_tokens: int) -> None:
        """Refund the unused part of the TPM pre-charge (ADR-0005)."""
        remainder = est_tokens - actual_tokens
        if remainder > 0 and team.tpm > 0:
            await self._refund_to(f"rl:{team.team_id}:tpm", team.tpm, remainder)
