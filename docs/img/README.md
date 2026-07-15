# Dashboard screenshots

Referenced from the root README. Regenerate after a fresh load run so the
window is dense and the panels tell the truth:

```bash
make up
docker exec relay-redis-1 redis-cli FLUSHDB   # one clean cache ramp, no mid-run dips
make seed
make loadtest
```

Then screenshot each dashboard in kiosk mode with a tight window, while the
run is still recent:

| File | URL |
|---|---|
| `grafana-business.png` | http://localhost:3000/d/relay-business/relay-business?from=now-15m&to=now&kiosk |
| `grafana-operations.png` | http://localhost:3000/d/relay-ops/relay-operations?from=now-15m&to=now&kiosk |
| `grafana-performance.png` | http://localhost:3000/d/relay-perf/relay-performance?from=now-15m&to=now&kiosk |

`&kiosk` hides the Grafana chrome; `from=now-15m` keeps the graphs dense
instead of trailing off into flat space.
