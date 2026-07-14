# PROJECT B — "Relay": Cost-Optimizing LLM Gateway with Semantic Caching
### SPEC v2 — supersedes v1. Updated for Relay Datasets v1.1.0 (frozen).

> **How to use this file:** This lives at the repo root as `SPEC.md`. Work with Claude Code
> milestone by milestone (§14). Two documents are normative and outrank this spec where they
> overlap: `datasets/TIERING_RUBRIC.md` (tier definitions, split contract, length-only control)
> and the two frozen JSONL datasets. Build this **after Argus hits its Definition of Done.**

---

## 0. Current Project State (read this first, Claude Code)

**Exists today — nothing else does:**

| Artifact | Contents | Status |
|---|---|---|
| `datasets/cache_traps.jsonl` | **120 pairs**: 60 `hit` (paraphrases that *must* collide) + 60 `miss` (traps that *must not*). Splits: 40 dev (20/20), 80 test (40/40). All 120 `human_confirmed`. | **FROZEN** |
| `datasets/complexity_labels.jsonl` | **600 prompts**, tiers 205/196/199, all `human_verified`, **34 logged bootstrap corrections**, 118 `split_group`s, `length_confound_stratum` populated (457 normal / 82 long_tier1 / 61 short_tier3), `cache_class` populated (554 stable / 39 no_cache / 7 temporal). | **FROZEN** |
| `datasets/TIERING_RUBRIC.md` | 10 tier rules, split contract, length-only control gate. | **FROZEN** |

**Does not exist yet:** repo, gateway, mock provider, Redis/Postgres, cache, router, verifier,
resilience layer, metrics, dashboards, load tests. Everything below is to be built.

**First actions (M0):** init repo, place the three files under `datasets/`, commit together,
tag `relay-datasets-v1.1.0`. From that tag forward `datasets/*` is immutable; a CI check fails
any PR that modifies them without a version bump.

**Verified baselines you inherit (do not re-derive, do enforce):**
- Length-only control (7-fold `StratifiedGroupKFold` on `split_group`, seed 42, pooled OOF):
  **logreg ≈45.3%, depth-3 tree ≈46.0%.** Both must stay **< 50%** or the release gate fails.
  Your real classifier must beat this by a wide margin — that gap *is* the result you report.
- Trap phenomena taxonomy already in the data: `word_order`, `framing`, `synonym`, `politeness`,
  `verbosity`, `number_format`, `formatting` (hit pairs); `negation_flip`, `source_target_swap`,
  `aggregation_swap`, `operation_inverse`, `temporal_boundary_flip`, `number_change` (miss pairs).

---

## 1. What You're Building

**One-liner:** An OpenAI-compatible API gateway that enforces per-team rate limits and dollar
budgets, answers repeated and paraphrased requests instantly from a semantic cache, routes each
cache-miss to the cheapest model capable of handling it, and survives provider outages with
retries, fallback chains, and circuit breakers — all observable in Prometheus/Grafana.

**Elevator pitch (memorize):**
"Every company running LLMs at scale overpays and goes down with their provider. Relay is a
drop-in gateway — change one base URL — that cut simulated API spend by X% two ways: a semantic
cache that recognizes paraphrased repeats without colliding on lookalike prompts, and a
complexity router that sends easy work to cheap models with an async quality-verification loop
watching for routing mistakes. When a provider degrades, circuit breakers and fallback chains
keep serving with zero dropped requests, at under Y ms of gateway overhead."

**Merged from:** Project 11 (gateway, limits, resilience, observability), Project 7 (semantic
cache), Project 2 (cost router + verification loop).

**Companion project:** [Argus](https://github.com/USERNAME/argus) — the evaluation/observability
platform. Relay is the *serving* layer (what did it cost, did it stay up); Argus is the *quality*
layer (was the output good). Cross-link the READMEs; they are one body of work.

**Comparable products (name them in the README):** LiteLLM, Portkey, Helicone, OpenRouter,
GPTCache, Cloudflare AI Gateway. Positioning: "I studied these and built the core myself to own
the hard parts — atomic distributed rate limiting, cache correctness under similarity matching,
streaming tee, breaker state machines — and to integrate caching + routing + budgets natively
instead of chaining three tools."

---

## 2. Why This Merge Works — One Request Pipeline

Everything composes into a single linear lifecycle. This is the README hero diagram (render as
Mermaid, same visual language as Argus's flywheel):

```
Client (OpenAI-format request, team API key)
   │
   ▼
[1] AUTH ──────────── team config lookup (allowed models, limits, budget, cache scope)
   │
   ▼
[2] RATE LIMIT ────── Redis Lua token buckets: RPM + TPM (pre-charge, reconcile after)
   │                   429 + Retry-After on breach
   ▼
[3] BUDGET CHECK ──── daily/monthly $ caps; Slack warn @80%; block @100%
   │
   ▼
[4] CACHE LOOKUP ──── exact-hash fast path → vector KNN within namespace (RedisVL)
   │                   HIT → return instantly (x-relay-cache: hit) ──────────────┐
   ▼ MISS                                                                        │
[5] COMPLEXITY ────── feature classifier → tier 1/2/3                            │
   │                   (confidence < 0.6 → promote one tier: fail-safe to quality)│
   ▼                                                                             │
[6] ROUTE ─────────── tier → model chain (routing.yaml, hot-reload)              │
   │                                                                             │
   ▼                                                                             │
[7] PROVIDER CALL ─── retry w/ backoff → fallback chain → circuit breaker        │
   │                   (streaming passthrough, tee to buffer)                    │
   ▼                                                                             │
[8] RESPOND ───────── + headers: x-relay-model / tier / cost / cache / fallback ◄┘
   │
   ▼ (async, post-response)
[9] VERIFY ────────── sample tier-1/2 responses vs top model (LLM-judge);
   │                   routing failures logged → classifier retrain set
[10] CACHE WRITE ──── complete successful responses only; TTL by cache_class
[11] OBSERVE ──────── OTel spans, Prometheus metrics, Postgres request log
```

---

## 3. Scope Contract

### In scope (v1)
1. OpenAI-compatible `POST /v1/chat/completions` (streaming + non-streaming), `GET /v1/models`.
2. Provider adapters: OpenAI, Anthropic, Ollama (local), **plus a chaos-injectable mock provider**
   (C1) so load tests are free and outage drills deterministic.
3. Team auth (API keys) + per-team config in Postgres.
4. Redis Lua token-bucket rate limiting (RPM + TPM), budget caps with warn/block + Slack.
5. **Semantic cache**: exact-hash fast path, vector similarity via local embeddings + RedisVL,
   strict namespacing, per-tier thresholds, TTLs driven by `cache_class`, singleflight stampede
   protection, invalidation API, near-miss analytics + threshold tuner.
6. **Complexity classifier** trained on the frozen 600-prompt dataset under the rubric's split
   contract, + tier→model-chain routing via hot-reloadable YAML.
7. Async quality-verification worker (sampled shadow comparison vs top-tier model) + routing-failure
   log + retrain script.
8. Resilience: health prober, retry w/ exponential backoff + jitter, per-tier fallback chains,
   circuit breakers (closed/open/half-open) with cooldown.
9. Observability: OTel spans, Prometheus metrics, 3 Grafana dashboards (JSON committed), Slack alerts.
10. Admin API: teams/limits CRUD, routing config get/put, cache invalidation, stats.
11. Locust load tests: steady mix, repeat-heavy, rate-limit storm, budget exhaustion, outage drill.
12. Docker Compose (gateway, verifier worker, mock provider, Redis Stack, Postgres, Prometheus,
    Grafana, optional Ollama). One-command startup.

### Out of scope (v2 roadmap — mention, don't build)
Go rewrite, priority queues for tiered limits, learned/adaptive cache thresholds, per-team
system-prompt injection, multi-region, JWT/OIDC auth.

### Cut order if behind (drop from the bottom LAST)
1. Near-miss analyzer + threshold-tuner endpoint
2. Verification retrain script (keep failure logging)
3. TPM buckets (keep RPM)
4. Complexity classifier → fall back to heuristic rules **and say so honestly in the README**
5. **Never cut:** OpenAI-compatible proxy, semantic cache, rate limits + budgets, fallback/breakers,
   Prometheus/Grafana, mock provider + load test. The cache is the differentiator.

---

## 4. Key Design Decisions (one ADR each)

1. **Local embeddings for the cache** (`fastembed` / bge-small-en-v1.5, 384-d, CPU, ~5–15 ms):
   calling a paid embedding API on *every* request would add cost and 20–60 ms latency to the thing
   meant to save both. Lookups need consistency, not SOTA quality. (ADR-0001)
2. **Cache key namespacing:** `namespace = sha256(system_prompt) + model + temp_bucket +
   top_p_bucket + team_scope`. Vector search happens *within* a namespace only. Prevents
   cross-feature and cross-team contamination — a correctness requirement, not an optimization.
   Default `team_scope=per-team`; global sharing opt-in. (ADR-0002)
3. **Exact-hash fast path before vector search:** `sha256(namespace + normalized_prompt)` → O(1)
   hit for true repeats; embed + KNN only on exact-miss. (ADR-0003)
4. **Singleflight stampede protection:** on miss, `SET NX` an in-flight lock per exact key;
   concurrent identical misses wait on the first result (with timeout fallback). 10 concurrent
   identical requests = 1 upstream call. (ADR-0004)
5. **TPM pre-charge + reconcile:** charge `prompt_tokens + max_tokens` before the call, refund the
   unused remainder after. Prevents burst overshoot when output length is unknown. (ADR-0005)
6. **Only cache complete, successful, low-variance responses:** `finish_reason == "stop"`, no error,
   and the prompt's `cache_class` allows it. Client disconnect mid-stream → discard buffer. (ADR-0006)
7. **Fallback never downgrades quality:** chains go same-tier-alternate-provider first, then tier+1.
   Never silently serve a weaker model on failure. (ADR-0007)
8. **Honest savings accounting:** per request store `counterfactual_cost` (tokens × flagship price)
   and `actual_cost`; savings = 1 − Σactual/Σcounterfactual, **attributed separately** to cache hits
   vs. down-routing. Report both. Never blend them into one inflated number. (ADR-0008)
9. **Dev/test discipline on the trap corpus:** thresholds are tuned **only** on the 40 dev pairs;
   the reported false-hit rate comes **only** from the 80 held-out test pairs. Tuning against the
   reported set is overfitting to your own benchmark. (ADR-0009)

---

## 5. Tech Stack (final — do not bikeshed)

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Consistency with Argus; async-native |
| API | FastAPI + Uvicorn + httpx (async) | Streaming proxy support |
| Cache + limits | **Redis Stack** (RedisVL vectors, Lua buckets) | One Redis: buckets, exact cache, vector index, locks |
| Durable store | PostgreSQL 16 | teams, config, request log, spend ledger, routing failures |
| Embeddings | fastembed (bge-small-en-v1.5, 384-d, local) | ADR-0001 |
| Classifier | scikit-learn LogisticRegression + hand-built feature extractor | Interpretable, <1 ms inference; report confusion matrix |
| Observability | opentelemetry-sdk + prometheus-client + Grafana | Industry standard |
| Config | YAML + `watchfiles` hot reload | Reroute without restart |
| Load test | Locust | Python-native scenarios |
| Alerts | Slack webhooks | Budget warnings, breaker transitions |
| Testing | pytest + pytest-asyncio + testcontainers (Redis, Postgres) | Real-infra integration tests |
| Packaging | Docker Compose | One-command stack |

**Model/cost budget:** load tests run against the **mock provider** ($0). Live demos use Ollama
(free) + small real-API runs. Verification judge sampled at 10–15%. Expected spend **$5–20**.
`MAX_DAILY_SPEND` kill-switch in the adapter is mandatory (same pattern as Argus).

---

## 6. Repo Layout

```
relay/
├── SPEC.md                     # this file
├── CLAUDE.md                   # §14
├── Makefile                    # up, seed, train, test, loadtest, drill, harvest
├── docker-compose.yml
├── datasets/                   # FROZEN — relay-datasets-v1.1.0
│   ├── cache_traps.jsonl
│   ├── complexity_labels.jsonl
│   └── TIERING_RUBRIC.md
├── config/
│   ├── routing.yaml            # tiers, chains, cache policy (hot-reload)
│   └── teams.seed.yaml
├── gateway/
│   ├── main.py                 # FastAPI app, /v1/chat/completions
│   ├── adapters/               # openai.py anthropic.py ollama.py mock.py base.py
│   ├── middleware/             # auth.py ratelimit.py (Lua) budget.py
│   ├── cache/                  # keys.py exact.py semantic.py singleflight.py ttl.py
│   ├── router/                 # features.py classifier.py routes.py
│   ├── resilience/             # health.py retry.py breaker.py fallback.py
│   └── obs/                    # otel.py metrics.py logging.py
├── verifier/                   # async shadow-compare worker
├── mockprovider/               # OpenAI-compatible chaos server
├── admin/                      # admin API + minimal Streamlit status page
├── loadtest/                   # locustfiles: steady.py repeats.py storm.py budget.py outage.py
├── infra/                      # prometheus.yml, grafana/dashboards/*.json
├── scripts/
│   ├── validate_datasets.py          # dataset contract tests (also pytest)
│   ├── split_complexity_dataset.py   # StratifiedGroupKFold per the rubric
│   ├── length_baseline.py            # the <50% control gate
│   ├── train_classifier.py
│   ├── eval_cache_traps.py           # dev-tune / test-report split enforced
│   ├── build_loadtest_corpus.py
│   ├── retrain_from_failures.py
│   └── harvest_metrics.py            # prints every README/resume number
├── docs/adr/
└── tests/
```

---

## 7. Data Model

**Postgres**
```sql
teams(id, name, api_key_hash, plan, created_at)
team_limits(team_id PK, rpm INT, tpm INT, daily_budget_usd NUMERIC, monthly_budget_usd NUMERIC,
            allowed_models JSONB, cache_scope ENUM(team,global,off))
request_log(id, ts, team_id, model_requested, model_served, provider, tier SMALLINT,
            cache ENUM(hit,miss,bypass), cache_kind ENUM(exact,semantic) NULL,
            cache_similarity FLOAT NULL,
            tokens_in INT, tokens_out INT,
            actual_cost_usd NUMERIC, counterfactual_cost_usd NUMERIC,
            latency_ms INT, overhead_ms INT, status SMALLINT, error_class TEXT NULL,
            retries SMALLINT, fallback_used BOOL, breaker_open BOOL,
            verified ENUM(pending,agree,disagree) NULL)
spend_ledger(team_id, date, usd NUMERIC, PRIMARY KEY(team_id, date))
routing_failures(id, request_id FK, tier, judge_agreement SMALLINT, cheap_model, top_model,
                 prompt_features JSONB, created_at)   -- classifier retrain set
config_audit(id, actor, path, old JSONB, new JSONB, ts)
provider_health_events(id, provider, model, from_state, to_state, reason, ts)
```

**Redis keys**
```
rl:{team}:rpm / rl:{team}:tpm     # Lua-managed token buckets
budget:{team}:{yyyy-mm-dd}        # fast counter, reconciled nightly to Postgres
ce:{namespace}:{sha}              # exact cache entry (response JSON + meta)
idx:cache                         # RedisVL vector index (ns TAG, emb VECTOR, key)
lock:{namespace}:{sha}            # singleflight, PX 30s
breaker:{provider}:{model}        # state + failure window
health:{provider}:{model}         # rolling error-rate / p99 window
```

---

## 8. Component Specs

### C0 — Dataset Integration, Validation & Governance (NEW — do this in M0)
The datasets already exist; this component makes them load-bearing.

1. **Freeze:** commit all three files; tag `relay-datasets-v1.1.0`. CI fails any PR touching
   `datasets/*` without a version bump in `TIERING_RUBRIC.md`.
2. **`scripts/validate_datasets.py`** (also a pytest, runs in CI):
   - Traps: 120 rows; 60 `hit` / 60 `miss`; splits 40 dev (20/20) + 80 test (40/40);
     all `adjudication_status == human_confirmed`; **no prompt appears in both dev and test.**
   - Labels: 600 rows; all `human_verified == true`; tiers ∈ {1,2,3}; `length_confound_stratum` ∈
     {normal, long_tier1, short_tier3}; `split_group` populated; no exact duplicate prompts.
3. **`scripts/split_complexity_dataset.py`:** `StratifiedGroupKFold(n_splits=7, shuffle=True,
   random_state=42)` grouped by `split_group`, stratified by tier (+ length stratum).
   Fold 0 = dev, fold 1 = test, folds 2–6 = train. **A pytest asserts zero `split_group` /
   `template_group` leakage across splits.**
4. **`scripts/length_baseline.py`:** the release gate. Character-count-only classifier, same fold
   protocol. Must reproduce ≈45.3% (logreg) and ≈46.0% (depth-3 tree), and **fail the build if
   either reaches 50%.** This is the control your real classifier is measured against in the README.
**Acceptance:** `make test` green on all dataset contracts; the length baseline reproduces.

### C1 — Mock Provider (build FIRST — everything tests against it)
Separate FastAPI service speaking the OpenAI API. Admin endpoints configure: base latency
distribution, error rate, 429 rate, streaming tokens/sec, hard-down toggle. Responses are
**deterministic from a prompt hash** so cache tests are stable.
**Acceptance:** `POST /chaos {error_rate:1.0}` takes it fully down; drills are scriptable;
unit-testable without network.

### C2 — OpenAI-Compatible Proxy Core
Accept the standard chat-completions schema. Response byte-compatible with OpenAI's, plus headers:
`x-relay-cache`, `x-relay-cache-kind`, `x-relay-model`, `x-relay-tier`, `x-relay-cost`,
`x-relay-fallback`, `x-relay-overhead-ms`.
**Streaming:** SSE passthrough chunk-by-chunk while teeing into a buffer; on `finish_reason=stop`
→ hand the buffer to the cache writer + logger; on provider error or client disconnect → discard.
**Acceptance:** the official `openai` Python client, pointed at Relay's base URL, works unmodified
for stream and non-stream. Conformance tests included.

### C3 — Auth + Team Config
API key → team lookup (hash compare); attach team context. Unknown key → 401; disallowed model →
403 with a clear message. Keys never logged.
**Acceptance:** integration tests for all three outcomes.

### C4 — Rate Limiting + Budgets
- **Token bucket in Lua** (single atomic round-trip): refill rate + capacity per team; one bucket
  RPM, one TPM. TPM pre-charge `prompt_tokens + max_tokens`, refund after (ADR-0005). Breach →
  429 + `Retry-After` computed from deficit/refill rate.
- **Budgets:** increment `budget:{team}:{date}` post-request; 80% → Slack warn (once/day/team);
  ≥100% → block with an explanatory body. Nightly reconcile Redis → `spend_ledger`.
**Acceptance:** property test — at 60 RPM configured, a 120-request burst yields ~60 accepted;
concurrency-safe under 50 parallel clients against a testcontainer Redis; `Retry-After` observed
accurate under Locust.
**Write the Lua yourself.** ~20 lines, and it's the classic distributed-systems interview question
in this project. If you can't reconstruct it from memory, you don't own it.

### C5 — Semantic Cache (your priority — go deep; the trap corpus is the proof)
Lookup order: (1) exact hash within namespace → hit; (2) embed prompt, KNN in RedisVL **within the
namespace**, top-1 similarity ≥ tier threshold → hit (record `cache_similarity`); else miss.
- Thresholds per tier from `routing.yaml` (start: tier1 0.90, tier2 0.95, tier3 disabled).
- **TTL from `cache_class`** — the complexity dataset already carries it (554 `stable` / 39
  `no_cache` / 7 `temporal`). Map: `stable` → 24h, `temporal` → 1h, `no_cache` → bypass. Use the
  labeled data to validate the runtime TTL heuristic instead of inventing regexes blind.
- Singleflight (ADR-0004). Invalidation API: by namespace (system-prompt change), by model,
  flush-all; every invalidation → `config_audit`.
- **Near-miss analytics:** log similarities in [0.85, threshold). `GET /admin/cache/tuning` replays
  historical near-misses at hypothetical thresholds → hit-rate vs. estimated-wrong-answer-rate table.
- **`scripts/eval_cache_traps.py` — the headline evaluation.** Tune the threshold on the **40 dev
  pairs only**; report on the **80 held-out test pairs only** (ADR-0009). Output: hit rate on
  paraphrase pairs, false-hit rate on trap pairs, **and a per-phenomenon breakdown**
  (`negation_flip`, `source_target_swap`, `aggregation_swap`, `operation_inverse`,
  `temporal_boundary_flip`) — *which trap families defeat embedding similarity* is the most
  interesting finding in the project and your best interview material.
**Acceptance:** on the held-out 80 — ≥80% of `hit` pairs collide at the tuned threshold; false-hit
rate on `miss` pairs **measured and reported honestly** (expect 2–8%; a flat 0% means the threshold
is too high or you evaluated on dev). Singleflight test: 10 concurrent identical misses → exactly
1 upstream call. Namespace isolation test: same prompt, different system prompt → never shares an
entry. Pairs flagged `known_hard_embedding_case` in the corpus are expected failures — report them
as such rather than tuning until they pass.

### C6 — Complexity Router
- **Features** (`gateway/router/features.py`): prompt token count, imperative-verb cues
  (analyze/compare/design/prove/…), constraint count, has-attached-context, requests JSON/code,
  question vs. task, message count, output-format complexity.
  **Rubric line 7 is a hard constraint:** length may be *a* feature, never a proxy for tier — the
  45.3% length-only control exists to prove the classifier learned more than character counting.
- **Training** (`scripts/train_classifier.py`): LogisticRegression on the frozen 600, using the
  group-aware split from C0. Report held-out accuracy + confusion matrix in the README **next to
  the 45.3% control.** Expect 80–88%. **>95% means split leakage — go check.**
- Confidence < 0.6 → promote one tier (fail-safe toward quality).
- **Routing map (`config/routing.yaml`, hot-reload):**
```yaml
tiers:
  1: { chain: [ollama/llama3.1-8b, mock/cheap-a], cache_threshold: 0.90 }
  2: { chain: [openai/gpt-4o-mini, anthropic/claude-haiku], cache_threshold: 0.95 }
  3: { chain: [openai/gpt-4o, anthropic/claude-sonnet], cache_threshold: disabled }
classifier: { min_confidence: 0.6, on_low_confidence: promote_one_tier }
verification: { sample_rate: 0.15, judge: openai/gpt-4o, agree_threshold: 4 }
cache_ttl: { stable: 86400, temporal: 3600, no_cache: 0 }
```
**Acceptance:** `make train` produces the model + confusion matrix; hot-editing `routing.yaml`
reroutes live traffic within 2s without restart; the length-baseline gate passes in CI.

### C7 — Async Quality Verification Loop
Worker consumes sampled tier-1/2 request IDs → re-runs the prompt on the top-tier model →
LLM-judge scores agreement 1–5 → `verified = agree|disagree`; disagreement below threshold → row in
`routing_failures` with extracted features. `scripts/retrain_from_failures.py` folds failures into
the training set and retrains (manual/weekly; automation = v2).
**Acceptance:** plant 10 prompts the classifier is known to under-route → the verifier flags an
honestly-reported fraction; retraining measurably shifts them (before/after confusion matrix in README).

### C8 — Resilience Layer
- **Health prober:** every 30s per provider/model; rolling 5-min error rate + p99 → healthy/degraded/down.
- **Retry:** retryable only (timeout, 429, 5xx); backoff 0.5s/1s/2s with jitter, max 3.
  Non-retryable (401, 400, content policy) surfaces immediately.
- **Fallback:** on exhausted retries, walk the tier chain (ADR-0007); annotate headers + log.
- **Circuit breaker per provider/model:** ≥N failures in M seconds → open (skip provider entirely);
  cooldown 30s → half-open (single probe); success → closed. Every transition →
  Prometheus + Slack + `provider_health_events`.
**Acceptance:** breaker FSM unit-tested with a **table-driven test you write yourself**
(closed→open→half-open→closed and half-open→open). Outage drill: kill the mock provider for 3 min
under load → **zero client-visible 5xx**, fallback rate spikes on Grafana, breaker opens and recovers.

### C9 — Observability
**Prometheus metrics (exact names):** `relay_requests_total{team,provider,model,tier,cache,status}`,
`relay_latency_seconds_bucket{path,cache}`, `relay_overhead_seconds_bucket`,
`relay_cost_usd_total{team,attribution}` (attribution = actual|counterfactual),
`relay_cache_hits_total{kind=exact|semantic}`, `relay_cache_similarity_bucket`,
`relay_ratelimit_rejections_total{team,kind}`, `relay_fallbacks_total{from,to}`,
`relay_breaker_state{provider,model}`, `relay_verification_disagreements_total{tier}`.
**OTel spans:** auth → ratelimit → cache_lookup → classify → provider_call(provider, attempt) →
respond; async verify + cache_write as linked spans.
**Grafana (JSON committed in `infra/grafana/`):**
1. *Operations:* RPS, error rate, latency p50/p95/p99 (hit vs miss), breaker states, fallbacks, provider health.
2. *Business:* cumulative $ saved **split cache vs routing** (ADR-0008), spend by team, budget bars, hit rate.
3. *Performance:* overhead histogram, similarity distribution, near-miss band, TPM bucket utilization.
**Acceptance:** `make up` → Grafana at :3000 pre-provisioned; all three dashboards populate after `make loadtest`.

### C10 — Admin API + Status Page
`/admin/teams` CRUD, `/admin/limits`, `/admin/routing` GET/PUT (validate + hot-apply + audit),
`/admin/cache/invalidate`, `/admin/cache/tuning`, `/admin/stats` (savings summary JSON).
Minimal Streamlit page rendering `/admin/stats` (Grafana is the real UI).
**Acceptance:** every mutating call writes `config_audit`; invalid YAML rejected with a diff-style error.

### C11 — Load & Chaos Testing
**Corpus** (`scripts/build_loadtest_corpus.py`): 1,000 unique prompts sampled from the **train split
only** of the complexity dataset (never dev/test — no leakage into reported numbers), 300 exact
repeats, 300 LLM-paraphrased variants, and the **trap prompts from the held-out test split** to
measure false-hits under realistic load.
**Scenarios:** `steady.py` (~50% unique / 30% repeat / 20% paraphrase at 20–50 RPS vs mock),
`repeats.py` (hit-rate convergence curve), `storm.py` (one team exceeds RPM → clean 429s, others
unaffected), `budget.py` (exhaust a daily cap), `outage.py` (chaos-kill primary mid-run).
**Numbers to harvest (`scripts/harvest_metrics.py`):** overhead p50/p95/p99 (target p50 < 10 ms;
report actual), cache hit rate at convergence **+ exact/semantic split + the corpus mix that
produced it**, savings % **with cache-vs-routing attribution**, false-hit rate on held-out traps
with per-phenomenon breakdown, 429 correctness, dropped requests during outage (target 0), breaker
open/close timings, verification disagreement rate per tier, classifier accuracy vs the 45.3% control.
**Acceptance:** `make loadtest && make drill` produce a `results/` folder with a generated summary
markdown. Those are your README and resume numbers.

---

## 9. Milestones (~4–5 weeks @ ~15 hrs/week with Claude Code)

**M0 (days 1–2):** repo init; dataset freeze + tag; **C0 dataset contracts, splits, length-baseline
gate**; scaffolding; compose boots (gateway, mock, Redis Stack, Postgres, Prometheus, Grafana);
Alembic baseline; Makefile; LLM adapter with cost tracking + `MAX_DAILY_SPEND`; CI skeleton;
**C1 mock provider**.
*DoD: `make up && make test` green; length baseline reproduces (<50%); curl through the gateway
round-trips to the mock provider.*

**M1 (week 1):** C2 proxy (incl. streaming tee) + C3 auth + real adapters + request logging.
*DoD: the official `openai` client works through Relay against mock and Ollama, stream + non-stream.*

**M2 (week 2):** C4 rate limits + budgets; **C5 semantic cache complete** + `eval_cache_traps.py`
producing the dev-tuned / test-reported false-hit number with per-phenomenon breakdown.
*DoD: all C4/C5 acceptance tests green — especially singleflight and namespace isolation. You now
have the project's headline number.*

**M3 (week 3):** C6 router (train on the frozen 600, report vs the 45.3% control) + C7 verification
worker + C8 resilience.
*DoD: outage drill passes with zero dropped requests; confusion matrix generated; classifier beats
the length control by a wide margin.*

**M4 (week 4):** C9 observability + C10 admin + C11 load tests + polish (README with Mermaid
pipeline diagram, 9 ADRs, Grafana screenshots, Loom, `harvest_metrics`, Argus cross-link).
*DoD: the demo script (§11) runs clean twice; every number harvested.*

Behind schedule → §3 cut order. Ship end of week 5 regardless.

---

## 10. Resume Bullet Drafts (fill numbers ONLY from `harvest_metrics.py`)

- "Built an OpenAI-compatible LLM gateway (FastAPI, Redis Stack, Postgres, Prometheus/Grafana) with
  **semantic caching** via local embeddings — X% hit rate at a **measured Y% false-hit rate** on an
  80-pair held-out adversarial trap corpus — plus **complexity-based model routing** (classifier at
  X% vs a 45.3% length-only control) with an async LLM-judge verification loop, cutting simulated
  API spend Z% (attribution: A% caching, B% routing)."
- "Implemented atomic **Lua token-bucket rate limiting** (RPM/TPM with pre-charge reconciliation),
  per-team dollar budgets, and **circuit-breaker failover**; sustained N RPS at <X ms p50 gateway
  overhead with **zero dropped requests** during simulated provider outages."
- "Designed cache-correctness guarantees (system-prompt/model/parameter namespacing, singleflight
  stampede protection, dev/test-split threshold tuning) verified against a **120-pair
  human-adjudicated paraphrase/trap corpus**."

---

## 11. Demo Script (Loom, ≤ 4 min)

1. (20s) One-liner + the request-pipeline diagram.
2. (40s) Same question three ways: cold miss (real latency) → exact repeat (instant,
   `x-relay-cache: hit`) → paraphrase (semantic hit, show the similarity score). Then a **trap pair**
   that correctly does *not* hit — that's the money shot.
3. (40s) Grafana Business dashboard: savings accumulating under Locust load; point at the
   cache-vs-routing attribution split.
4. (40s) Chaos: kill the mock provider mid-load → fallback spike, breaker opens, **zero client
   errors**, auto-recovery.
5. (30s) One team hammers past its RPM → clean 429s with `Retry-After`; others unaffected; budget bar
   hits 80% → Slack warning.
6. (30s) Router: trivial prompt → tier 1, reasoning prompt → tier 3; show a verification disagreement
   logged. Close on the savings number.

---

## 12. Interview Prep (know these cold)

1. *Why not LiteLLM/Portkey/GPTCache?* — Studied them (comparison table in the README). Built it to
   own the hard parts and to integrate cache + router + budgets natively. I can now argue when a team
   *should* just adopt LiteLLM.
2. *Biggest risk of semantic caching?* — Serving a wrong cached answer. Mitigations: strict
   namespacing, per-tier thresholds, disable for creative/temporal, and a **human-adjudicated trap
   corpus** measuring the false-hit rate on held-out pairs. Say the measured number out loud, and
   name which trap families still defeat embeddings.
3. *How did you avoid tuning on your own benchmark?* — 40 dev pairs for tuning, 80 held-out test pairs
   for the reported number. ADR-0009.
4. *Cache stampede?* — Singleflight lock per exact key; discuss the timeout fallback and why
   per-exact-key rather than per-semantic-neighborhood.
5. *Why token bucket over sliding-window log?* — O(1) memory, natural burst allowance, one Lua
   round-trip; sliding log is exacter but costlier; fixed window has boundary bursts.
6. *How is TPM enforced when output length is unknown?* — Pre-charge prompt+max_tokens, refund after
   (ADR-0005); discuss overshoot vs. under-utilization.
7. *How do breakers avoid flapping?* — Failure-count window + cooldown + half-open single probe
   (hysteresis); transitions alerted and logged.
8. *Is the savings number honest?* — Counterfactual = everything at flagship price; attribution split;
   workload simulated; methodology in the README.
9. *Why logistic regression, not an LLM router?* — <1 ms, interpretable, retrainable from the failure
   loop; an LLM classifier adds latency and cost to *every* request. **And I measured a length-only
   control at 45.3%, so I know my 8X% classifier learned complexity, not character counting.**
10. *Biggest limitation?* — Single Redis (SPOF), simulated traffic, API-key-only auth. Name them first;
    sketch the hardening path.

---

## 13. README Skeleton

Problem (redundant calls + over-provisioned models + provider outages) → Mermaid request-pipeline
diagram → **measured results table** (savings % with attribution, hit rate + the corpus mix that
produced it, false-hit rate on held-out traps + per-phenomenon breakdown, classifier vs 45.3% length
control, overhead p50/p95, zero-drop outage drill) → **known gaps** section (simulated traffic,
single-Redis SPOF, trap families that still defeat embeddings) → the datasets as first-class artifacts
(120 human-adjudicated pairs; 600 prompts with 34 logged corrections; the length-control release gate)
→ design decisions (9 ADRs) → comparison table vs LiteLLM/Portkey/GPTCache → run-it-yourself
(`make up seed train loadtest drill`) → v2 roadmap → **cross-link to Argus**.

Mirror Argus's structure: measured numbers only in the results table; anything unmeasured goes in
"Known gaps" with a defined path to a credible number.

---

## 14. Working With Claude Code

**`CLAUDE.md` (create at repo root):**
```md
# Relay — Working Agreement
- SPEC.md drives the build. datasets/TIERING_RUBRIC.md is the labeling/split/eval contract and
  OUTRANKS SPEC.md on tier semantics, splitting, and the length-control gate.
- datasets/* is FROZEN (relay-datasets-v1.1.0). Never edit those files. Changes require the human
  to run a version bump.
- NEVER tune cache thresholds against the test split of cache_traps.jsonl. Dev split (40 pairs) for
  tuning; test split (80 pairs) for reporting only. Same rule for the complexity dataset: the
  load-test corpus draws from the TRAIN split only.
- Splitting is group-aware by split_group. A pytest must assert zero group leakage across splits.
- The length-only baseline (~45.3% logreg / ~46.0% tree) is a release gate: CI fails if it hits 50%.
- Build the mock provider FIRST. Everything is tested against it; real APIs are for demos only.
- One milestone at a time: read the SPEC section, propose a short plan, wait for approval.
- Test-first for: the Lua token bucket, cache key derivation + namespacing, singleflight, breaker FSM,
  fallback selection, feature extraction, savings attribution math.
- Integration tests use testcontainers (real Redis + Postgres). Do not mock Redis for bucket/cache tests.
- Commands: make up, seed, train, test, loadtest, drill, harvest.
- Style: Python 3.11, full type hints, Pydantic v2, ruff + mypy pass required.
- All provider calls go through gateway/adapters (cost tracking + MAX_DAILY_SPEND cap).
- Nontrivial choices → docs/adr/NNNN-*.md. No LangChain, Celery, or Kafka. Do not rewrite in Go.
- Small commits per subtask. Secrets via .env only (.env.example provided).
```

**First message to Claude Code:**
> Read SPEC.md §0–§6 and CLAUDE.md. We're starting M0. Do not write application code yet — propose a
> plan for M0 only: repo scaffolding, dataset freeze + C0 validators/splits/length-baseline gate,
> docker-compose (gateway, mock provider, Redis Stack, Postgres, Prometheus, Grafana), Alembic
> baseline, Makefile targets, the LLM adapter with cost tracking and MAX_DAILY_SPEND kill-switch, and
> the mock chaos provider. Then stop and wait for my approval.

**Two things you write yourself, not Claude Code:** the **Lua token-bucket script** and the
**circuit-breaker FSM test table**. Those plus the singleflight test are what interviewers poke
hardest (§12). If you can't reconstruct them from memory, have Claude Code walk you through the diff
line by line before moving on — that review time is interview prep, not overhead.

---

## 15. Definition of Done

Clean machine: `git clone` → `make up seed train loadtest drill harvest` → Grafana populated,
`results/` generated, every §10 number real and reproducible, README (measured results + known gaps +
Mermaid diagram + Argus cross-link) + 9 ADRs + Loom done, pytest green (incl. dataset contracts and
the length-control gate), spend < $20. Then update the resume with real numbers, pin both repos, and
go do LeetCode.
