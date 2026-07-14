# Relay Complexity Tiering Rubric

**Dataset version:** 1.1.0  
**Status:** Frozen, human-verified release.  
**Corpus:** 600 prompts; final tiers 205/196/199; 34 logged bootstrap corrections.

1. **Tier 1 - bounded work:** extraction, reformatting, lookup, direct classification, or one straightforward transformation.
2. Tier 1 has a local answer, little synthesis, and normally one obvious method; short deterministic code can qualify.
3. **Tier 2 - structured work:** summarization, comparison, multi-item classification, moderate planning, or analysis using a known method.
4. Tier 2 may combine several explicit constraints or moderate context, but it does not require deep search over alternatives.
5. **Tier 3 - reasoning-intensive work:** multi-step reasoning, proof, architecture, complex debugging, nuanced judgment, research design, or heavily constrained creation.
6. A single critical subtask that genuinely needs a higher tier raises the whole prompt to that tier.
7. Tier is a property of the required work, not prompt length or requested output length; long deterministic work can be Tier 1 and short nuanced judgment can be Tier 3.
8. Words such as "analyze," "compare," "design," and "prove" are cues, not labels; judge the actual operations and constraints.
9. Choose the lowest tier that can reliably satisfy every explicit requirement, not the cheapest tier that might occasionally work.
10. When a boundary remains ambiguous, record both candidates and route upward at runtime; adjudication still assigns one dataset label.

**Split contract:** Split by `split_group`/`template_group`, never by individual row. Every row carries `length_confound_stratum` as `long_tier1`, `short_tier3`, or `normal`. Group-aware splitting must preserve these audit strata and produce zero template-group leakage.

**Length-only control:** Run `python scripts/length_baseline.py`. The committed control uses prompt character count as its only feature, seven `StratifiedGroupKFold` folds, `split_group` as the grouping key, tier as the stratification target, and seed `42`. Pooled out-of-fold accuracy is **45.3%** for logistic regression (`C=0.3`) and **46.0%** for a depth-3 decision tree (`min_samples_leaf=5`, balanced class weights). Both controls must remain below 50%; `tests/test_dataset_contract.py` enforces this release gate.
