# Session Handover — pick up here

## Summary

**Where things stand:** App is live at `https://meeting-audio-processor.onrender.com/?token=<ACCESS_TOKEN>`. Currently running on Claude Sonnet 4.6 for MoM extraction (~₹25/meeting). We tried Gemini Flash 2.5 free and DeepSeek V3 paid as cheaper alternatives — both produced unacceptable output (collapsed 12 decisions into 3, dropped safety items). Plan to test Claude Haiku 4.5 and DeepSeek-Reasoner R1 next as final cost-optimisation attempts before accepting Sonnet as permanent.

**Aim of the next session:** Land two small code edits, then A/B-test Haiku and Reasoner against Sonnet on a real meeting. Pick a production default based on the results.

**High-level steps:**
1. Read the prepared plan at `~/.claude/plans/also-can-t-we-use-lively-adleman.md` — it has the exact code edits and verification steps
2. Apply the two code edits in `process.py` (one-liner each), plus small docs updates
3. Push to main — Render auto-deploys
4. A/B test Haiku → if quality passes, set as default. If not, A/B test Reasoner → same gate. If both fail, stay on Sonnet.

---

## Critical files to read first (in this order)

1. `~/.claude/plans/also-can-t-we-use-lively-adleman.md` — the exact code-change plan, already approved by Chetan, ready to implement
2. `HANDOVER.md` in this repo — full project context (architecture, env vars, R2 setup, all branches, etc.)
3. `~/.claude/projects/-Users-chetanmanjesh/memory/project_meeting_processor.md` — project-level context that persists across sessions
4. This file — the bridge between this session and the next

---

## The exact next work (from the plan file)

Two edits in `process.py`:

**Edit 1 — line 31** make `CLAUDE_MODEL` env-var configurable:
```python
# Replace:
CLAUDE_MODEL = "claude-sonnet-4-6"
# With:
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
```

**Edit 2 — lines 339-349** apply `response_format` conditionally in `_deepseek_stream_json` (Reasoner doesn't support it):
```python
create_kwargs = dict(
    model=DEEPSEEK_MODEL,
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": user_content}],
    max_tokens=max_tokens,
    temperature=0.2,
    stream=True,
)
if DEEPSEEK_MODEL.startswith("deepseek-chat"):
    create_kwargs["response_format"] = {"type": "json_object"}
stream = client.chat.completions.create(**create_kwargs)
```

Plus small updates to `.env.example`, `render.yaml`, and the env-var table in `HANDOVER.md` (add `CLAUDE_MODEL` row). Total change ≈ 15 lines.

After deploy:
- Test 1: Set `CLAUDE_MODEL=claude-haiku-4-5` on Render → upload meeting → compare docx to existing `mom_detailed_claude.docx` (Sonnet baseline)
- Test 2: Set `LLM_PROVIDER=deepseek` + `DEEPSEEK_MODEL=deepseek-reasoner` → upload same meeting → compare

**Pass criteria for either:** decision count within ~15% of Sonnet's 12, sign-offs section non-empty, **both safety items** present (fire-exit obstruction AND mid-landing height non-compliance — these are the deal-breakers; V3 and Gemini both dropped them), materials section ≥3 items.

---

## Test audio and reference outputs

The benchmark audio for comparisons is whatever was used to produce these existing files in `~/Downloads/`:
- `mom_detailed_claude.docx` — Sonnet reference (12 decisions, 2 sign-offs, 4 materials, 8 action items, both safety items)
- `mom_detailed_gemini.docx` — Gemini Flash 2.5 free output (failed — 3 decisions, 0 sign-offs, safety dropped)
- `mom_detailed_deepseek.docx` — DeepSeek V3 paid output (failed — 3 decisions, 1 sign-off, 1 material, safety dropped)

Use the **same audio** for Haiku and Reasoner tests so the comparison is apples-to-apples.

## Comparison script (drop into `/tmp/`)

The structured-diff script used for previous comparisons is at `/tmp/mom_comparison_v3.json` (in this machine's `/tmp/`, possibly cleared on reboot). The pattern is reproducible:
```python
from docx import Document
# Parse paragraphs, bucket by section heading, count bullets/sub-headings,
# parse the action-items table, surface summary char count.
# See server.py's existing rendering helpers for the section names.
```
A future Claude session can re-derive this in 30 lines if needed.

---

## State of branches and what's deployed

- `main` is at commit `98376fc` ("Add DeepSeek V3 as third LLM provider"). Pushed and deployed.
- All branches consolidated into `main`. No outstanding work-in-progress branches.
- Render auto-deploys on push to `main` (build ~3 min).

## Current Render env vars (as of last session)

- `LLM_PROVIDER=claude` (default, working)
- `CLAUDE_MODEL` not yet set as env var (currently hardcoded; this is what the next edit fixes)
- `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`, `SARVAM_API_KEY`, `R2_*`, `ACCESS_TOKEN` — all set
- `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-2.0-flash` — set but currently inactive (Gemini route tested and dismissed)
- `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL=deepseek-chat` — set but currently inactive (V3 tested and dismissed)

## Pattern observed across cheap-model attempts

Both Gemini Flash 2.5 (free tier) and DeepSeek V3 (paid) produced the **same failure mode** on this prompt: aggressively collapsed 12 distinct decisions into 3, dropped the entire sign-offs section or shrank it to 1 item, lost safety-flagged items. This isn't a prompt problem (Sonnet on the same prompt produces 12 granular items) — it's instruction-following depth at the smaller/cheaper model class.

That's why the next attempts target sibling models that may inherit better instruction-following:
- **Claude Haiku 4.5** — Sonnet's family, possibly inherits the discipline
- **DeepSeek-Reasoner R1** — reasoning step often improves complex-instruction adherence

If both fail too, the honest conclusion is: cost optimisation isn't winnable on this prompt/schema combination, and ~₹25/meeting on Sonnet is the right operating cost for the studio.

---

## What NOT to do (rabbit holes already explored)

- **Don't suggest self-hosting Qwen / Llama on the Oracle Ampere A1.** Already analysed — single-user latency is 40-130 min per meeting depending on model size. Plus smaller models drop facts. Archived in the plan file under "Self-hosting research".
- **Don't suggest switching providers without a quality test first.** Two providers already failed by skipping that step.
- **Don't suggest Gemini Pro paid** as the next thing to try — pricing makes it worse than DeepSeek-Reasoner, and we don't know it'd do better.
- **Don't reformat the docx template, prompts, or schema.** Those are stable and not what's failing.

---

## How to resume cleanly in a fresh session

1. Read `~/.claude/projects/-Users-chetanmanjesh/memory/MEMORY.md` first (auto-loaded anyway)
2. Read this file (`SESSION_HANDOVER.md`)
3. Read the plan file (`~/.claude/plans/also-can-t-we-use-lively-adleman.md`)
4. Verify deployed commit: `cd ~/Desktop/dashboard_peggy/meeting-audio-processor && git log --oneline -3 main`
5. Apply the two edits in `process.py`
6. Commit, push, wait for Render to redeploy
7. Run the Haiku A/B test → then Reasoner if needed
8. Pick a winner, update production env var

The next session should NOT have to re-derive what's already known. Everything's documented above; trust the prior analysis.
