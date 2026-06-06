# Vault Cleanup, Claude Code Migration & RecSys Course — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean the Obsidian vault (zero dangling links/orphans, no duplicate concepts), migrate the authoring rulebook to a Claude-Code-native `CLAUDE.md`, and add a complete Recommender Systems course (MOC + 5 lecture notes + wired concept notes) built from 5 lecture PDFs.

**Architecture:** A read-only Python audit script (`docs/superpowers/tools/vault_audit.py`) is the verification harness for the whole effort — it must end at 0 dangling links and 0 unintended orphans. Phase A is deterministic cleanup + the instruction migration. Phase B is workflow-driven: parallel read-only PDF extraction → concept reconciliation (fix canonical names before any parallel writing) → parallel note creation → MOC/graph wiring → completeness review. Work happens on branch `vault-cleanup-and-recsys` with incremental commits.

**Tech Stack:** Markdown + Obsidian (wikilinks, YAML frontmatter, callouts, LaTeX, Mermaid); Python 3 stdlib for the audit; Claude Code `Workflow`/`Agent` tools for extraction and generation; the `Read` tool for PDF page ranges and images.

**Spec:** `docs/superpowers/specs/2026-06-06-vault-cleanup-and-recsys-design.md`

---

## File Structure

**Created:**
- `docs/superpowers/tools/vault_audit.py` — read-only audit (verification harness)
- `CLAUDE.md` — migrated authoring rulebook (vault root, auto-loaded)
- `Templates/Template - Course Overview (MOC).md` and 5 sibling templates
- `Courses/RL/Exam Prep/RL - Exam Cheat Sheet.md`, `Courses/RL/Exam Prep/RL - Exam 2024 Analysis.md` (stubs)
- `Courses/RecSys/RecSys - Overview.md` (MOC)
- `Courses/RecSys/Lectures/RS-L01..RS-L04` (5 lecture notes incl. L03a/L03b)
- `Concepts/<new RecSys concepts>.md` (final list from extraction)

**Modified:**
- `VAULT-INSTRUCTIONS.md` — deleted after migration
- `.obsidian/app.json` — add `userIgnoreFilters` for `CLAUDE.md`, `docs/`
- `.obsidian/graph.json` — add RecSys color group
- `Courses/IR/IR - Overview.md` — remove dead `[[IR-L01 - Introduction]]` link, fix `[[Concepts/]]`
- `Courses/RL/RL - Overview.md` — fix `[[Concepts/]]` folder reference
- Various `Concepts/*.md` — dedup merges, `course:` additions for reuse, backlink fixes

---

## PHASE A — Cleanup & Claude Code Migration

### Task A1: Build the audit script (verification harness)

**Files:**
- Create: `docs/superpowers/tools/vault_audit.py`

- [ ] **Step 1: Write the audit script**

```python
#!/usr/bin/env python3
"""Read-only vault audit. Run from anywhere:
    python3 docs/superpowers/tools/vault_audit.py
Reports dangling wikilinks, orphan notes (no inbound links), and frontmatter gaps.
Exit code 0 iff there are no dangling links, else 1."""
from __future__ import annotations
import re, sys
from pathlib import Path

VAULT = Path(__file__).resolve().parents[3]      # docs/superpowers/tools -> vault root
SKIP_DIRS = {'.git', '.obsidian', 'docs', '.vscode', 'Assets', 'Templates'}
SKIP_FILES = {'CLAUDE.md', 'VAULT-INSTRUCTIONS.md', 'README.md'}

wikilink_re = re.compile(r'\[\[([^\[\]|#]+)(?:[#|][^\[\]]*)?\]\]')
fm_re = re.compile(r'\A---\s*\n(.*?)\n---\s*\n', re.DOTALL)

def iter_md():
    for p in sorted(VAULT.rglob('*.md')):
        rel = p.relative_to(VAULT)
        if set(rel.parts) & SKIP_DIRS or p.name in SKIP_FILES:
            continue
        yield p

def get_fm(text):
    m = fm_re.match(text)
    return m.group(1) if m else None

def fm_field(fm, key):
    if fm is None:
        return None
    m = re.search(rf'^{key}\s*:\s*(.*)$', fm, re.MULTILINE)
    return m.group(1).strip() if m else None

def aliases_of(fm):
    raw = fm_field(fm, 'aliases')
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith('['):
        raw = raw[1:-1]
    return [a.strip().strip('"\'') for a in raw.split(',') if a.strip()]

def main():
    files = list(iter_md())
    resolve = {}                 # lowercase name/alias -> Path
    notes = {}                   # Path -> info dict
    for p in files:
        text = p.read_text(encoding='utf-8', errors='replace')
        fm = get_fm(text)
        notes[p] = {'fm': fm, 'text': text, 'inbound': set()}
        for k in {p.stem.lower(), *(a.lower() for a in aliases_of(fm))}:
            resolve.setdefault(k, p)

    dangling = []
    for p, info in notes.items():
        for m in wikilink_re.finditer(info['text']):
            if m.start() > 0 and info['text'][m.start() - 1] == '!':
                continue                       # image/embed, not a link
            target = m.group(1).strip()
            if not target:
                continue
            key = target.lower()
            base = key.rstrip('/').split('/')[-1]
            tgt = resolve.get(key) or resolve.get(base)
            if tgt is None:
                dangling.append((str(p.relative_to(VAULT)), target))
            elif tgt != p:
                notes[tgt]['inbound'].add(p)

    orphans = [str(p.relative_to(VAULT))
               for p, info in notes.items()
               if not info['inbound'] and fm_field(info['fm'], 'type') != 'moc']

    fm_gaps = []
    for p, info in notes.items():
        rel = str(p.relative_to(VAULT))
        if info['fm'] is None:
            fm_gaps.append((rel, 'no frontmatter')); continue
        for key in ('type', 'status'):
            if fm_field(info['fm'], key) is None:
                fm_gaps.append((rel, f'missing {key}'))

    print(f"# Vault audit — {len(files)} notes\n")
    print(f"## Dangling links ({len(dangling)})")
    for f, t in dangling:
        print(f"  - {f}  ->  [[{t}]]")
    print(f"\n## Orphans / no inbound links ({len(orphans)})")
    for f in orphans:
        print(f"  - {f}")
    print(f"\n## Frontmatter gaps ({len(fm_gaps)})")
    for f, why in fm_gaps:
        print(f"  - {f}: {why}")
    ok = not dangling
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"(dangling={len(dangling)}, orphans={len(orphans)}, fm_gaps={len(fm_gaps)})")
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it against the current vault (expect FAIL)**

Run: `python3 docs/superpowers/tools/vault_audit.py`
Expected: a FAIL result whose Dangling-links section includes at least
`Courses/IR/IR - Overview.md -> [[IR-L01 - Introduction]]` and a `[[Concepts/]]` entry.
This confirms the harness detects the real, known issues.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/tools/vault_audit.py
git commit -m "Add read-only vault audit script (link/orphan/frontmatter harness)"
```

---

### Task A2: Triage the audit report

**Files:** none (analysis step; output drives A3–A5)

- [ ] **Step 1: Capture the full report**

Run: `python3 docs/superpowers/tools/vault_audit.py > /tmp/audit_before.txt; cat /tmp/audit_before.txt`

- [ ] **Step 2: Classify every dangling link** into one of: `typo` (fix the link text),
`stub` (the target is genuinely planned — create a stub), or `dead` (remove the link).
Known going in: `[[IR-L01 - Introduction]]` → **dead** (L1.1 is admin, lives in the MOC);
`[[Concepts/]]` (in RL + IR MOCs) → **dead** (replace with prose, it's a folder);
`[[RL - Exam 2024 Analysis]]`, `[[RL - Exam Cheat Sheet]]` → **stub**.

- [ ] **Step 3: Note each orphan** and the MOC/concept it should be wired into.
No commit (analysis only).

---

### Task A3: Fix dangling links and orphans

**Files:**
- Modify: `Courses/IR/IR - Overview.md`, `Courses/RL/RL - Overview.md`, and any other note surfaced in A2.

- [ ] **Step 1: Remove the dead `[[IR-L01 - Introduction]]` link.**
In `Courses/IR/IR - Overview.md`, the Week-1 table row `L1.1 | Administration & Course Intro`
currently links `[[IR-L01 - Introduction]]`. Replace that cell's link with plain text
`— (admin; see metadata above)` so the row stays but the dead link is gone.

- [ ] **Step 2: Fix the `[[Concepts/]]` folder references.**
In both MOCs, replace `see [[Concepts/]] folder` with `see the **Concepts/** folder`
(plain bold text, not a wikilink).

- [ ] **Step 3: Wire each orphan** from A2 into its MOC's Concept Index and/or add the
missing `[[link]]` from the lecture/exercise that should reference it. (Specific edits depend
on the A2 list; each is a one-line link addition.)

- [ ] **Step 4: Re-run the audit.**
Run: `python3 docs/superpowers/tools/vault_audit.py`
Expected: dangling count dropped to only the **stub** category from A2 (exam-prep links);
orphans reduced.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Fix dangling links and wire orphan notes"
```

---

### Task A4: Populate Templates/ and Exam Prep stubs

**Files:**
- Create: `Templates/Template - Course Overview (MOC).md`
- Create: `Templates/Template - Lecture Note.md`
- Create: `Templates/Template - Book Chapter Note.md`
- Create: `Templates/Template - Concept Note.md`
- Create: `Templates/Template - Exercise.md`
- Create: `Templates/Template - Coding Assignment.md`
- Create: `Courses/RL/Exam Prep/RL - Exam Cheat Sheet.md`
- Create: `Courses/RL/Exam Prep/RL - Exam 2024 Analysis.md`

- [ ] **Step 1: Write the Concept template** (the others mirror the rulebook frontmatter + structure):

```markdown
---
type: concept
aliases: []
course: []
tags: []
status: stub
---

# <Concept Name>

## Definition
> [!definition] <Concept Name>
> <precise, formal statement>

## Intuition
> [!intuition] <plain-language explanation / analogy>

## Mathematical Formulation
$$<formula>$$

where:
- <symbol> — <meaning>

## Key Properties / Variants
- <bullet>

## Connections
- Related to: [[<Concept>]]

## Appears In
- [[<Lecture / Book / Exercise note>]]
```

- [ ] **Step 2: Write the other 5 templates** using the exact frontmatter blocks from the
rulebook (`moc`, `lecture`, `book-chapter`, `exercise`, `coding-assignment`) plus the section
skeletons described there. Each starts `status: stub`.

- [ ] **Step 3: Write the two Exam Prep stubs.** Each:

```markdown
---
type: exam-prep
course: RL
status: stub
---

# RL - Exam Cheat Sheet

> [!note] Placeholder
> Single-page formula sheet for last-minute cramming. To be filled from the concept notes
> tagged `#key-formula`. Linked from [[RL - Overview]].
```

(Second stub analogous, title `RL - Exam 2024 Analysis`, scope = past-exam topic breakdown.)

- [ ] **Step 4: Re-run the audit.**
Run: `python3 docs/superpowers/tools/vault_audit.py`
Expected: **0 dangling links** (the exam-prep stubs now resolve). Templates are skipped by the
audit (`SKIP_DIRS`), so they will not be flagged. RESULT: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Populate Templates/ and add Exam Prep stub notes"
```

---

### Task A5: Migrate VAULT-INSTRUCTIONS.md → CLAUDE.md

**Files:**
- Create: `CLAUDE.md`
- Delete: `VAULT-INSTRUCTIONS.md`

- [ ] **Step 1: Create `CLAUDE.md` as the full content of `VAULT-INSTRUCTIONS.md` with these exact edits:**
  1. Title line `# Vault Instructions — For Bob (AI Assistant)` → `# CLAUDE.md — Authoring Rulebook for this Vault`.
  2. Replace every "Bob" / "the AI assistant" reference with "Claude".
  3. **§5.2 "Handling Images from Source PDFs":** replace `Use the `image` tool` with
     `Use the **Read** tool — it reads PDF page ranges (e.g. `Read(path, pages="1-20")`, max 20
     pages/request) and renders images directly. Process every figure.`
  4. **§6 "Sub-Agent Usage":** rewrite for Claude Code. Replace `sessions_spawn` with the
     `Agent` tool and `Workflow` orchestration. Keep the *when to / when not to* guidance and
     the post-subagent checklist. Add: "Fix canonical concept names BEFORE dispatching parallel
     note-writing agents, so wikilinks are consistent across notes."
  5. Add a new short section **"0. Working in this vault with Claude Code"** near the top:
     - Source materials (lecture PDFs, books) live in `/home/stas/Desktop/University/<COURSE>/`
       (e.g. `RECSYS/`, `RL/`, `IR/`); the git repo holds only notes + `docs/`.
     - Run `python3 docs/superpowers/tools/vault_audit.py` after edits; it must end PASS
       (0 dangling links).
     - A sibling `../university-vault-site/` publishes this vault — **never rename a kept note**
       (filenames anchor wikilinks and published URLs).
  6. Leave §1–4, 7–9 substantively intact (purpose, completeness, note types, formatting,
     maintenance, quality bar, naming) — these are good.

- [ ] **Step 2: Delete the old file.**
Run: `git rm VAULT-INSTRUCTIONS.md`

- [ ] **Step 3: Verify.** Run: `python3 docs/superpowers/tools/vault_audit.py` → RESULT: PASS.
Confirm `CLAUDE.md` contains no "Bob", "sessions_spawn", or "`image` tool" strings:
Run: `grep -nE 'Bob|sessions_spawn|image. tool' CLAUDE.md || echo "clean"` → `clean`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Migrate VAULT-INSTRUCTIONS.md to Claude-Code-native CLAUDE.md"
```

---

### Task A6: De-duplicate concept notes (semantic)

**Files:**
- Modify/Delete: `Concepts/*.md` (only true duplicates)

- [ ] **Step 1: Run a semantic dedup scan.** Dispatch a workflow (or read directly) that groups
the 190 concept titles + first paragraphs and flags **true duplicates** (same concept, two
files) vs **distinct-but-related** (e.g. `Neural Networks` vs `Convolutional Neural Networks` —
keep both). Produce a list: `{survivor, duplicate, reason}`.

- [ ] **Step 2: For each true duplicate:** fold any unique content from the duplicate into the
survivor; add the duplicate's title to the survivor's `aliases:` (so existing `[[links]]` still
resolve); `git rm` the duplicate file. Do **not** rename the survivor.

- [ ] **Step 3: Re-run the audit.**
Run: `python3 docs/superpowers/tools/vault_audit.py`
Expected: PASS, 0 dangling (aliases preserve all links). Note any borderline near-dupes left
intact in the commit body.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Merge duplicate concept notes (aliases preserve links); report near-dupes"
```

---

### Task A7: Obsidian ignore filter

**Files:**
- Modify: `.obsidian/app.json`

- [ ] **Step 1: Add ignore filters** so `CLAUDE.md` and `docs/` don't clutter the graph/explorer.
`app.json` is currently `{}`. Write:

```json
{
  "userIgnoreFilters": [
    "CLAUDE.md",
    "docs/"
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add .obsidian/app.json
git commit -m "Hide CLAUDE.md and docs/ from Obsidian file explorer"
```

---

## PHASE B — RecSys Course

### Task B1: Deep PDF extraction (read-only workflow)

**Files:** none written (produces structured extraction in-workflow)

- [ ] **Step 1: Run an extraction workflow.** One agent per PDF reads **all** pages (Read with
20-page ranges) and returns a structured object: `{lecture, title, lecturer, outline[],
detailed_content (per section: theory/formulas/algorithms/figures-described), worked_examples[],
concepts_used[{name, one_line_def, reuse_likely}]}`. Read-only; safe to parallelize across the
5 PDFs. Source paths from spec §5.2. **Do not truncate** Lecture 2 (101 pp) or Lecture 4 (74 pp);
if any coverage is bounded, `log()` what was dropped.

- [ ] **Step 2: Persist the extraction** to `/tmp/recsys_extraction.json` (working scratch, not
committed) for the downstream tasks. No vault commit.

---

### Task B2: Reconcile the global concept inventory

**Files:** none (analysis; fixes canonical names before any parallel writing)

- [ ] **Step 1: Merge** the 5 `concepts_used` lists into one deduplicated set.

- [ ] **Step 2: Classify each** against the existing `Concepts/` folder:
  - **reuse** — an existing note covers it (e.g. `Precision at K`, `Recall`, `NDCG`, `MRR`,
    `MAP`, `Generative Retrieval`, `Transformers`, `BERT for IR`, `PPO`, `GRPO`). Record the
    exact existing filename as the canonical wikilink.
  - **new** — needs a note. Record the canonical filename (plain name; `(RecSys)` suffix only
    when the meaning is genuinely distinct from an existing same-named concept).

- [ ] **Step 3: Output** `/tmp/recsys_concepts.json` = `{new:[...], reuse:[...]}` with canonical
names. This list is frozen and handed to every writer agent so wikilinks match. No commit.

---

### Task B3: Create new RecSys concept notes (parallel write)

**Files:**
- Create: `Concepts/<each new concept>.md`

- [ ] **Step 1: Dispatch one writer agent per new concept.** Each gets: canonical name, the
relevant extracted content, the **full frozen concept-name list** (for consistent wikilinks),
and the Concept template/structure from `CLAUDE.md` §3.4. Each writes exactly one file with
correct frontmatter (`type: concept`, `course: [RecSys]` (+others if shared), `aliases`, `tags`,
`status: complete`), Definition → Intuition → Math → Properties/Variants → Connections →
Appears In (linking the RecSys lecture(s) it came from). Different files + frozen names ⇒ safe to
parallelize.

- [ ] **Step 2: Acceptance check per note:** has all 6 sections, valid frontmatter, every
`[[link]]` is either in the frozen list or an existing note. Spot-fix any agent that emitted a
non-canonical link.

- [ ] **Step 3: Commit**

```bash
git add Concepts/
git commit -m "Add RecSys concept notes"
```

---

### Task B4: Extend reused concept notes

**Files:**
- Modify: each `reuse` concept from B2.

- [ ] **Step 1: For each reused concept,** add `RecSys` to its `course:` list and add the
relevant `[[RS-L0x ...]]` lecture note to its **Appears In** section. Do not otherwise rewrite.

- [ ] **Step 2: Commit**

```bash
git add Concepts/
git commit -m "Link shared concepts to RecSys (course field + Appears In backlinks)"
```

---

### Task B5: Create the 5 RecSys lecture notes (parallel write)

**Files:**
- Create: `Courses/RecSys/Lectures/RS-L01 - Course Overview & Introduction.md`
- Create: `Courses/RecSys/Lectures/RS-L02 - Evaluation Beyond Accuracy.md`
- Create: `Courses/RecSys/Lectures/RS-L03a - Sequential Recommendation Models.md`
- Create: `Courses/RecSys/Lectures/RS-L03b - From LLMs to LRMs.md`
- Create: `Courses/RecSys/Lectures/RS-L04 - Generative Recommendation.md`

- [ ] **Step 1: Dispatch one writer agent per lecture.** Each gets: its B1 extraction, the
frozen concept-name list, the Lecture template (`CLAUDE.md` §3.2), and the exam-substitute
completeness bar. Frontmatter: `type: lecture`, `course: RecSys`, `lecture: <n>`,
`date:` (L1=2026-06-01, L2=2026-06-02, L3a/L3b=2026-06-04, L4=2026-06-05),
`topics:` (wikilinks from the frozen list), `status: complete`. Body mirrors the slide flow;
**every figure reproduced** (Mermaid/ASCII/table/described); all formulas in LaTeX with
term explanations; concepts wikilinked. Different files + frozen names ⇒ safe to parallelize.

- [ ] **Step 2: Acceptance check per lecture:** reading it (plus linked concepts) substitutes
the slides; no "see slide X"; no skipped figures; frontmatter valid; links resolve.

- [ ] **Step 3: Commit**

```bash
git add Courses/RecSys/Lectures/
git commit -m "Add RecSys lecture notes RS-L01..RS-L04"
```

---

### Task B6: Create the RecSys Overview MOC

**Files:**
- Create: `Courses/RecSys/RecSys - Overview.md`

- [ ] **Step 1: Write the MOC** (main agent, not parallel — it links everything). Frontmatter
`type: moc`, `course: RecSys`, `tags: [moc]`. Sections: course metadata (spec §5.2 — title,
UvA MSc AI/IRLab, lecturers, June 1–5 2026, *Recommender Systems Handbook* (Ricci 2011)); a
**clearly-marked Assessment section** with a placeholder line for Stanisław to fill project
deadlines/grading; a lecture table linking all 5 notes; a Concept Index grouping the new +
reused RecSys concepts (mirroring the RL/IR MOC style).

- [ ] **Step 2: Commit**

```bash
git add Courses/RecSys/
git commit -m "Add RecSys course Overview MOC"
```

---

### Task B7: Graph color group + final wiring

**Files:**
- Modify: `.obsidian/graph.json`

- [ ] **Step 1: Add a RecSys color group.** Append to the `colorGroups` array in
`.obsidian/graph.json` (blue, distinct from the two reddish RL/IR groups):

```json
{ "query": "path:Courses/RecSys  ", "color": { "a": 1, "rgb": 3447003 } }
```

- [ ] **Step 2: Commit**

```bash
git add .obsidian/graph.json
git commit -m "Add RecSys graph color group"
```

---

### Task B8: Completeness review + final audit

**Files:** spot-fixes only

- [ ] **Step 1: Run the audit.**
Run: `python3 docs/superpowers/tools/vault_audit.py`
Expected: RESULT PASS, 0 dangling links, 0 unexpected orphans, 0 frontmatter gaps among the new
notes. Fix anything flagged.

- [ ] **Step 2: Completeness-critic pass.** Dispatch a reviewer per RecSys lecture note asking:
"What from the slides is missing, what figure was skipped, what claim is unverified?" Apply fixes
its findings justify.

- [ ] **Step 3: Final commit.**

```bash
git add -A
git commit -m "Final RecSys review fixes; audit clean"
```

- [ ] **Step 4: Summary for the user.** Report: notes added, concepts created vs reused,
duplicates merged, dangling links fixed, and the audit PASS. Leave the branch for the user to
review and merge to `main`.

---

## Self-Review (against the spec)

**Spec coverage:**
- §4.1 CLAUDE.md migration → A5 ✓
- §4.2 structural audit + fixes (dangling/orphans/frontmatter/empty folders) → A1, A2, A3, A4 ✓
- §4.3 dedup → A6 ✓
- §5.1 structure → B6 (MOC) + B5 (Lectures) ✓
- §5.3 lecture notes → B5 ✓
- §5.4 concepts (reuse + new) → B2, B3, B4 ✓
- §5.5 wiring (MOC, graph, backlinks) → B4, B6, B7 ✓
- §6 execution approach → Phase A/B task ordering ✓
- §7 acceptance criteria → enforced by B8 audit PASS + per-note checks ✓
- Judgment call "hide CLAUDE.md/docs from Obsidian" → A7 ✓

**Placeholder scan:** Generative content (note bodies) is produced by extraction-driven writer
agents, not pre-written here — this is the correct granularity for PDF→note synthesis, not a
placeholder. All deterministic artifacts (audit script, templates, JSON edits, exact link fixes,
CLAUDE.md edit list) are fully specified.

**Type consistency:** the audit script's resolver (stem + aliases, lowercased) is the same
contract relied on by A6 (merges preserve names as aliases) and B3/B5 (frozen canonical names) —
consistent throughout.
