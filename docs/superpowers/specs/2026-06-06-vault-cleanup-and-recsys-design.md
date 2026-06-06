# Design Spec — Vault Cleanup, Claude Code Migration & RecSys Course

**Date:** 2026-06-06
**Branch:** `vault-cleanup-and-recsys`
**Author:** Claude (with Stanisław)
**Status:** Approved design → ready for implementation plan

---

## 1. Context

`university-vault` is an Obsidian vault that is Stanisław's single source of truth for
UvA MSc-AI coursework. Two courses are currently captured — **Reinforcement Learning (RL)**
and **Information Retrieval 1 (IR)** — across 245 markdown notes:

- `Concepts/` — 190 flat, course-agnostic atomic concept notes (the wiki layer)
- `Courses/<COURSE>/` — per-course MOC, `Lectures/`, `Book Notes/`, `Exercises/`,
  `Coding Assignments/`, `Exam Prep/`
- `Templates/` — referenced by the instructions but **currently empty**
- `Assets/` — figures
- `VAULT-INSTRUCTIONS.md` — the authoring rulebook, written for the prior assistant
  ("Bob" / OpenClaw) and referencing OpenClaw-specific tooling (`sessions_spawn`, an
  `image` tool)

A sibling directory `../university-vault-site/` is a published mirror — **out of scope, do
not touch** — but it means **filenames must stay stable** (they anchor wikilinks and URLs).

The work has two goals, to be done in order:

1. **Clean up & migrate to Claude Code** — remove dangling links, orphans, and duplicate
   notes; make the authoring instructions clear and Claude-Code-native.
2. **Add the Recommender Systems (RecSys) course** — high-quality lecture/wiki notes built
   from 5 lecture PDFs. The course has *no textbook* and is otherwise project-based, so the
   lecture notes must be especially complete (they are the only durable record).

## 2. Goals & Non-Goals

**Goals**
- A clean knowledge graph: zero dangling links, zero unintended orphans, no duplicate concepts.
- A single, Claude-Code-native authoring rulebook (`CLAUDE.md`).
- A complete RecSys course section: Overview MOC + 5 lecture notes + concept notes, fully
  wired into the existing concept graph (reusing shared concepts).

**Non-Goals**
- No changes to `../university-vault-site/`.
- No RecSys project note, exercises, coding assignments, or book notes (none exist for the course).
- No rewrite of existing RL/IR *content* beyond fixing the audit findings (links, orphans,
  dupes, frontmatter). We are not re-authoring correct notes.
- No restructuring of the folder taxonomy — it works; we conform to it.

## 3. Conventions (authoritative, carried from the rulebook)

These are the rules every new/edited note must follow. They are summarized here so this spec
is self-contained; the full rulebook becomes `CLAUDE.md` (see §4.1).

- **Completeness bar:** a note must substitute its source. Reading the note (plus linked
  concepts) is enough to pass the exam — no "see the slides/book".
- **Note types & frontmatter:** `moc`, `lecture`, `book-chapter`, `concept`, `exercise`,
  `coding-assignment`. Every note has a `status` of `complete | draft | stub`.
- **Concepts:** one concept per note, course-agnostic, flat folder, `course:` is a list.
  Structure: Definition → Intuition → Mathematical Formulation → Key Properties/Variants →
  Connections → Appears In.
- **Math** in LaTeX with term-by-term explanation. **Callouts** (`definition`, `formula`,
  `intuition`, `example`, `warning`, `tip`) for scannability. **Algorithms** as full
  pseudocode. **Images** from sources are reproduced (Mermaid/ASCII/table/described), never
  "see Figure X"; embed to `Assets/` only when irreproducible in text.
- **Links:** always wikilink concepts; use aliases where grammar needs them; concepts
  back-link via "Appears In"; lectures link to their concepts and book chapters.
- **Naming:** `RL-L01 - Title.md`, `RL-Book Ch5 - Title.md`, `RL-ES03 - ...`, `RL-HW01 - ...`,
  `RL-CA01 - ...`, concepts as plain `Name.md`.

---

## 4. Part A — Cleanup & Claude Code Migration

### 4.1 Instructions → `CLAUDE.md`

Convert `VAULT-INSTRUCTIONS.md` into a root **`CLAUDE.md`** (auto-loaded by Claude Code).

- **Keep verbatim-in-spirit:** purpose, completeness requirements, note types & templates,
  formatting standards, the per-topic workflow, maintenance rules, quality bar, naming
  conventions. This content is good; preserve it.
- **Rewrite the tooling-coupled parts:**
  - "For Bob (AI Assistant)" → Claude Code framing.
  - §6 "Sub-Agent Usage / `sessions_spawn`" → Claude Code subagents (the `Agent` tool /
    `Workflow` orchestration): when to fan out per-lecture/per-concept, when to keep MOC and
    cross-referencing concept work on the main agent, and the post-subagent review checklist.
  - §5.2 the `image` tool → the **Read tool**: PDFs are read with page ranges
    (`Read(file, pages="1-4")`), images are read directly. "Process all images" stays.
  - Add a short **"Source materials"** section: lecture PDFs and books live under
    `/home/stas/Desktop/University/<COURSE>/` (e.g. `RECSYS/`, `RL/`, `IR/`); the vault repo
    holds only notes.
- **Delete** `VAULT-INSTRUCTIONS.md` once content is fully migrated (single source of truth).

### 4.2 Structural audit + fixes (deterministic)

Run a deterministic pass (scripted link/graph analysis; an LLM is not needed to find a broken
`[[link]]`). Produce a report, then apply fixes.

- **Dangling links** — every `[[target]]` that resolves to no note and no alias. Confirmed
  examples: `[[IR-L01 - Introduction]]`, `[[RL - Exam 2024 Analysis]]`, `[[Concepts/]]`
  (a folder, not a note). Resolution per case: fix a typo, create a `stub`, or remove a dead link.
  - `[[IR-L01 - Introduction]]`: the L1.1 slot is pure administration (which belongs in the
    MOC, not a lecture note) → remove the dead link from the IR MOC.
  - `[[Concepts/]]`: replace the folder reference with prose, not a link.
- **Orphans** — notes with no inbound links (other than possibly the MOC). Wire each into the
  correct MOC and/or relevant concept. `showOrphans` is on in the graph, so these are visible.
- **Backlink integrity** — each concept's "Appears In" should list the notes that actually
  link it, and each lecture's `topics` should link concepts that back-link to it. Repair both
  directions.
- **Frontmatter consistency** — every note has a valid `type` and `status`; `course` is
  present (and a list for concepts); aliases exist where links depend on them.
- **Empty folders:**
  - `Templates/` — create the 6 template files the rulebook describes (MOC, lecture,
    book-chapter, concept, exercise, coding-assignment) so the folder matches the docs and the
    Templates core-plugin works.
  - `Exam Prep/` — create `stub`-status placeholder notes for the linked-but-absent exam notes
    (`RL - Exam Cheat Sheet` and `RL - Exam 2024 Analysis`), eliminating those dangling links
    cleanly. Stubs carry a one-line scope note so they read as deliberate placeholders, not
    abandoned content.

### 4.3 Duplicate / near-duplicate resolution (semantic)

LLM semantic scan across all 190 concept notes for true duplicates and significant overlaps.

- **Merge true duplicates:** keep the best version, fold the other's unique content in, add the
  loser's name as an `alias` on the survivor, and repoint/verify all backlinks. (Filenames of
  *kept* notes stay stable; only genuine duplicate files are removed, and only because their
  name survives as an alias so existing links still resolve.)
- **Report near-duplicates** that are legitimately distinct (e.g. `Neural Networks` vs
  `Convolutional Neural Networks`) in the commit message — do not merge those.
- Per the approved "fix everything in one pass" decision, act and surface the changes in the
  diff/commit message rather than asking item-by-item.

---

## 5. Part B — RecSys Course

### 5.1 Structure

```
Courses/RecSys/
├── RecSys - Overview.md      (type: moc)
└── Lectures/
    ├── RS-L01 - Course Overview & Introduction.md
    ├── RS-L02 - Evaluation Beyond Accuracy.md
    ├── RS-L03a - Sequential Recommendation Models.md
    ├── RS-L03b - From LLMs to LRMs.md
    └── RS-L04 - Generative Recommendation.md
```

No `Book Notes/`, `Exercises/`, `Coding Assignments/`, `Exam Prep/`, or project note.
Concepts go in the shared flat `Concepts/`.

### 5.2 Source PDFs (in `/home/stas/Desktop/University/RECSYS/`)

| Note | PDF | Lecturer(s) | Pages |
|---|---|---|---|
| `RS-L01` | `Lecture_1.pdf` | de Rijke, Tang | 56 |
| `RS-L02` | `Lecture_2_Jun 1.pdf` | Rus | 101 |
| `RS-L03a` | `Lecture 3 - Part 1 - Sequential recommendation models.pdf` | Zhang | 48 |
| `RS-L03b` | `Lecture 3 - Part 2 -  From LLMs to LRMs.pdf` | Zhao | 42 |
| `RS-L04` | `Lecture 4 Generative Recommendation.pdf` | Mekonnen, Seputis | 74 |

Course metadata for the MOC: *Recommender Systems*, UvA MSc AI, IRLab; lectures June 1–5, 2026;
coordinators de Rijke & Yubao Tang; based in part on the *Recommender Systems Handbook*
(Ricci et al., 2011). The course is project-based; the MOC will include an assessment section
with a clearly-marked field for Stanisław to add project specifics (deadlines, grading) — admin
detail is owned by the user, not invented by Claude.

### 5.3 Lecture notes

`RS-` prefix, `type: lecture`, exam-substitute completeness, **every slide image processed**
(no textbook safety net). One note per PDF/part. `RS-L03a`/`RS-L03b` keep the course's own
split-lecture identity. Each lecture's `topics` frontmatter and body link the concepts below.

### 5.4 Concept notes

After deep extraction, classify every key concept as **reuse** or **new**.

- **Reuse** (extend the existing note's `course:` list to include `RecSys`, add an "Appears In"
  backlink to the RecSys lecture): evaluation metrics already in the vault — `Precision at K`,
  `Recall`, `NDCG`, `MRR`, `MAP`; and model/architecture concepts — `Generative Retrieval`,
  `Transformers`, `BERT for IR`, `PPO`, `GRPO`, and others surfaced during extraction.
- **New** (create per the concept template) — candidate list, finalized after extraction:
  `Recommender System`, `Collaborative Filtering`, `Content-Based Filtering`,
  `Matrix Factorization`, `Implicit vs Explicit Feedback`, `Cold Start Problem`,
  `Sequential Recommendation`, `Session-Based Recommendation`, `GRU4Rec`, `SASRec`, `BERT4Rec`,
  `Diversity (RecSys)`, `Novelty (RecSys)`, `Serendipity`, `Catalog Coverage`,
  `Beyond-Accuracy Evaluation`, `LLM-based Recommendation`, `Generative Recommendation`,
  `Item Tokenization / Semantic IDs`, `RQ-VAE`, `TIGER`, `Large Recommendation Model (LRM)`.

  Naming for ambiguous metrics that already exist for IR (e.g. diversity/novelty): prefer a
  shared note with both `course:` entries when the concept is genuinely the same; only suffix
  with `(RecSys)` when the RecSys meaning is distinct enough to warrant a separate note.

### 5.5 Wiring

- `RecSys - Overview.md` MOC: metadata, lecture table, concept index, links to all notes.
- Add a **RecSys color group** to `.obsidian/graph.json` (alongside RL and IR).
- All new/edited concepts carry correct "Appears In" backlinks.

---

## 6. Execution approach (ultracode, workflow-driven)

**Phase A**
1. Deterministic structural-audit script → JSON/markdown report (dangling links, orphans,
   backlink mismatches, frontmatter gaps, empty folders).
2. LLM semantic-duplicate pass over `Concepts/`.
3. Apply fixes; migrate `CLAUDE.md`; populate `Templates/`; stub Exam Prep; update
   `.obsidian` ignore filter for `CLAUDE.md` + `docs/`.
4. Re-run the audit to confirm zero dangling links / unintended orphans. **Commit.**

**Phase B**
1. Per-lecture deep-extraction agents read *all* slides + images → structured topic + concept
   map per lecture, plus an inventory of reusable vs new concepts.
2. Reconcile the global concept list against existing `Concepts/` (dedup, reuse decisions).
3. Parallel creation: concept notes (one agent per concept) and lecture notes (one per
   lecture); MOC and cross-referencing handled on the main agent.
4. Wire backlinks, MOC, graph color group.
5. Review + completeness-critic pass (frontmatter, link resolution, no skipped figures,
   formula rendering, exam-substitute check). **Commit.**

Commits are incremental and clearly messaged so the diff is reviewable in stages.

## 7. Acceptance criteria

- [ ] `CLAUDE.md` exists at vault root, Claude-Code-native, no OpenClaw tooling references;
      `VAULT-INSTRUCTIONS.md` removed.
- [ ] Audit re-run shows **0 dangling links** and **0 unintended orphans**.
- [ ] No duplicate concept notes remain; merges preserved old names as aliases (no broken links).
- [ ] `Templates/` contains the 6 templates; Exam Prep dangling links resolved.
- [ ] `Courses/RecSys/` has the MOC + 5 lecture notes, all `status: complete`.
- [ ] Every RecSys lecture note reproduces its slides' content and figures; meets the
      exam-substitute bar.
- [ ] RecSys concepts created/reused with correct frontmatter and bidirectional links;
      shared concepts have `RecSys` added to `course:`.
- [ ] Graph has a RecSys color group.

## 8. Risks & mitigations

- **Filename stability** (published site depends on it): never rename a *kept* note; merges
  keep old names as aliases. New RecSys filenames are decided once, here.
- **Lecture 2 is large (101 slides):** budget for a thorough extraction; do not truncate —
  if coverage must be bounded anywhere, log what was dropped.
- **Over-merging concepts:** only merge true duplicates; report borderline cases instead of
  merging. Incremental commits make any over-merge easy to revert.
- **Spec/CLAUDE files appearing in Obsidian:** mitigated by the `.obsidian` ignore filter.

## 9. Out of scope

`../university-vault-site/`; RecSys project/exercises/book notes; re-authoring correct RL/IR
content; folder-taxonomy changes.
