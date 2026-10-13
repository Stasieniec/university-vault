# CLAUDE.md — Authoring Rulebook for this Vault

> [!warning] Read this before touching any note in this vault.
> These instructions govern how notes are created, updated, and maintained. Follow them strictly.

---

## 0. Working in this Vault with Claude Code

- **Source materials** (lecture PDFs, textbooks) live **outside** the repo, under
  `/home/stas/Desktop/University/<COURSE>/` — e.g. `RECSYS/`, `RL/`, `IR/`. The git repo
  (`university-vault/`) holds only the notes plus `docs/` (specs, plans, tools).
- **Reading sources:** use the **Read** tool. For PDFs, pass a page range
  (`Read(path, pages="1-20")`, max 20 pages/request) — it renders the pages as images so you
  can read text *and* figures. Read images directly the same way.
- **Audit before you finish:** run `python3 docs/superpowers/tools/vault_audit.py`. It must end
  `RESULT: PASS` (0 dangling links). It also reports orphans and frontmatter gaps. The audit
  understands Obsidian table-cell alias syntax (`[[Target\|display]]`) and skips embeds.
- **A sibling `../university-vault-site/` publishes this vault.** Filenames anchor wikilinks
  *and* published URLs — **never rename a note that is kept.** When merging duplicates, keep the
  survivor's filename and add the loser's name as an `alias` so existing links still resolve.
- **`CLAUDE.md` and `docs/` are hidden** from Obsidian's file explorer via
  `.obsidian/app.json` (`userIgnoreFilters`), so they don't clutter the graph.

---

## 1. Purpose

This Obsidian vault is Stanisław's **single source of truth** for university coursework. The goal: reading these notes alone should be **sufficient to pass every exam, solve every exercise, and understand every concept** — without needing to go back to lecture slides, textbooks, or recordings.

**A lecture note replaces attending the lecture and doing the reading.** After reading it once, Stas understands the ideas well enough to answer exam questions on them. He has no other contact with the subject, so the note explains; it does not transcribe. Coverage serves that goal: a detail that helps neither understanding nor an exam answer is left out, and nothing an exam or tutorial question needs is left out. Nothing is "see the slides for details": what matters is explained here.

He reads each note once, then drills its flashcards, then practises exam answers. A note that will be read once has to be understood on that one read. He is fine with losing nuance for readability, but never with a simplification that changes what a thinker or method actually says. The research behind this, and the review of the first rewrite, are in `docs/superpowers/specs/2026-10-12-readable-lecture-notes.md`.

---

## 2. What Goes Into Notes

### 2.1 What a note must carry

- **Every idea an exam or tutorial question could need**: definitions, arguments, theorems, derivations, methods. Check the mock exam, its answer key and the tutorial questions, not only the slides.
- **The exact wording the answer key uses** for key terms and definitions, kept in a definition box, even where the surrounding prose is plain.
- **Algorithms**: full pseudocode when the course examines the algorithm, not "see Algorithm 5.1"
- **Formulas**: in LaTeX, named, and with intuition for what each term means
- **Examples**: the best example for each idea, worked through. More only when the examples are the point (a thinker's own scorecard, a worked computation the exam repeats).
- **Figures**: reproduce what a figure shows about an idea, and only when the text needs it: a redrawn diagram, a transcribed table, a described trend. Never describe the figure itself (layout, colours, portraits, photo credits, export glitches). A diagram that repeats the prose next to it is cut.
- **Exercises and solutions** (exercise notes): full worked solutions with step-by-step reasoning, not just final answers
- **Intuitions and "why"** — not just "what." If a method has a known weakness, say why. If a formula looks a certain way, explain the intuition behind each term.

### 2.2 What NOT to Include

- Administrative info (deadlines, group policies, grading weights) — that lives in the course overview MOC only
- Redundant re-explanations of the same concept across multiple notes — use `[[wikilinks]]` to the Concept note instead
- Raw copy-paste of source text without formatting — everything must be structured and scannable
- **Talk about the source**: "the slide says", "the deck", "added to the deck on", slide numbers, figure descriptions, notes on what a PDF export lost. The note is about the ideas.
- **Seductive details**: publication history, biography, epigraphs, footnote trivia, anecdotes, unless the argument needs them or an exam or tutorial question touches them. Interesting-but-irrelevant detail measurably lowers learning, most of all for newcomers.
- **Page numbers in the body.** Sources go in `## Links`.
- **Inline uncertainty notes about the source** (#needs-review asides about which paragraph a quote is on). If something genuinely unresolved matters for the exam, say so once in plain words.
- **Exam signposting in the middle of the text** ("worth marks", "this is the escape hatch"). Exam pointers go once at the top and in the closing list.
- **Numbers the exam does not ask you to recall.** For a course with a cheat sheet, exact figures go on the cheat sheet; the note keeps the direction or comparison.

### 2.3 How a lecture note reads

The order follows the ideas, not the slides. Each lecture tells a story: a problem, a proposal, an example, what goes wrong with it, the next proposal. Write it that way.

1. **`> [!abstract] In short`**: 3 to 6 sentences giving the question the lecture answers and the answer, in plain words, plus one line naming the mock exam question(s) it feeds.
2. **Sections named after ideas** ("The objection: scientists have thick skins"), never after sources or slide titles.
3. **For each idea: plain explanation first, then one good example, then the precise wording** in a `> [!definition]` box when the exam needs it. Novices need the explanation before the formal statement.
4. **Explain every term on first use**, including names ("Alfred Adler, a breakaway from Freud...") and words the source treats as obvious. Glosses go in brackets, a few words long.
5. **Spell out the links between ideas** ("because", "which means", "so"). Shorter does not mean compressed: cut side material, never the reasoning between steps.
6. **One comparison table** where several thinkers or methods answer the same questions, with short cells. Long-cell tables are miserable on a phone: use short bold-led paragraphs instead.
7. **`## What to be able to write`** (or `## What to be able to answer` for multiple-choice courses): a short list at the end, pointing to the full model answers in `## Exam questions`.
8. Then `## Exam questions`, `## Flashcards` and `## Links` (section 3.8).

**Accuracy rules for plain writing**, each learned from a real mistake in the first rewrite:
- A simplified statement must not contradict the careful version later in the note. Introduce the simple version so the later correction refines it.
- Keep two thinkers' verdicts on the same case apart, and say whose view each is. Never present one thinker's reading of another as the other's own view.
- Keep necessary conditions necessary and "trivial confirmations" trivial: do not strengthen or weaken a claim while shortening it.
- Additions from outside the course material (historical background, a real-world case) are labelled as such.

**Length.** A lecture note's body (everything above `## Exam questions`) should read in about 15 to 20 minutes: **2,500 to 3,500 words, about 4,000 at most** for an essay-exam course. A lecture covering two readings and two mock questions can run a little over (PhilSci L01b is about 4,400). A thin lecture gets a shorter note; nothing is padded to reach the range. Courses whose exam is multiple choice on specific details may need more room; their target is set in the course's Overview.

**Review.** A rewritten or new lecture note is not `complete` until it has been reviewed by independent subagents: a first-time reader who then answers the mock questions from the body alone, an examiner checking coverage against the keys, tutorial questions and slides, an accuracy check against the sources, and a line editor (flow, padding, phone rendering, the writing rules below). After the fixes, a fresh check of the changed lines, including every flashcard and exam answer against the new body.

**Writing rules** (Stas's own): no em-dashes, no emoji, no "X, not Y" constructions unless the contrast is the information, no slop phrasing, no headings that lead with a count, no invented category labels.

---

## 3. Note Types & Templates

Blank templates for every note type live in `Templates/`.

### 3.1 Course Overview (MOC — Map of Content)

**Location:** `Courses/<COURSE>/<COURSE> - Overview.md`

The central hub for a course. Contains:
- Course metadata (credits, instructor, exam format, textbook)
- Week-by-week schedule with links to all lecture notes, book chapters, exercises
- Assessment breakdown
- Status tracker (which notes are done, which need review)

**Frontmatter:**
```yaml
---
type: moc
course: <COURSE_CODE>
tags: [moc]
---
```

### 3.2 Lecture Notes

**Location:** `Courses/<COURSE>/Lectures/<COURSE>-L<##> - <Title>.md`

Replaces attending the lecture and doing the readings. Self-contained, written to be understood on one read. Structure, length and review: section 2.3.

**Structure:**
1. Frontmatter with metadata
2. `In short` box
3. Content sections following the argument, named after ideas
4. Comparison table where it helps, then `What to be able to write`
5. `Exam questions`, `Flashcards`, `Links`

**Frontmatter:**
```yaml
---
type: lecture
course: <COURSE_CODE>
week: <N>
lecture: <N>
date: <YYYY-MM-DD>  # lecture date if known, otherwise omit
book_sections: ["Ch X.Y", "Ch X.Z"]
topics:
  - "[[Concept A]]"
  - "[[Concept B]]"
status: complete  # complete | draft | stub
---
```

### 3.3 Book Chapter Notes

**Location:** `Courses/<COURSE>/Book Notes/<COURSE>-Book Ch<N> - <Title>.md`

Covers material from the textbook. Focuses on content that **adds to or deepens** what's in the lecture notes. If the book and lecture cover the same thing identically, the lecture note handles it and the book note links there. But if the book has extra proofs, examples, or nuance — it goes here.

**Frontmatter:**
```yaml
---
type: book-chapter
course: <COURSE_CODE>
book: "<Book Title>"
chapter: <N>
sections: ["X.Y", "X.Z"]
topics:
  - "[[Concept A]]"
status: complete
---
```

### 3.4 Concept Notes (The Wiki Layer)

**Location:** `Concepts/<Concept Name>.md`

The **atomic units of knowledge**. Each concept gets one note. These are linked from everywhere — lectures, book notes, exercises, other concepts.

**Rules:**
- **One concept per note** (e.g., "Bellman Equation" not "Bellman Equations and Dynamic Programming")
- **Course-agnostic when possible** — a concept like "Gradient Descent" applies to RL, DL, ML1. Tag which courses use it.
- **Must include:** definition, intuition, math (if applicable), relation to other concepts, where it's used
- **Flat folder** — no nesting by course. Use tags/properties to filter.

**Frontmatter:**
```yaml
---
type: concept
aliases: [<alt name 1>, <alt name 2>]
course: [<COURSE_1>, <COURSE_2>]
tags: [<relevant-tags>]
status: complete
---
```

**Structure:**
1. **Definition** — precise, formal
2. **Intuition** — plain-language explanation, analogies
3. **Mathematical Formulation** — LaTeX, with term-by-term explanation
4. **Key Properties / Variants** — bullet points
5. **Connections** — links to related concepts
6. **Appears In** — which lectures, book chapters, exercises reference this

### 3.5 Exercise & Homework Notes

**Location:** `Courses/<COURSE>/Exercises/<COURSE>-HW<N> - <Title>.md`

**Every exercise and homework problem** must have:
- Full problem statement
- Complete worked solution with all steps shown
- Explanation of which concepts are being tested (with `[[links]]`)
- Common pitfalls or tricky parts highlighted

**Frontmatter:**
```yaml
---
type: exercise
course: <COURSE_CODE>
week: <N>
source: "Exercise Set <N> / Homework <N> / Coding Assignment <N>"
concepts:
  - "[[Concept A]]"
  - "[[Concept B]]"
status: complete
---
```

### 3.6 Coding Assignment Notes

**Location:** `Courses/<COURSE>/Coding Assignments/<COURSE>-CA<N> - <Title>.md`

Captures:
- What the assignment implements and why
- Key algorithms used (linked to concepts)
- Important implementation details / gotchas
- Core code snippets (not the entire notebook — just the important parts)
- Results and interpretation

**Frontmatter:**
```yaml
---
type: coding-assignment
course: <COURSE_CODE>
week: <N>
language: python
concepts:
  - "[[Concept A]]"
status: complete
---
```

### 3.7 Exam Prep Notes

**Location:** `Courses/<COURSE>/Exam Prep/`

- **Exam Analysis** — breakdown of past exams: what topics were tested, question types, weight
- **Cheat Sheet** — single note with every critical formula, named and linked to its concept note. Designed for last-minute cramming.

---

### 3.8 Flashcards and course pages

Lecture notes for exam courses end with two sections, in this order, placed immediately before `## Links` (or at the end if there is no Links heading):

- `## Exam questions`: long-form exam-style questions for written practice. Real mock exam questions are quoted verbatim and follow the official key. The first answer line is `> **Key points:** a; b; c.` (3 to 5 points a grader would tick), then a blank `>` line, then the full model answer.
- `## Flashcards`: short single-fact cards for drilling on a phone in spare moments, about ten seconds each.

```markdown
> [!exam]- Exam-style question, on one line
> **Key points:** first; second; third.
>
> The full model answer.

> [!card]- Recall question, on one line
> The answer.
```

Flashcard rules (rewritten 2026-10-08, after the long-form cards turned out to be undrillable: median answer 69 words, most questions asking several things at once):

- **One thing per card.** The question asks exactly one thing. An idea with three parts is up to three cards.
- **Short answers.** Aim for 15 words or fewer, never more than 30. A list only when the list is the fact, at most 5 items of a word or phrase each.
- **Exam relevance.** Every card must be something an exam question could test or need. No biography, years, page numbers or anecdotes unless the note treats them as the point. For multiple-choice courses, no benchmark numbers on cards: turn a number into a direction or comparison ("which gains more?"), and leave exact figures to the cheat sheet. Keep defining parameters of a method (e.g. the masking rate).
- **Understanding, as well as recall.** At least a third of the cards ask why, how, or what distinguishes two similar things. Some key terms get reverse cards (description in, name out).
- **Exam format.** For multiple-choice courses, each set ends with true/false statement cards (`True or false: ...`, answer starting `**True.**` or `**False.**` plus one reason, false ones built with one word wrong) and a few `Multiple selection: ...` cards in the exam's own format.
- **Standalone wording.** Cards are drilled mixed across the whole course, so a card never says "the lecture", "the slides", "this note", "above" or "below". Name the thinker, paper or method instead.
- **Only what the note says.** No facts from outside the note. Anything the note flags as uncertain is left out or marked as not settled in the answer.
- Every answer line starts with `> `; one blank line between cards; no tables, code blocks or wikilinks inside cards.

The long-form recall cards that existed before 2026-10-08 are archived, unpublished, in `docs/flashcards-archive/`.

`Courses/<COURSE>/index.md` (the course home, which is the course's folder page on the site), `Courses/<COURSE>/<COURSE> - Flashcards.md` and `Courses/<COURSE>/<COURSE> - Exam Questions.md` are **generated** by `python3 docs/superpowers/tools/course_pages.py`. Never edit them by hand: change the notes or the script, then re-run it. The site's `sync.sh` runs it on every publish.

## 4. Formatting Standards

### 4.1 Math
All math in LaTeX. Inline: `$...$`. Display: `$$...$$`.

Always explain what variables mean the first time they appear:
```markdown
$$V(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$$

where:
- $V(s)$ — value of state $s$ under policy $\pi$
- $G_t$ — return (total discounted reward) from time step $t$
- $S_t$ — state at time $t$
```

### 4.2 Callouts
Use Obsidian callouts for visual scanning:

```markdown
> [!definition] Bellman Equation
> Recursive decomposition of the value function...

> [!formula] State-Value Function
> $$V(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')]$$

> [!intuition] Why This Works
> Think of it as asking: "What's the average reward I'll get..."

> [!example] Gridworld
> Consider a 4×4 grid where...

> [!warning] Common Mistake
> Don't confuse $V(s)$ with $Q(s,a)$...

> [!tip] Exam Hint
> This shows up in almost every exam...
```

### 4.3 Algorithms
Full pseudocode in code blocks with line-by-line comments where non-obvious:

````markdown
```pseudo
Algorithm: First-Visit MC Prediction (for estimating V ≈ v_π)
─────────────────────────────────────────────────────────────
Input: policy π to evaluate
Initialize:
  V(s) ∈ ℝ arbitrarily, for all s ∈ S
  Returns(s) ← empty list, for all s ∈ S

Loop forever (for each episode):
  Generate episode following π: S₀,A₀,R₁, ..., S_{T-1},A_{T-1},R_T
  G ← 0
  Loop t = T-1, T-2, ..., 0:
    G ← γG + R_{t+1}
    Unless S_t appears in S₀,...,S_{t-1}:     // first-visit check
      Append G to Returns(S_t)
      V(S_t) ← average(Returns(S_t))
```
````

### 4.4 Diagrams

- **Mermaid** for flowcharts, trees, and concept relationships
- **ASCII art** for simple grids, backup diagrams
- **Described figures** when reproducing complex plots — describe axes, trends, key data points

### 4.5 Links

- **Always wikilink** concepts: `[[Monte Carlo Methods]]` not "Monte Carlo methods"
- **Use aliases** where grammatically needed: `[[Monte Carlo Methods|MC methods]]`
- **Inside table cells**, the alias pipe must be escaped: `[[Monte Carlo Methods\|MC]]`
- **Link exercises to concepts** they test
- **Link lectures to book chapters** they correspond to
- **Back-link from concepts** to where they appear
- **Never leave a dangling link.** If the target does not (yet) exist, either create it, point to
  an existing note (add an alias there if needed), or write plain text instead of a `[[link]]`.

### 4.6 Tags

Use sparingly and consistently:
- `#exam-topic` — high-probability exam content
- `#key-formula` — must-memorize equations
- `#needs-review` — note needs another pass
- `#intuition` — contains a particularly good explanation

---

## 5. Process for Creating Notes

### 5.1 Source Material Priority

When creating notes for a topic, synthesize from **all available sources** in this order:
1. **Lecture slides** (PDF) — primary structure and flow
2. **Textbook chapters** — depth, proofs, additional examples
3. **Exercise sets with answers** — test understanding, reveal what's important
4. **Coding assignments** — practical implementation details

When a course has **no textbook** (e.g. RecSys), the lecture notes carry the full load — there is
no fallback, so read every slide and figure especially carefully. Reading everything is not the same
as writing everything down: the note keeps what section 2.1 asks for.

### 5.2 Handling Images from Source PDFs

**Read every image; reproduce only what carries an idea.** When a lecture slide or book page contains a figure:
1. Use the **Read** tool to view the page/image (`Read(path, pages="N-M")` for PDFs) and extract its content.
2. If the figure carries an idea the note needs, reproduce that content (never a description of the figure itself):
   - **Diagrams/flowcharts** → Mermaid or ASCII art
   - **Plots/graphs** → Describe axes, trends, key values; redraw as ASCII if simple
   - **Tables** → Markdown tables
   - **Backup diagrams** → ASCII art showing the tree structure
   - **Math on slides** → Transcribe to LaTeX
3. If the image is essential and cannot be adequately reproduced in text (e.g., complex plots with many data points), save a copy to `Assets/` and embed it: `![[image-name.png]]`

### 5.3 Workflow Per Topic

For each week/topic:
1. Read the lecture PDF (and the corresponding book chapters)
2. Create/update **Concept notes** for each key concept introduced
3. Create the **Lecture note** — referencing concepts via wikilinks
4. Create the **Book chapter note** — adding depth beyond the lecture
5. Create the **Exercise note** — full solutions with concept links
6. Create the **Coding assignment note** — if one exists for that week
7. Update the **Course Overview MOC** with links and status

---

## 6. Sub-Agent Usage (Claude Code)

Claude Code can parallelise note creation with the **`Agent`** tool (one subagent per task) or the
**`Workflow`** tool (deterministic fan-out across many tasks). Use this for scale; keep
synthesis and cross-referencing on the main agent.

### 6.1 When to Use Sub-Agents

- **Good:** one subagent per lecture PDF → its lecture note, run in parallel across lectures.
- **Good:** one subagent per standalone concept note (different files → no write conflicts).
- **Good:** a read-only extraction pass over a long PDF or exercise set.

### 6.2 When NOT to Use Sub-Agents

- **Concept notes that cross-reference each other** while both are being written — **freeze the
  canonical concept names first** (decide every filename/alias up front), then fan out, so
  wikilinks are consistent across notes. Without frozen names, parallel writers invent
  inconsistent links.
- **The Course Overview MOC** — the main agent owns it, since it links everything together.
- **Quality review passes** — the main agent reviews subagent output for consistency.
- **Small, quick notes** — the spawn overhead isn't worth it.

### 6.3 Sub-Agent Instructions

When dispatching a subagent for note creation, always include:
1. The exact note type and template to follow (point to the file in `Templates/` or paste the section).
2. The source file path(s) to read (PDFs via `Read(path, pages=...)`).
3. The output file path.
4. Instruction to read every image and reproduce only what carries an idea (section 5.2).
5. The **frozen list of concept names** to wikilink to (so links are consistent and resolve).
6. Section 2.3 (how a lecture note reads, length, accuracy rules), and a finished note in the same course as the model to match.

### 6.4 Post-Sub-Agent Checklist

After a subagent delivers a note, the main agent must:
- [ ] Verify frontmatter is correct
- [ ] Check all wikilinks resolve (run the audit script)
- [ ] Ensure no figure that carries an idea was skipped, and no figure is described for its own sake
- [ ] Confirm formulas render correctly
- [ ] Run the review in section 2.3 before marking a lecture note `complete`
- [ ] Add the note to the Course Overview MOC

---

## 7. Vault Maintenance Rules

1. **Never delete content without asking** — mark with `#needs-review` instead.
2. **Keep note filenames stable** — changing filenames breaks wikilinks (and published URLs) everywhere.
3. **Update the MOC** every time a new note is added.
4. **Concept notes are shared** — if editing a concept note, check it doesn't break context for another course.
5. **Status tracking** — every note must have a `status` field: `complete`, `draft`, or `stub`.
6. **No orphan notes** — every note must be linked from at least the Course Overview MOC.
7. **No dangling links** — run `docs/superpowers/tools/vault_audit.py`; it must end `RESULT: PASS`.

---

## 8. Quality Bar

A note is `complete` when:
- [ ] Someone reading **only this note** (plus linked concept notes) could answer exam questions on the topic, checked by a reviewer answering the mock questions from it
- [ ] It reads in about 20 minutes or less (section 2.3), and explains each idea before stating it formally
- [ ] Every term is explained on first use; nothing talks about the slides
- [ ] All formulas are in LaTeX and explained
- [ ] All examined algorithms have full pseudocode
- [ ] All exercises/solutions are worked through with reasoning
- [ ] Wikilinks connect to all relevant concepts
- [ ] The note has correct frontmatter and tags

A note is `draft` when it covers the main content but is missing some of the above.
A note is `stub` when it's a placeholder with minimal content.

---

## 9. File & Folder Conventions

```
university-vault/
├── Courses/
│   ├── <COURSE>/
│   │   ├── <COURSE> - Overview.md
│   │   ├── Lectures/
│   │   ├── Book Notes/
│   │   ├── Exercises/
│   │   ├── Coding Assignments/
│   │   └── Exam Prep/
│   └── Archive/                  ← past/completed courses
├── Concepts/                     ← flat, course-agnostic
├── Templates/                    ← note templates
├── Assets/                       ← images, diagrams
├── docs/                         ← specs, plans, tooling (hidden from Obsidian)
└── CLAUDE.md                     ← this file
```

Naming conventions:
- Lectures: `RL-L01 - Intro, MDPs & Bandits.md`
- Book chapters: `RL-Book Ch5 - Monte Carlo Methods.md`
- Exercises: `RL-ES03 - Exercise Set Week 3.md`
- Homeworks: `RL-HW01 - Homework 1.md`
- Coding assignments: `RL-CA01 - Dynamic Programming.md`
- Concepts: `Monte Carlo Methods.md` (plain name, no course prefix)

Course codes in use: **RL** (Reinforcement Learning), **IR** (Information Retrieval 1),
**RS** (Recommender Systems).
