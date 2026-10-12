# Readable lecture notes

2026-10-12. Stas reads the PhilSci and MNLP notes in place of going to the lecture, and finds them hard
going: they track the slides and readings too closely, explanations follow the source's structure
rather than what would make the idea click, and they carry diagrams and detail he doesn't need. He
often ends up asking an LLM to re-explain them. He is fine with losing some accuracy if the result
is easier to understand. The goal is to pass the exam, so they should not be dumbed down. He wants
to read each note once and then drill the flashcards.

This file covers how the PhilSci notes are written now, what the research says about writing study
text, the length to aim for, and the changes to the notes and to the vault rules. MNLP comes next.

## How the PhilSci notes are written now

Main body only (everything above `## Exam questions`), in words:

| note | body words | references to slides or the deck | ASCII or code blocks | exam/marks mentions |
|---|---|---|---|---|
| L01 Logical empiricism | 5,254 | 22 | 3 | 8 |
| L01b Popper and Lakatos | 10,546 | 0 | 8 | 26 |
| L02 Kuhn | 5,483 | 41 | 7 | 18 |
| L03 Under-determination | 3,746 | 8 | 6 | 21 |
| L03b GenAI and writing | 3,596 | 37 | 2 | 9 |
| L04 Explanation | 6,866 | 15 | 7 | 30 |
| L05 Realism | 5,934 | 16 | 4 | 10 |
| L06 Realism and the PMI | 8,567 | 46 | 7 | 19 |
| **total** | **49,992** | | | |

Brysbaert's 2019 meta-analysis puts average silent reading of English non-fiction at 238 words per
minute. At that rate the PhilSci bodies come to about 3.5 hours, and dense philosophy read for
understanding goes slower than average non-fiction. L01b alone is 45 minutes or more.

Length is not the main problem, though. These are:

**1. The vault's own rule creates the problem.** `CLAUDE.md` section 1 says every note must be "a
full substitute for its source material", and section 2.1 asks for all examples and all images, with
"never write 'see Figure X' and move on". The notes do exactly what the rule says: they reproduce the
source. That gives you coverage of the source rather than understanding of the ideas, and they
aren't the same thing.

**2. Describing the source instead of the idea.** Some of this is plain noise:
- L06: "The slide shows an engraved portrait of a bearded man in profile, credited to Wikimedia,
  evidently Maxwell." Also a full description of a night-sky photo with its photo credit, and "a
  second marker that did not survive the PDF export (it shows as an empty box), presumably a cross".
- L03: a paragraph describing the Escher print on the Poincare slide, plus a #needs-review about
  whether a quotation falls before or after the last paragraph of p. 324.
- "Added to the deck on 2026-09-15", "the deck's (A), (B), (C)", "The slide's points:", "PUA is
  never expanded in the deck."

None of this helps him understand anything. It reads like a log of how the note was put together.

**3. Following the source's order and detail.** L01b walks through the Curd & Cover chapter in
order: the 1979 AAAS letter about parapsychology, where Popper's lecture was first published, the
Trollope epigraph, Popper's footnotes 3 and 4 (the Oedipus effect, Newton and astrology), Lakatos on
how seventeenth-century theology shaped standards of proof, Ampère's book title. Each bit is
accurate and some are fun, but none of it is in the exam key and every one adds load between him and
the five ideas the exam does ask about.

**4. The formal statement comes before the explanation.** L03 opens Duhem with the formal
premises-and-conjunction statement and only then gives the telescope example. Quine's two dogmas sit
in dense table cells. Quine vs Laudan is a two-column table of verbatim quotations. Someone who
already knows the material can read that. A novice has to decode the formal version and only gets
the meaning afterwards.

**5. Exam signposting all the way through.** "Worth marks", "the escape hatch in question 3", "this
one sentence is the difference between a pass and a good mark" appear 8 to 30 times per note. Then
the Key Takeaways section repeats the model answers, and the `## Exam questions` section repeats
them again.

**6. Diagrams that just repeat the prose.** ASCII versions of modus tollens, a box diagram of
"theory A forbids more than theory B", a flowchart of progressive vs degenerating. Each one restates
the paragraph next to it.

What already works and stays: the hooks (the two men and the child, Adler's "thousand-and-one-fold"),
the intuition callouts, the cross-thinker comparison tables, the mapping to mock questions, the
"traps" boxes, and the plain voice the better passages already have.

## What the research says

- **Leave out interesting-but-irrelevant material.** "Seductive details" (interesting material
  that isn't needed for the learning goal) reduce learning: Sundararajan & Adesope 2020, a
  meta-analysis of 58 studies, average g = -0.33, with novices hit harder than people who already
  know the topic. Mayer's coherence principle says the same about extraneous words and pictures,
  consistent across experiments (median effect sizes reported between 0.86 and 1.32 in two editions
  of his handbook chapter [secondary]). This is the strongest reason to cut the anecdotes, the
  publication history and the slide descriptions.
- **Make the links explicit for a reader who is new to the topic.** McNamara, Kintsch, Songer &
  Kintsch 1996: low-knowledge readers learn more from high-coherence text, where the links between
  ideas are spelled out. High-knowledge readers can gain from gaps they fill in themselves. Britton
  & Gülgöz 1991 revised textbook passages to fill in missing connections and recall improved, most
  for readers short on background or motivation [secondary]. So shorter shouldn't mean compressed.
  The cuts come out of side material, and the "because" and "which means" between ideas stay or
  get added.
- **Stories are easier than essays.** Mar et al. 2021, a meta-analysis of 75+ samples and 33,000+
  people: narrative text is understood more easily and remembered better than expository text. One
  later study that held the topic constant found no difference, so take it as a moderate effect.
  Philosophy suits this well, because each lecture is already a chain: someone sees a problem,
  proposes an answer, the answer breaks, the next person fixes it.
- **Novices need the explanation before the formal version.** The worked-example and
  expertise-reversal research (Kirschner, Sweller & Clark 2006; Kalyuga et al. 2003) finds that
  novices need explicit guidance and concrete cases, and that the same guidance becomes redundant
  for experts. So: plain explanation, then an example, then the exact wording the exam key wants.
- **Signal the structure once instead of everywhere.** Mayer's signalling principle: cues that show
  how the material is organised help (median d around 0.4 to 0.6 [secondary]). So a short summary at
  the top and a "what to be able to write" list at the bottom, not "worth marks" in every section.
- **Flashcards are the right second step.** Dunlosky et al. 2013 rate practice testing and spaced
  practice as the two high-utility techniques and rereading as low. Adesope et al. 2017: practice
  tests beat restudy. This matches the plan already in life-ops
  (`40-areas/university/exam-study-method-2026-10.md`): read the note once, then do cards and
  closed-book answers. A note that will be read once has to be understood on that one read, which
  is another reason it has to be easy.

## How long a note should be

Working backwards from how the note gets used:

- The exam: about 8 half-page answers in 2 hours, roughly one question per lecture. A lecture note
  has to support its mock question plus the likely variants, which is around 3 to 6 half-page answers'
  worth of ideas.
- The study plan gives each lecture about an hour on the first pass: a 5-minute cold attempt,
  10 to 15 minutes of flashcards, about 25 minutes writing and checking an answer. That leaves
  **15 to 20 minutes for reading**.
- 15 to 20 minutes at 175 to 240 words per minute is **2,500 to 4,000 words**.

Target: **2,500 to 3,500 words of body per lecture, 4,000 at most.** A lecture with less content
gets a shorter note; nothing gets padded to hit the range. At this size the PhilSci body comes to
roughly 25,000 words in total, half of what's there now.

The pilot (L01b) went from 10,546 to about 4,350 words after review. It is over the 4,000 ceiling: it covers
two readings, two mock questions and two tutorial weeks, the additions came from gaps the reviewers
found, and none of the reviewers found a section that dragged. Treat 4,000 as the normal ceiling and
this as about the most a double lecture should need.

## On sacrificing accuracy

He said he'd trade some accuracy for readability. Most of that trade is leaving things out and
saying things plainly, and it costs nothing at the exam. The one thing not to give up: wherever the
answer key uses a specific term or wording (Lakatos's four components, "predicts novel facts" vs
"only accommodates known facts", Duhem's thesis), the note keeps that wording in one definition
box. The exam is marked against the key, and the person who wrote it does research on exactly these
distinctions.

## How a lecture note should read

1. **In short** (3 to 6 sentences): the question the lecture answers and the answer, in plain words.
   One line saying which mock question it feeds.
2. **The argument as a story.** Problem, then proposal, then example, then what goes wrong, then the
   next thinker. Headings name ideas ("The objection: scientists have thick skins"), not sources or
   slide titles.
3. **Each idea: plain explanation first, then one good example, then the precise wording** in a
   definition box when the exam needs it.
4. **One comparison table** where several thinkers answer the same question.
5. **What to be able to write**: a short list at the end, pointing to the full model answers in
   `## Exam questions`.
6. Then `## Exam questions`, `## Flashcards` and `## Links` as now.

Leave out: slide and deck references, figure descriptions, page numbers in the body (sources go in
Links), publication history and biography unless the argument needs them, footnote detail, diagrams
that repeat the prose, inline uncertainty notes about the source. If something really is unresolved
and matters for the exam, say so once in plain words.

## Changes to `CLAUDE.md`

Proposed, applied once he has read the L01b pilot and is happy with it:

- **Section 1 (Purpose).** Replace "Every note must be a full substitute for its source material"
  with: a lecture note replaces attending the lecture and doing the reading. After one read he
  understands the ideas well enough to answer exam questions on them. Coverage only matters as far
  as it helps that; a detail that helps neither understanding nor an exam answer is left out.
- **Section 2.1 (Completeness).** "All examples" becomes "the best example for each idea, more
  only when the examples are the point". "All images" becomes: reproduce what a figure shows about
  an idea, and only when the text needs it. Never describe the figure itself (layout, portraits,
  photo credits, export glitches).
- **New section: how a lecture note reads.** The structure above, the leave-out list, and the
  length target (2,500 to 3,500 words, 4,000 at most, roughly 15 to 20 minutes).
- **Section 8 (Quality bar).** Drop "all images reproduced or described". Add "reads in about 20
  minutes or less" and "explains each idea before stating it formally".

## Status

- L01b rewritten as the pilot, then reviewed (see below). Its flashcards and exam questions were
  brought in line with the new body.
- Remaining PhilSci notes (L01, L02, L03, L03b, L04, L05, L06): to do once he approves the pilot.
- MNLP: next. Its notes are longer (the L05 to L07 files run 22,000 to 26,000 words each,
  cards included) and the exam is multiple choice with a cheat sheet allowed, so the length target needs
  a separate look.

## How the pilot was reviewed

Six independent reviews on L01b, 2026-10-12:

1. A first-time reader with no philosophy background read the body cold, flagged every confusing
   spot, then wrote the mock answers from the body alone and compared them with the key.
2. An examiner listed everything examinable (mock keys, tutorial questions, slides, other notes) and
   marked each item covered, thin or missing.
3. A philosophy-of-science accuracy check, claim by claim, against the old note and the readings.
4. A line editor for flow, padding, phone rendering and the writing rules.
5. and 6. After the fixes: a fresh cold read and a fresh accuracy check that also verified every
   flashcard and exam answer against the new body.

What they caught that is worth turning into rules for the other notes:

- **Simplifying can change the philosophy.** The first rewrite had Popper condemning Le Verrier's
  invented planet as ad hoc, which he would not (it made a new testable prediction). It also
  presented Lakatos's reading of Kuhn as Kuhn's own view. Plain wording needs an accuracy pass of
  its own.
- **Two verdicts on one case must be kept apart.** Popper calls Einstein's "dear Lord" line
  dogmatic; Lakatos says holding on to Kaufmann-refuted relativity was right. A note that blurs
  them costs marks.
- **Simplified statements must not contradict the later careful version.** "One black swan
  refutes the law" needed the $(T \wedge A)$ step before "falsification is a decision" made sense.
- **Explain every term on first use**, including names (Adler) and words the source treats as
  obvious (sublimation, conventionalist, crucial experiment, heuristic).
- **Tables with long cells are miserable on a phone.** Use short paragraphs, or a table with cells of
  a few words.
- **Tutorial questions count as much as the mock key** when deciding what to keep. The first
  rewrite dropped the passage a tutorial question was built on.
- **When the body changes, the cards and exam answers below it must be re-checked.** Several still
  carried claims the new body had dropped or corrected.
- **Cross-references by section number break** when a note is restructured. Refer to headings.

Each remaining note gets the same review before it is called done.

## Sources

- Brysbaert 2019, reading rate meta-analysis: https://www.gwern.net/doc/psychology/linguistics/2019-brysbaert.pdf
- Sundararajan & Adesope 2020, seductive details: https://news.wsu.edu/press-release/2020/03/19/seductive-details-inhibit-learning
- Mayer, coherence and signalling principles: https://www.cambridge.org/core/books/abs/cambridge-handbook-of-multimedia-learning/principles-for-reducing-extraneous-processing-in-multimedia-learning-coherence-signaling-redundancy-spatial-contiguity-and-temporal-contiguity-principles/C98AB3A6CE760DD63C048936EA0B3B44
- McNamara et al. 1996, coherence and prior knowledge: https://editingresearch.byu.edu/2022/11/23/how-less-clarity-could-improve-active-processing
- Britton & Gülgöz line of work, via Gülgöz et al. 1999 "Designing learnable texts": https://www.jbe-platform.com/content/journals/10.1075/dd.1.2.04gul
- Mar et al. 2021, narrative vs expository: https://pmc.ncbi.nlm.nih.gov/articles/PMC8219577/
- Expertise reversal effect: https://en.wikipedia.org/wiki/Expertise_reversal_effect
- Dunlosky et al. 2013: https://www.psychologicalscience.org/journals/pspi/1529100612453266/
- Adesope et al. 2017, practice testing: https://journals.sagepub.com/doi/10.3102/0034654316689306

[secondary] means the figure came from a summary or abstract, not the paper itself.
