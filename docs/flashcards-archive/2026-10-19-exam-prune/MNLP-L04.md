Cards removed on 2026-10-19 from the Flashcards section of Courses/MNLP/Lectures/MNLP-L04 - Subword Segmentation.md, when the set was pruned to what the multiple-choice exam on 22 Oct 2026 can ask: on the slides, decidable as a statement, not a number, not a duplicate, not obscure.

## Duplicate

> [!card]- What happens to an unseen word under a word-level NMT vocabulary?
> It collapses to **`<UNK>`**, the unknown-word symbol.
Reason: duplicate of the L03 short-list cards (the L04 slides do not mention <UNK>)

> [!card]- Why are hand-built FST analyzers ruled out for multilingual NLP even though they are accurate?
> They are **totally language dependent**: one expert-built analyzer per language does not scale.
Reason: duplicate of "What are the three weaknesses of hand-built FST morphological analyzers?"

> [!card]- How do BPE and Unigram LM differ in the direction of vocabulary construction?
> BPE starts from characters and **merges upward**; Unigram LM starts large and **prunes downward** to size $k$.
Reason: duplicate of "True or false: Unigram LM builds its vocabulary bottom-up..." and "Multiple selection: which are true of BPE?" option D

> [!card]- What is BPE's only hyperparameter?
> The **number of merge operations**, which sets the final vocabulary size.
Reason: duplicate of "Multiple selection: which are true of BPE?" option C

> [!card]- How is a trained BPE model applied to a new word?
> Split it into characters and apply the merges **in the order they were learned**.
Reason: duplicate of "True or false: Applying a trained BPE model replays the learned merges in the order they were learned."

> [!card]- What are BPE's two structural limitations?
> **Greedy** (a merge is never undone) and **deterministic** (one segmentation per word).
Reason: duplicate of "Multiple selection: which are true of BPE?" options A and B

> [!card]- Why can BPE not sample alternative segmentations of a word?
> It is **deterministic**: one segmentation per word, so there is **no distribution** to sample from.
Reason: duplicate of "Multiple selection: which are true of BPE?" option B

> [!card]- Is SentencePiece a segmentation algorithm?
> **No.** It is Google's **toolkit**, implementing both BPE and Unigram LM.
Reason: duplicate of "Multiple selection: which are true of SentencePiece?" option A

> [!card]- How does Unigram LM choose the segmentation of a string $X$?
> The $\arg\max$ of $P(\mathbf{x})$ over **all segmentations** $S(X)$: a global choice.
Reason: duplicate of "What are the four properties of Unigram LM (Kudo)?"

> [!card]- Which method starts from all substrings occurring more than once and prunes them to the target size?
> **Unigram LM**.
Reason: duplicate of "True or false: Unigram LM builds its vocabulary bottom-up, merging pairs until it reaches the desired size."

> [!card]- Which tokens does the Unigram LM algorithm as given in MNLP remove each round?
> Those with the **highest** $L_t$: the ones whose removal loses most likelihood.
Reason: duplicate of "True or false: In Unigram LM vocabulary construction, the tokens with the highest loss $L_t$ are removed each round."

> [!card]- Why can dynamic programming find the best Unigram LM segmentation?
> Segments are **independent**, so the best cut of the left part does not depend on the right part.
Reason: duplicate of "True or false: DP segmentation under a unigram model is valid because segments are treated as independent."

> [!card]- How do top-down and bottom-up dynamic programming differ?
> Top-down is **recursion with memoization**; bottom-up **fills a table** small to large. Same asymptotic run time.
Reason: duplicate of "True or false: Bottom-up DP has a better asymptotic run time than top-down DP with memoization."

> [!card]- With what value is each DP cell $m[i,j]$ initialised before trying split points?
> $p(c[i..j])$, the **no-split** option; zero if the span is not in the vocabulary.
Reason: duplicate of "True or false: In DP segmentation, each cell is initialised with the probability of the unsplit span."

> [!card]- How does SentencePiece treat its input?
> As a **raw stream of Unicode characters including spaces**, with the space escaped as `▁`.
Reason: duplicate of "True or false: SentencePiece treats its input as a raw stream of Unicode characters, including spaces."

> [!card]- What two consequences follow from SentencePiece's raw-stream input?
> **No language-specific pre-tokenizer**, and **lossless** desegmentation.
Reason: duplicate of "Multiple selection: which are true of SentencePiece?" options C and D

> [!card]- How is SentencePiece output desegmented?
> **Join the pieces** and turn `▁` back into a space: no rules, exactly invertible.
Reason: duplicate of "Multiple selection: which are true of SentencePiece?" option D

> [!card]- At the same vocabulary size, which produces longer segments on average: BPE or Unigram LM?
> **Unigram LM**, in both English and Japanese.
Reason: duplicate of "True or false: At the same vocabulary size, Unigram LM produces longer segments on average than BPE."

> [!card]- Where is Unigram LM's downstream advantage over BPE much larger: English tasks or Japanese TyDi QA?
> **Japanese TyDi QA**; on English tasks it is small.
Reason: duplicate of "Multiple selection: which are true of Bostrom and Durrett's BPE against Unigram LM comparison?" option B

> [!card]- How does byte-level BPE guarantee a zero OOV rate?
> It runs on **UTF-8 bytes**; the base alphabet is fixed at **256 symbols**.
Reason: duplicate of "Multiple selection: which are true of byte-level BPE?" option A

> [!card]- Which method starts from a fixed base alphabet of 256 symbols?
> **Byte-level BPE**.
Reason: duplicate of "Multiple selection: which are true of byte-level BPE?" option A

> [!card]- Why does byte-level BPE give non-Latin scripts shorter subsegments?
> They take **more bytes per character**, so merges are spent getting **back to characters**.
Reason: duplicate of "True or false: Byte-level BPE gives non-Latin scripts shorter subsegments because those languages are morphologically richer."

> [!card]- What is the formula for fertility?
> $\text{fertility} = \frac{\text{number of subword tokens}}{\text{number of words}}$; 1 means one token per word.
Reason: duplicate of "Multiple selection: which are true of fertility?" option A

> [!card]- Why do agglutinative languages like Finnish or Turkish have high fertility?
> They have many more distinct, **rarer word forms**, so each gets **split further**.
Reason: duplicate of "Multiple selection: which are true of fertility?" option D

> [!card]- Why do high-fertility languages pay more on commercial LLM APIs?
> APIs **price by token**, and the same content takes **more tokens**.
Reason: duplicate of "What three costs does high fertility bring a language?"

## Not on the slides

> [!card]- Why is a character-level vocabulary a poor fix for OOV words, even though its OOV rate is near zero?
> Each symbol carries almost **no meaning**, and sequences become **long and slow**.
Reason: not on slides: the character-vocabulary argument is the note's own

> [!card]- Why does Minimum Description Length push segmentation towards morpheme-like pieces?
> Whole words make the model huge, characters make the data expensive; the optimum is **reusable pieces**.
Reason: not on slides: the note's own explanation of MDL

> [!card]- Why is BPE training cheap?
> It runs over the **word types with their frequencies**, never over the whole corpus.
Reason: not on slides: training cost is the note's own commentary

> [!card]- What does BPE's end-of-word marker `</w>` let it distinguish?
> **Word-final** pieces from word-internal ones, so pieces can be turned back into words.
Reason: not on slides: the slides show </w> but never state its purpose

> [!card]- Why is BPE's greediness a problem for segmentation quality?
> A **locally best** merge can block a **better** segmentation, and it is never undone.
Reason: not on slides: the note's own commentary on greediness

> [!card]- In Kudo's paper, how is the unigram LM fitted during vocabulary construction?
> By **EM**, with each string's segmentation as a **hidden variable**.
Reason: paper only: EM is in Kudo's paper, not on the slides

> [!card]- Why does Unigram LM prune at most a fraction $\alpha$ of the vocabulary per round?
> Each $L_t$ assumes all other tokens stay, so the model is **refitted** between rounds.
Reason: not on slides: the reason for gradual pruning is the note's own

> [!card]- Why is the DP segmentation table filled by increasing span length?
> Both halves of any split are **shorter**, so their best scores are already final.
Reason: not on slides: the note's own explanation of fill order

> [!card]- What is the complexity of span-table DP segmentation in word length $n$?
> $O(n^3)$ **time** and $O(n^2)$ **space**.
Reason: not on slides: complexity of DP segmentation is not given

> [!card]- Why do implementations of DP segmentation add log probabilities instead of multiplying probabilities?
> Long products **underflow** to zero.
Reason: not on slides: implementation detail

> [!card]- In DP segmentation of `cats`, why can `cat | s` win even though `cats` is in the vocabulary?
> $p(\text{cat}) \cdot p(\text{s})$ exceeds $p(\text{cats})$: a **rare whole word** loses to two **common pieces**.
Reason: not on slides: the cats example is the note's own

> [!card]- What does Kudo's subword regularization do during training?
> **Samples** a different segmentation each time a word is seen, instead of always the best one.
Reason: paper only: subword regularization is not on the slides

> [!card]- How did Bostrom and Durrett make their BPE against Unigram LM comparison controlled?
> **Same data, same vocabulary size**, identical pretraining and model; only the tokenizer differs.
Reason: paper only: the experimental controls are not on the slides

> [!card]- Why are both tokenizers' recall scores against CELEX2 English morphology low?
> Probably because frequent words **stay whole** (`walked` misses `walk|ed`).
Reason: paper only: the explanation for low recall is not on the slides

> [!card]- Does a larger gold-standard F1 gap between tokenizers predict a larger downstream gap?
> **No.** The F1 gap is larger in English, the downstream gap larger in Japanese.
Reason: not on slides: the note's own inference from two tables

> [!card]- Why do unknown characters cause OOVs for character-based subword vocabularies?
> The base vocabulary holds only **characters seen in training**.
Reason: not on slides as a separate point; covered by "What two things can still cause OOVs under BPE or Unigram LM?"

> [!card]- True or false: The `</w>` end-of-word marker is what prevents BPE from merging pairs across two words.
> **False.** Words are counted separately, so no pair spans two words; `</w>` marks word-final pieces.
Reason: not on slides: the slides never say what </w> prevents

> [!card]- Multiple selection: which are true of Unigram LM? (A) It scores a segmentation as a product of subword probabilities. (B) It is deterministic, like BPE. (C) It fits its model by EM. (D) It merges the most frequent adjacent pair.
> **A and C.** B: it supports n-best and sampled segmentations. D: that is BPE.
Reason: option C (EM) is paper only; A and D are covered by other cards (product scoring, BPE merging), B by the single-segmentation true/false card

## Number or table detail

> [!card]- Why do multilingual models sit at the top of BPE's usual merge range?
> A vocabulary shared by **many scripts** needs more room.
Reason: merge range is a number for the cheat sheet; the reason is not on the slides

> [!card]- In Sennrich et al.'s English to German NMT, how did 300K/500K word-level systems compare with 60K to 90K subword systems?
> The best **neural** scores came from **subword** systems, despite their much **smaller vocabularies**.
Reason: table values (vocabulary sizes, BLEU), cheat sheet; the direction is kept in "What three major advantages does BPE bring to NMT?"

> [!card]- In Sennrich et al.'s English to German experiments, which was better: joint or separate BPE?
> **Joint BPE**, one set of merges learned on both languages together.
Reason: one row of the Sennrich table, obscure

> [!card]- In Sennrich et al.'s experiments, did any single neural model beat the syntax-based statistical system?
> **No.** Only ensembles of 8 subword models beat it.
Reason: one row of the Sennrich table, obscure

> [!card]- Which extra tokens are characteristic of BPE versus Unigram LM in English?
> BPE: **word-initial single capitals**; Unigram LM: **suffixes** like `s`, `ed`, `ing`, `ly`.
Reason: table detail (which tokens each over-produces)

## Obscure or side detail

> [!card]- What does `@@` mark in BPE output such as `K@@ yrg@@ yz@@ stan`?
> A piece that **continues** into the next; only a display convention, the merges use `</w>`.
Reason: display convention in one example; the slides never explain @@

> [!card]- When cutting a sequence into scored pieces, why must complete cuttings be compared?
> Neither extreme (whole, or all single units) need win; the optimum can be an **interior split**.
Reason: lesson drawn from the abstract sequence-cutting example by the note, not stated on the slides

> [!card]- What goes wrong when BPE segments numbers such as years?
> It splits them like `184` + `8`, a **meaningless prefix** plus a digit.
Reason: one example (years) from the comparison figure

> [!card]- What does BPE do wrong when segmenting Japanese, compared with Unigram LM?
> It **glues particles** (は, に) onto content words.
Reason: one Japanese example sentence from the comparison figure
