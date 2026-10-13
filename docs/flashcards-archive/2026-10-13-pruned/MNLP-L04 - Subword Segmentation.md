Cards removed on 2026-10-13 from the Flashcards section of Courses/MNLP/Lectures/MNLP-L04 - Subword Segmentation.md, pruned to what the MNLP multiple-choice exam can ask.

## Outside material

Facts the note labels as outside the course material: Sennrich et al.'s reason joint BPE helps, the WordPiece mechanism (only named in this lecture; WordPiece itself is examined through MNLP-L06's cards), and Kudo's paper's pruning rule and single-character exception.

> [!card]- Why, according to Sennrich et al., does joint BPE help translation?
> A name is **segmented identically on both sides**, so the pieces map one to one.

> [!card]- Which segmentation method merges the pair maximising $\frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$?
> **WordPiece**.

> [!card]- What is the difference between BPE's and WordPiece's choice of which pair to merge?
> BPE takes the highest **raw count**; WordPiece divides by the parts' counts, preferring parts that **rarely occur apart**.

> [!card]- How does WordPiece segment a new word at inference?
> By repeatedly taking the **longest vocabulary entry** matching from the left.

> [!card]- Which well-known encoder model uses WordPiece as its tokenizer?
> **BERT**.

> [!card]- Which marker does BPE use during training, and which does WordPiece's vocabulary use?
> BPE: `</w>` for **word end**. WordPiece: `##` for **continuation** (`##ing` attaches left).

> [!card]- Outside MNLP, which tokens do Kudo's paper and SentencePiece's code actually remove each round?
> Those with the **smallest** $L_t$, the reverse of the MNLP algorithm.

> [!card]- In Kudo's paper, why are single-character tokens never pruned?
> So every string stays **segmentable** and pruning cannot create **OOVs**.

> [!card]- True or false: WordPiece merges the most frequent adjacent pair at each step.
> **False.** It divides the pair count by its parts' counts; BPE takes the most frequent pair.

> [!card]- True or false: In Kudo's paper, Unigram LM can prune single-character tokens.
> **False.** Single characters are never pruned, so every string stays segmentable.

> [!card]- Multiple selection: which are true of WordPiece? (A) It builds its vocabulary bottom-up by merging pairs. (B) It merges the pair with the highest raw count. (C) It prefers pairs whose parts rarely occur apart. (D) It prunes a large seed vocabulary.
> **A and C.** B is BPE's criterion. D is Unigram LM.

## Not examinable

A cheat-sheet number (the typical merge range, exact downstream gains) and details specific to the note's own worked BPE trace.

> [!card]- How many merge operations is BPE typically trained with?
> About **30K to 128K** merge operations.

> [!card]- When counting BPE pairs in `the tall man is taller than the tallest man`, does `than` count towards `t+a`?
> **No.** In `than` the `t` is followed by `h`.

> [!card]- Which pair does BPE merge first on `the tall man is taller than the tallest man`, breaking ties by first occurrence?
> **`t+h`**: one of six pairs at count 3, and met first.

> [!card]- True or false: In Bostrom and Durrett's comparison, Unigram LM gained about a point on every English task.
> **False.** About a point on SQuAD and MNLI matched; only a few tenths on MNLI mismatched and CoNLL NER.

## Duplicate

Same fact as a kept card: the ordered merge list (kept: applying merges in order, plus its true/false), BPE's greediness, Unigram LM's morpheme-like splits, the UTF-8 cause of shorter subsegments, and byte-level BPE's zero OOV rate.

> [!card]- What does a trained BPE model consist of?
> An **ordered list of merge operations**.

> [!card]- Why does BPE trained on `the tall man is taller than the tallest man` end with `talle + r` where morphology says `tall + er`?
> The greedy `tall + e` merge was locally most frequent and **buried the boundary**; BPE never undoes a merge.

> [!card]- How does BPE segment `nanotechnology`, compared with Unigram LM?
> Into **generic letter clusters** with no `nano`; Unigram LM gives `nano` + `technology`.

> [!card]- What is the cause of byte-level BPE's shorter subsegments for scripts like Devanagari?
> **UTF-8 encoding length**, nothing about the language itself.

> [!card]- True or false: Byte-level BPE guarantees a zero OOV rate.
> **True.** Its base alphabet is all 256 bytes.
