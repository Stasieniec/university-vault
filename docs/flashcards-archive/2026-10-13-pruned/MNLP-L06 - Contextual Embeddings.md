Cards removed on 2026-10-13 from the Flashcards section of Courses/MNLP/Lectures/MNLP-L06 - Contextual Embeddings.md, when the set was pruned to what the exam can ask.

## Outside material

> [!card]- Which training signal, absent from mBERT, could improve its cross-lingual transfer?
> One candidate: parallel data with an **alignment loss** pulling translation equivalents together. The question is left open.

## Duplicate

> [!card]- Which segmentation method merges the pair maximising count(a,b) / (count(a) · count(b))?
> **WordPiece**.

> [!card]- Which segmentation method builds XLM-R's 250k vocabulary?
> **Unigram LM** via SentencePiece.

> [!card]- True or false: Language sampling with α = 0.7 down-weights high-resource languages.
> **True.** It up-weights low-resource and down-weights high-resource languages.

> [!card]- True or false: Removing all subword overlap between the two languages (fake English) destroys most of B-BERT's cross-lingual transfer.
> **False.** Transfer survives almost intact.

> [!card]- True or false: In Wu and Dredze, mBERT's zero-shot XNLI score shows no type-level correlation with wordpiece overlap with English.
> **True.** Sentence-level inference does not depend on shared wordpieces.

> [!card]- True or false: Deeper B-BERT models at roughly constant parameter count narrow the gap between in-language and zero-shot accuracy.
> **True.** Zero-shot keeps improving with depth while in-language saturates.

> [!card]- True or false: XLM-R outperforms mBERT across languages on every task.
> **False.** It is better across on XNLI but, on average, worse across on cross-lingual QA.
