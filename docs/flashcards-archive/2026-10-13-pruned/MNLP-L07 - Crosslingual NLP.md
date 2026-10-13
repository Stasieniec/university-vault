Cards removed on 2026-10-13 from the Flashcards section of Courses/MNLP/Lectures/MNLP-L07 - Crosslingual NLP.md, when the set was pruned to what the exam can ask.

## Outside the course material

> [!card]- What is catastrophic forgetting in fine-tuning?
> Training only on new data **overwrites** what the model knew, so quality on the original domain drops.

> [!card]- How does the sampling temperature $T$ relate to the exponent $\alpha$ in $p_l \propto n_l^{\alpha}$?
> $\alpha = 1/T$.

## Duplicate

> [!card]- True or false: In BART's text infilling, every removed token is replaced by its own `[MASK]`.
> **False.** A whole span is replaced by a single `[MASK]`; that is what separates it from token masking.

> [!card]- True or false: In parent-child NMT transfer, the source embeddings should stay frozen.
> **False.** The **target** embeddings stay frozen; the source embeddings must be retrained.

> [!card]- True or false: Fine-tuning a general-domain NMT model on a small in-domain set continuously improves with the number of in-domain updates.
> **False.** Gains stop and then reverse as the model overfits the small data.

> [!card]- True or false: In Johnson et al.'s multilingual NMT, the tag added to each example names the source language.
> **False.** It names the **target** language; the source is never named.

> [!card]- True or false: A sampling temperature of T = 100 approximates uniform sampling, which favours low-resource languages.
> **True.** $1/T$ is near 0, so all languages are sampled about equally.

> [!card]- True or false: On average over the 15 XNLI languages, zero-shot XLM (MLM+TLM) beats translate-test.
> **True.** Slightly, although it never saw non-English NLI data.

> [!card]- True or false: mBART is fine-tuned for translation as a whole, with no new source encoder.
> **True.** It saw every language in pretraining; English-only BART needs the new encoder.
