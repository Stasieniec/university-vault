Cards removed on 2026-10-13 from the Flashcards section of Courses/MNLP/Lectures/MNLP-L05 - Static Embeddings.md, pruning the set to what the multiple-choice exam can ask.

## Outside material (reasons or definitions the note labels as not from the lecture)

> [!card]- Why might the unconstrained mapping get worse as the embedding dimension grows?
> Likely overfitting: **$d^2$ free parameters** for the same dictionary (a reason not given by Monz).

> [!card]- What is mixed marking?
> Some constructions mark the **head**, others the **dependent**.

## Duplicate (same fact already tested by a kept recall card and a kept true/false or multiple-selection card)

> [!card]- Which method counts co-occurrences once over the whole corpus and then fits vectors to the log counts?
> **GloVe.**

> [!card]- True or false: fastText improves over Word2Vec on both semantic and syntactic analogies.
> **False.** Gains are syntactic; semantic analogies barely change.

> [!card]- True or false: CSLS penalises target words that are close to many mapped source vectors.
> **True.** That is the $r_S(y)$ term.

> [!card]- True or false: Adversarial unsupervised alignment does not suffer from a domain mismatch between the two monolingual corpora.
> **False.** Across domains it collapses to about zero, even for English-Spanish.
