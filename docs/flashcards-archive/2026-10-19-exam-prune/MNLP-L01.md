Cards removed on 2026-10-19 from the Flashcards section of Courses/MNLP/Lectures/MNLP-L01 - Overview.md, when the set was pruned to what the exam (multiple choice and multiple selection) can ask.

> [!card]- Which four aspects of human language does NLP aim to model algorithmically and formally?
> Word formation, sentence structure (grammar), sentence meaning, document or discourse structure.
Reason: obscure list; the definition of NLP is kept in the three-fields card

> [!card]- How does document retrieval treat a well-formed question?
> It chops it into **keywords** (after stop-word removal).
Reason: side detail of document retrieval; QA against IR is kept in the multiple-selection card

> [!card]- How does question answering differ from document retrieval?
> QA takes a **well-formed question** and returns the **actual answer**, not ranked documents.
Reason: duplicate of the multiple-selection card, option A (QA returns a ranked list of documents)

> [!card]- Why does sentiment analysis require named entity recognition?
> Fine-grained sentiment ("X likes Y about Z") has to identify who X is and what Z is.
Reason: duplicate of the multiple-selection card, option B (sentiment analysis requires NER); the "X likes Y about Z" reasoning is note commentary

> [!card]- At which levels does sentiment analysis operate?
> **Sentence** and **document** level.
Reason: duplicate of "True or false: Sentiment analysis operates only at the document level"

> [!card]- What unit does machine translation typically translate?
> **Individual sentences**.
Reason: duplicate of the multiple-selection card, option C (MT translates individual sentences)

> [!card]- What are the disadvantages of deep learning for NLP?
> Needs **large amounts of training data**, errors are **hard to trace**, can **fail spectacularly**.
Reason: duplicate of the multiple-selection card on deep learning (options B and C)

> [!card]- Why do deep-learning insights transfer between NLP, vision, speech and other fields?
> Network types are **uniform** and there are few **application-specific features**.
Reason: duplicate of the multiple-selection card on deep learning (option D)

> [!card]- How do dedicated multilingual models such as Aya-Expanse compare with large general LLMs?
> They tend to **lag behind** (read as behind the general models; not spelled out).
Reason: duplicate of "True or false: Dedicated multilingual models such as Aya-Expanse outperform general LLMs"

> [!card]- What is the guiding question of crosslinguality?
> **How does knowledge transfer across languages?**
Reason: duplicate of "What is the difference between multilingual and crosslingual NLP?"

> [!card]- Is sentence or document classification across languages a multilingual or crosslingual scenario?
> **Multilingual**: one classifier trained without being language-specific.
Reason: duplicate of the multiple-selection card on crosslingual scenarios (option D)

> [!card]- Which crosslingual scenario asks how to manipulate information across languages?
> **Crosslingual unlearning**.
Reason: duplicate of "True or false: Crosslingual unlearning asks how to combine information across languages"

> [!card]- Which crosslingual scenario asks how to combine information across languages?
> **Crosslingual reasoning**.
Reason: duplicate of "True or false: Crosslingual unlearning asks how to combine information across languages" (its answer names reasoning)

> [!card]- Which end-user tasks does NLP address?
> Text categorization, summarization, machine translation, question answering.
Reason: obscure list of example tasks from the opening slides

> [!card]- Why does a five-star review still need sentence-level sentiment analysis?
> It can be uninformative, or hide a real complaint in one sentence.
Reason: not on slides (note commentary on a review screenshot)

> [!card]- What made pre-deep-learning NLP costly to extend to a new task?
> Each task needed its own approach and hand-built features.
Reason: duplicate of "True or false: In pre-deep-learning NLP, each application required a different approach"

> [!card]- What distinguishes how pre-deep-learning NLP and neural networks get their features?
> Before: people hand-built features and ML weighed them. Neural networks learn their own.
Reason: duplicate of the multiple-selection card on deep learning (option A)

> [!card]- How is a task like parsing made multilingual?
> By finding language-independent, or universal, features.
Reason: duplicate of the multiple-selection card on crosslingual scenarios (option B)

> [!card]- Why did Mistral Small 3.2's translation in the four-LLM comparison show that fluency is no evidence of correctness?
> It was fluent English describing a different event from the source.
Reason: obscure one-off example; the identification of the language is note commentary

> [!card]- True or false: Deep-learning NLP still requires hand-engineered features such as POS tags and parse trees.
> **False.** Deep learning needs very little or no feature engineering; those features belong to pre-deep-learning NLP.
Reason: duplicate of the multiple-selection card on deep learning (option A)

> [!card]- True or false: Neural networks require large amounts of training data.
> **True.** Data hunger is one of their disadvantages, with hard-to-trace errors and spectacular failures.
Reason: duplicate of the multiple-selection card on deep learning (option B)

> [!card]- True or false: Language-independent parsing is a crosslingual scenario.
> **False.** It is multilingual: it looks for universal features of sentence structure.
Reason: duplicate of the multiple-selection card on crosslingual scenarios (option B)

> [!card]- True or false: Sentence or document classification across languages is a multilingual scenario.
> **True.** Its question is how to train one classifier without being language-specific; "across" does not make it crosslingual.
Reason: duplicate of the multiple-selection card on crosslingual scenarios (option D)
