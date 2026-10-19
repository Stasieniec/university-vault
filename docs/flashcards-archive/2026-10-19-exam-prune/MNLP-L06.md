Cards removed on 2026-10-19 from the Flashcards section of Courses/MNLP/Lectures/MNLP-L06 - Contextual Embeddings.md, when the set was pruned to what the multiple-choice exam on 22 Oct 2026 can ask.

## Not on the slides

> [!card]- What makes BERT "bidirectional"?
> Every position attends to positions on **both** its left and its right.
Reason: not on slides: the slides never explain what "bidirectional" means

> [!card]- What three embeddings are summed to form BERT's input vector at each position?
> **Token**, **segment** (A or B) and **position** embeddings.
Reason: not on slides (token/segment/position embeddings appear only in the note)

> [!card]- Why does BERT not replace every selected token with `[MASK]`?
> `[MASK]` never appears at fine-tuning time; random and unchanged tokens force good representations of **every** input token.
Reason: not on slides: the reason for the 80/10/10 split is the note's commentary

> [!card]- Why can WordPiece merge a rare pair before a frequent one?
> The score is high when the pair occurs together **more often than its parts' frequencies suggest**; pairs containing a very common symbol score low.
Reason: not on slides: the slide gives only the score formula, the explanation is the note's

> [!card]- Why does an exponent α < 1 in $p_l \propto n_l^\alpha$ shrink the gap between languages?
> $x^\alpha$ is **concave** and compresses large values more: the ratio $n_1/n_2$ becomes $(n_1/n_2)^\alpha$.
Reason: not on slides: the concavity argument is the note's own maths

> [!card]- Why does mBERT translation retrieval get worse in the top layers?
> Top layers are shaped by **predicting words in a specific language**, so they become language-specific again.
Reason: not on slides: the reason for the top-layer drop is the note's interpretation; the drop itself is in the true/false card

> [!card]- What distinguishes XLM from XLM-R?
> XLM's full version adds **TLM on parallel sentences**; XLM-R is a later RoBERTa-style mBERT with **no** parallel data.
Reason: not on slides: the slides never describe XLM or TLM

> [!card]- Why does a correlation between wordpiece overlap and transfer not show that overlap causes transfer?
> High-overlap languages are also **typologically closer** to English, so the two are confounded.
Reason: not on slides: the confound argument is the note's commentary

> [!card]- Why would an NLI word-overlap shortcut fail when premise and hypothesis are in different languages?
> Lexical overlap is **near zero by construction**, so the shortcut predicts non-entailment. (An interpretation, not an established result.)
Reason: not on slides: the note itself flags it as an interpretation, not an established result

> [!card]- Why is dynamic masking better than static masking?
> Under static masking unselected tokens are **never targets**; dynamic masking gives **new training signal** each pass at no data cost.
Reason: not on slides: the reason dynamic masking helps is the note's commentary

## Duplicate

> [!card]- Why do embeddings taken from a trained task model's input layer (e.g. a translation system's) still lack context?
> The network saw the whole sentence, but only the **input embedding layer** is kept, so the context computed in deeper layers is **thrown away**.
Reason: duplicate of "What is the shared flaw of static word embeddings, however they are trained?"

> [!card]- At which positions is BERT's MLM loss computed?
> Only at the **selected** positions, whether they show `[MASK]`, a random word or the original word.
Reason: duplicate of the true/false card on where the MLM loss is computed

> [!card]- From which output does BERT predict next sentence prediction?
> **C**, the final-layer output at the **`[CLS]`** position.
Reason: duplicate of "Which BERT special token is never masked..." and the fine-tuning head card

> [!card]- What does BERT output for SQuAD question answering?
> A **start/end span** in the paragraph; nothing is generated.
Reason: duplicate of "Which new parameters does BERT learn when fine-tuned for SQuAD?"

> [!card]- Since mBERT's training objective is not cross-lingual, what do its languages share?
> Only the **parameters** and the **subword vocabulary**.
Reason: duplicate of "Which cross-lingual signals does mBERT's training lack?" and the mBERT training multiple-selection card

> [!card]- Can an mBERT next sentence prediction pair contain two languages?
> **No**: NSP pairs are always within the **same language**.
Reason: duplicate of the mBERT training multiple-selection card, option B

> [!card]- What does language sampling with α = 0.7 do relative to raw data sizes?
> **Up-weights low-resource** and down-weights high-resource languages: a compromise between proportional and uniform.
Reason: duplicate of the language sampling multiple-selection card, option C

> [!card]- How does a language-sampling temperature T relate to α, and what does raising T do?
> $\alpha = 1/T$; raising T makes sampling **flatter**, closer to uniform.
Reason: duplicate of the sampling temperature true/false card; temperature is on the L07 slides, not L06

> [!card]- How does zero-shot NER F1 depend on entity-wordpiece overlap for mBERT versus English BERT?
> mBERT: roughly **flat**, decent even near zero overlap. English BERT: near zero at low overlap, **rising** with overlap.
Reason: duplicate of the true/false card on English BERT's NER F1 against overlap

> [!card]- In which mBERT layers is translation retrieval most accurate?
> The **middle-upper** layers (about 6 to 8 of 12).
Reason: duplicate of the true/false card on translation retrieval peaking in the middle layers

> [!card]- Why does freezing mBERT's lower layers during English fine-tuning help zero-shot transfer?
> English-only fine-tuning would pull them **toward English**; frozen, they keep their **multilingual** pre-trained representation.
Reason: duplicate of the true/false card on freezing lower layers

> [!card]- Why does fake English isolate the effect of subword overlap?
> Words, grammar, **word order** and frequencies stay identical; only the **characters** change, so shared vocabulary is zero.
Reason: duplicate of "What is fake English (K et al. 2020)?"

> [!card]- How much XNLI transfer does B-BERT lose when fake English removes all subword overlap?
> **Very little**: transfer survives almost intact.
Reason: duplicate of the transfer-drivers multiple-selection card, option A

> [!card]- How did K et al. test whether word order matters for cross-lingual transfer?
> **Randomly permuting words during pre-training**, with no permutation during fine-tuning.
Reason: duplicate of the true/false card on permuting words during pre-training

> [!card]- As B-BERT gets deeper at roughly constant size, which improves more: in-language or zero-shot accuracy?
> **Zero-shot** keeps improving; in-language **saturates** early, so the gap shrinks.
Reason: duplicate of the transfer-drivers multiple-selection card, option C

> [!card]- For mBERT on XNLI, which works better: a foreign premise with an English hypothesis, or an English premise with a foreign hypothesis?
> A foreign premise with an **English hypothesis**.
Reason: duplicate of the different-languages multiple-selection card, option D

> [!card]- Is XLM-R better than mBERT within a language, on XNLI and on QA?
> **Yes**, on both, on average.
Reason: duplicate of the XLM-R multiple-selection card, option D

> [!card]- On which task is XLM-R worse than mBERT across languages, on average?
> **Cross-lingual QA**; on XNLI it is better across too.
Reason: duplicate of the different-languages multiple-selection card, option C

> [!card]- What is the bottom line on what drives mBERT's cross-lingual transfer?
> **No single factor**: vocabulary overlap plays a minor role, structure and depth matter.
Reason: duplicate of the transfer-drivers multiple-selection card, option D

## Number or table detail

> [!card]- For mBERT on XNLI, which is better: machine-translated training data (translate-train) or zero-shot transfer?
> **Translate-train** beats zero-shot.
Reason: table comparison with no slide bullet

> [!card]- On zero-shot XNLI, how does mBERT compare with systems pre-trained with cross-lingual signal?
> It narrowly **beats** the older X-LSTM but **loses** to the stronger ones (Artetxe and Schwenk; XLM with MLM+TLM).
Reason: table comparison with named systems, cheat sheet material

> [!card]- On zero-shot MLDoc, how does mBERT compare with the best system trained with parallel text?
> **Nearly matches** it, within a fraction of a point, without seeing translations.
Reason: table comparison with a named system, cheat sheet material

> [!card]- On zero-shot NER, how does mBERT compare with Xie et al.'s dedicated cross-lingual system?
> mBERT is **better** on average.
Reason: table comparison with a named system (Xie et al.)

> [!card]- Does zero-shot mBERT beat Kim et al.'s cross-lingual POS tagger that also uses a few hundred target-language sentences?
> **No**, not on average: adding a little target-language data wins.
Reason: table comparison with a named system (Kim et al.)

> [!card]- In Wu and Dredze's experiment, how does Feat (mBERT not fine-tuned, only the task head trained) compare with fine-tuning?
> **Worse** on average on every task than the best fine-tuned setting: fine-tuning is necessary.
Reason: heatmap row reading; feature extraction against fine-tuning is kept as a concept card

## Obscure

> [!card]- Why does mBERT's zero-shot POS transfer drop for pairs involving Japanese?
> Possibly **larger typological differences** (Japanese is verb-final, with postpositions).
Reason: single-language aside (Japanese)

> [!card]- In Pires et al.'s overlap measure, which wordpieces are compared?
> Those in **labelled named entities** of the fine-tuning and evaluation data (Jaccard similarity).
Reason: analysis-method detail of one paper's overlap measure

> [!card]- From an SOV fine-tuning language, does mBERT's zero-shot POS transfer favour SOV targets over SVO targets?
> **Barely**: the two are almost equal.
Reason: fine-grained cell reading of the word-order table

> [!card]- Which XLM variant beats zero-shot mBERT on XNLI although, like mBERT, it uses no parallel data?
> **XLM with MLM** alone (its model was chosen on target-language dev data).
Reason: table detail about one competing system

> [!card]- In Wu and Dredze's freezing experiment, what does "frozen up to layer 0" mean?
> Only the **embedding layer** is frozen; all Transformer layers are fine-tuned.
Reason: experimental setup detail

> [!card]- Of Spanish, Hindi and Russian, which loses most transfer when B-BERT is pre-trained with permuted words?
> **Hindi**.
Reason: single-language table reading (Hindi)

> [!card]- In K et al.'s mixed-language XNLI test, why does Hindi have the lowest mixed scores?
> Its **monolingual baseline is lowest** too, so the low mixed scores do not show that mixing hurts Hindi most.
Reason: single-language table reading (Hindi)

> [!card]- What structure does XLM-R's cross-lingual QA (context by question language) matrix show?
> "**Own language, or English question**": both work, little in between.
Reason: heatmap pattern reading

> [!card]- Multiple selection: which are true of zero-shot mBERT in Wu and Dredze? (A) Translate-train beats zero-shot on XNLI (B) It beats Kim et al.'s POS tagger, which also uses 320 target-language sentences (C) It beats Xie et al.'s cross-lingual NER system (D) Its XNLI score correlates strongly with wordpiece overlap
> **A and C.** B: Kim et al.'s tagger is better on average. D: XNLI shows no correlation with overlap.
Reason: multiple selection built mostly on table comparisons with named systems (Kim et al., Xie et al.)

## Not examinable

> [!card]- Why does mBERT sample languages with a smoothed distribution?
> Wikipedia sizes are **extremely skewed**: English alone is larger than dozens of small languages combined.
Reason: motivation for sampling; the method itself is tested by the language sampling multiple-selection card

> [!card]- In natural language inference, what relations can a premise have to a hypothesis?
> **Entailment**, **contradiction** or **neutral**: a 3-way classification.
Reason: basic task definition, unlikely as a statement
