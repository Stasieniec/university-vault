Cards removed on 2026-10-19 from the Flashcards section of Courses/MNLP/Lectures/MNLP-L07 - Crosslingual NLP.md, when the set was pruned to what the multiple-choice exam can ask.

## Duplicate

> [!card]- Which of the four senses in which mBERT and XLM-R are "multilingual" leaves languages unlinked?
> Their predictions are based on context **within the same language** (MLM).
Reason: duplicate of "Why does training on many languages not by itself link them?"

> [!card]- What is crosslingual transfer called when no task data in the test language was seen at all?
> **Zero-shot** crosslingual transfer.
Reason: duplicate of "What is crosslingual knowledge transfer?"

> [!card]- How does the LASER encoder turn a sentence into one vector?
> BPE embeddings, a **stack of BiLSTMs**, then **max pooling** over time into a fixed-size sentence embedding.
Reason: duplicate of "True or false: LASER's sentence embedding is obtained by max pooling over the BiLSTM states."

> [!card]- Why are LASER sentence embeddings language-independent?
> The decoder sees the source only through the sentence vector and gets the output language from a **language ID embedding**, so the encoder need encode only **meaning**.
Reason: duplicate of "True or false: LASER's decoder is told the output language by a language ID embedding, not by the encoder."

> [!card]- Which XLM objective masks words in a sentence concatenated with its translation?
> **TLM**, translation language modelling.
Reason: duplicate of "What are XLM's three pretraining objectives?" and "Why does TLM force crosslingual alignment?"

> [!card]- How does TLM differ from MLM?
> The loss is the **same**; only the **input** differs: a translation pair instead of monolingual text.
Reason: duplicate of "True or false: TLM uses a different loss function from MLM."

> [!card]- What does InfoXLM's XLCo objective classify?
> Whether a sentence pair is **parallel**, from the two `[CLS]` representations, with **random sentences** as negatives.
Reason: duplicate of "Multiple selection: which are true of XLM and InfoXLM pretraining? (A) XLM keeps BERT's next sentence prediction. (B) XLM's TLM is always combined with MLM or CLM. (C) InfoXLM's XLCo classifies whether a sentence pair is parallel from `[CLS]` representations. (D) InfoXLM is trained from scratch.", option C

> [!card]- What is paradoxical about XLM-R against mBERT in mixed-language evaluation?
> XLM-R is much better **within** languages but **worse across** languages on QA, on average.
Reason: duplicate of "True or false: XLM-R is better than mBERT across languages on mixed-language QA."

> [!card]- How does InfoXLM compare with XLM-R on within and across scores?
> Only **slightly better within**, **much better across**: the within-across gap shrinks sharply.
Reason: duplicate of "True or false: InfoXLM is only slightly better than XLM-R within a language but much better across languages."

> [!card]- How did Sutskever et al. (2014) connect the LSTM encoder and decoder?
> By **initialising the decoder LSTM with the encoder's last state**.
Reason: duplicate of "What problem does $\mathbf{h}^{dec}_0 = \mathbf{h}^{enc}_n$ cause in an RNN encoder-decoder?"

> [!card]- Which Transformer decoder component replaces the RNN encoder-decoder bottleneck?
> The **source-target context layer** (cross-attention) over **all** top-layer encoder outputs, in every decoder layer.
Reason: duplicate of "True or false: In the Transformer encoder-decoder, every decoder layer can attend to all encoder output positions."

> [!card]- In Zoph et al.'s freezing experiment, which parameters should stay frozen in the child?
> Only the **target-language embeddings**. Training the decoder RNN and attention still helps.
Reason: duplicate of "Multiple selection: which are true of parent-child transfer learning for low-resource NMT? (A) The child model is initialised with the parent's parameters. (B) The target-language embeddings stay frozen during child training. (C) The source embeddings stay frozen during child training. (D) Transfer from a smaller parent to a larger child works as well as the reverse.", option B

> [!card]- With a shared parent-child vocabulary, do unrelated parents help NMT transfer (Kocmi and Bojar, 2018)?
> **Yes**, about as much as related ones. Having a large parent matters most.
Reason: duplicate of "True or false: With a shared vocabulary, an unrelated parent can help an English→Estonian NMT child about as much as a related Finnish parent."

> [!card]- According to Aji et al. (2020), which part of a parent NMT model carries most of the transfer benefit?
> The **inner** (non-embedding) layers.
Reason: duplicate of "True or false: Transferring only a parent's embeddings, without its inner layers, beats training the NMT child from scratch."

> [!card]- According to Aji et al. (2020), how do parent embeddings transferred alone compare with training the child from scratch?
> **Worse** on average: without the parent's inner layers they are a bad initialisation.
Reason: duplicate of "True or false: Transferring only a parent's embeddings, without its inner layers, beats training the NMT child from scratch."

> [!card]- Which multilingual NMT language combination is used for testing only?
> **Many-to-many**. Many-to-one and one-to-many are used in training.
Reason: duplicate of "True or false: In multilingual NMT, many-to-many translation is used for testing only."

> [!card]- What does a tag like `<2es>` name in Johnson et al.'s multilingual NMT?
> The **target** language. Nothing names the source language.
Reason: duplicate of "Multiple selection: which are true of multilingual NMT as in Johnson et al. (2017)? (A) Each language gets its own encoder. (B) A tag naming the target language is added to the source sentence. (C) Source, target and output embeddings are shared. (D) Training uses every language pair, including pairs without English.", option B

> [!card]- Do low-resource languages gain more from multilingual NMT when translating into English or out of it?
> **Into English** (many-to-one).
Reason: duplicate of "Multiple selection: which are true of Arivazhagan et al.'s (2019) 103-language NMT results? (A) Low-resource languages gain most when translating into English. (B) Multilingual models beat bilingual baselines on high-resource languages. (C) All→All is weaker than the dedicated Any→En and En→Any models. (D) One-to-many gains more than many-to-one for low-resource languages.", option A

> [!card]- What does training on more languages (10 against 102) do to zero-shot translation?
> Improves **most** directions, but not all.
Reason: duplicate of "True or false: Adding more languages to a multilingual NMT model always improves zero-shot translation."

> [!card]- What is BART's architecture?
> A **bidirectional encoder** (as BERT) plus an **autoregressive decoder** (as GPT).
Reason: duplicate of "True or false: BART's decoder is bidirectional."

> [!card]- What is the difference between BART's token masking and token deletion?
> Masking **marks** the gap with `[MASK]`; deletion leaves no mark, so the model must find **where** tokens are missing.
Reason: duplicate of "Multiple selection: which are true of BART's noise functions? (A) Text infilling replaces a span with a single `[MASK]`. (B) Token deletion marks each deleted position with `[MASK]`. (C) Sentence permutation alone performs poorly. (D) Text infilling is the best single noise function.", option B

> [!card]- Which single BART noise function works best?
> **Text infilling**.
Reason: duplicate of "Multiple selection: which are true of BART's noise functions? (A) Text infilling replaces a span with a single `[MASK]`. (B) Token deletion marks each deleted position with `[MASK]`. (C) Sentence permutation alone performs poorly. (D) Text infilling is the best single noise function.", option D

> [!card]- How do sentence permutation and document rotation perform as BART's only noise function?
> **Poorly**: rearranging whole sentences gives too weak a token-level learning signal.
Reason: duplicate of "Multiple selection: which are true of BART's noise functions? (A) Text infilling replaces a span with a single `[MASK]`. (B) Token deletion marks each deleted position with `[MASK]`. (C) Sentence permutation alone performs poorly. (D) Text infilling is the best single noise function.", option C

> [!card]- What is the difference between translating with BART and with mBART?
> BART (English-only) needs a **new source encoder**; mBART is **fine-tuned as is** on parallel data.
Reason: duplicate of "Multiple selection: which are true of mBART? (A) It is pretrained on parallel data. (B) Its pretraining objective has no crosslingual signal. (C) A language ID token starts the decoder input. (D) Translation fine-tuning requires a new randomly initialised source encoder.", option D

> [!card]- In Monz's description of InfoXLM's MMLM, what is used instead of directly optimising the ground-truth loss?
> **Negative sampling.**
Reason: duplicate of "True or false: InfoXLM's MMLM uses negative sampling."

> [!card]- What model is InfoXLM initialised from?
> **XLM-R**, then pretrained further with MMLM, TLM and XLCo.
Reason: duplicate of "Multiple selection: which are true of XLM and InfoXLM pretraining? (A) XLM keeps BERT's next sentence prediction. (B) XLM's TLM is always combined with MLM or CLM. (C) InfoXLM's XLCo classifies whether a sentence pair is parallel from `[CLS]` representations. (D) InfoXLM is trained from scratch.", option D

> [!card]- In which direction of corpus size must NMT transfer flow to pay off?
> From the **larger** corpus (parent) to the **smaller** (child).
Reason: duplicate of "Multiple selection: which are true of parent-child transfer learning for low-resource NMT? (A) The child model is initialised with the parent's parameters. (B) The target-language embeddings stay frozen during child training. (C) The source embeddings stay frozen during child training. (D) Transfer from a smaller parent to a larger child works as well as the reverse.", option D

> [!card]- Which sampling temperature is effectively uniform?
> $T = 100$.
Reason: duplicate of "Multiple selection: which are true of language sampling in multilingual NMT? (A) Proportional sampling lets high-resource languages dominate. (B) Uniform sampling hurts low-resource languages. (C) A temperature of T = 100 approximates uniform sampling. (D) The best compromise over-samples low-resource languages.", option C

> [!card]- Which sampling temperature is best for high-resource languages and worst for low-resource ones?
> $T = 1$ (proportional).
Reason: duplicate of "Which sampling temperature gives proportional sampling?" and "Multiple selection: which are true of language sampling in multilingual NMT? (A) Proportional sampling lets high-resource languages dominate. (B) Uniform sampling hurts low-resource languages. (C) A temperature of T = 100 approximates uniform sampling. (D) The best compromise over-samples low-resource languages.", option A

> [!card]- Which multilingual model was weakest in every resource group in Arivazhagan et al. (2019)?
> **All→All**, one model for both directions.
Reason: duplicate of "Multiple selection: which are true of Arivazhagan et al.'s (2019) 103-language NMT results? (A) Low-resource languages gain most when translating into English. (B) Multilingual models beat bilingual baselines on high-resource languages. (C) All→All is weaker than the dedicated Any→En and En→Any models. (D) One-to-many gains more than many-to-one for low-resource languages.", option C

> [!card]- Do multilingual NMT models beat bilingual baselines for high-resource languages in Arivazhagan et al. (2019)?
> **No.** High-resource languages lose to interference.
Reason: duplicate of "Multiple selection: which are true of Arivazhagan et al.'s (2019) 103-language NMT results? (A) Low-resource languages gain most when translating into English. (B) Multilingual models beat bilingual baselines on high-resource languages. (C) All→All is weaker than the dedicated Any→En and En→Any models. (D) One-to-many gains more than many-to-one for low-resource languages.", option B

> [!card]- Where does mBART put the language ID token?
> At the **end of the encoder input** and the **start of the decoder input**.
Reason: duplicate of "Multiple selection: which are true of mBART? (A) It is pretrained on parallel data. (B) Its pretraining objective has no crosslingual signal. (C) A language ID token starts the decoder input. (D) Translation fine-tuning requires a new randomly initialised source encoder.", option C

> [!card]- True or false: mBART's pretraining objective uses a crosslingual signal.
> **False.** Denoising is always within the **same** language.
Reason: duplicate of "Multiple selection: which are true of mBART? (A) It is pretrained on parallel data. (B) Its pretraining objective has no crosslingual signal. (C) A language ID token starts the decoder input. (D) Translation fine-tuning requires a new randomly initialised source encoder.", option B

## Not on the slides

> [!card]- Why is sentence alignment a separate step after document alignment?
> Parallel segments **do not follow paragraph boundaries**, so links other than paragraph-to-paragraph are needed.
Reason: not on slides (the note's own explanation; the slide is only the aligned-documents figure)

> [!card]- Why does crosslingual QA expose purely multilingual models more than XNLI?
> Extractive QA needs **word-level matching** of question to context across languages; XNLI can rely on a coarse sentence gist.
Reason: not on slides (the note's own explanation)

> [!card]- Why does unfreezing the target embeddings hurt a low-resource NMT child?
> The parent learned them from far more target-language data; retraining them on the small child data **overfits**.
Reason: not on slides (the note's own explanation of the freezing chart)

> [!card]- Why does many-to-one multilingual NMT help more consistently than one-to-many?
> Many-to-one shares one **target** side; in one-to-many the decoder produces several languages that **compete** for parameters.
Reason: the reason is the note's, not on slides; the direction is kept in "True or false: Many-to-one multilingual NMT beat the bilingual baselines in every direction in Johnson et al. (2017)."

## Numbers and table details

> [!card]- What is the difference between translate-train and translate-test on XNLI?
> Translate-train machine-translates the **English training set** into each language; translate-test translates each **test set into English**.
Reason: table labels from the XNLI results, on cheat sheet

> [!card]- Does zero-shot XLM (MLM+TLM) beat translate-test on XNLI?
> **On average, slightly**, without ever seeing non-English NLI data; not in every language.
Reason: number, on cheat sheet

> [!card]- Which languages gain most on XNLI when TLM is added to MLM?
> Those **distant from English** in script or structure (e.g. Vietnamese, Turkish, Arabic, Chinese). English itself gains least.
Reason: per-language gains read off the XNLI table, not a statement the slides make

> [!card]- Which sampling temperature is the compromise between proportional and uniform in Arivazhagan et al. (2019)?
> $T = 5$.
Reason: number read off the temperature table; the slides label only T=1 and T=100

> [!card]- In parent-child NMT transfer, what does the Uzbek child score with every parameter frozen?
> **Zero BLEU**: the source embeddings must be retrained first.
Reason: number from the freezing chart

> [!card]- True or false: Zero-shot XLM (MLM+TLM) beats translate-test on XNLI for every language.
> **False.** Only on average; translate-test wins for the other European languages, Turkish and Arabic.
Reason: table detail, on cheat sheet

## Obscure

> [!card]- For XLM language modelling of Nepali, does adding Hindi or English data help more?
> **Hindi**, the related language, by far. Adding both is best.
Reason: one-off table result (Nepali perplexity)

> [!card]- Which child language gains most from parent-child transfer in Zoph et al. (2016)?
> The one with the **smallest corpus** (Urdu).
Reason: single-language result from a bar chart

> [!card]- What does NMT transfer to French' (French with randomly reshuffled vocabulary) show?
> Transfer is **not only about shared words**: structure such as word order and syntax transfers too.
Reason: side experiment; the interpretation is the note's, the slide only defines French'

> [!card]- How are InfoXLM's three losses combined?
> **Equally weighted**: a plain sum.
Reason: minor row of the InfoXLM comparison table

> [!card]- In mixed-language QA, when does XLM-R work across languages?
> **Mainly when the question is in English.**
Reason: fine reading of the QA heatmap

> [!card]- In parent-child NMT transfer, which parameter group gives the biggest single gain when unfrozen?
> The **source embeddings**.
Reason: fine reading of the freezing chart; that the source embeddings must be retrained is kept in "Multiple selection: which are true of parent-child transfer learning for low-resource NMT? (A) The child model is initialised with the parent's parameters. (B) The target-language embeddings stay frozen during child training. (C) The source embeddings stay frozen during child training. (D) Transfer from a smaller parent to a larger child works as well as the reverse.", option C
