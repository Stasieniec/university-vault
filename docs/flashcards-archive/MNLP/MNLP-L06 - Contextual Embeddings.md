# Archived flashcards: MNLP-L06 - Contextual Embeddings

The long-form cards this note carried until 2026-10-08, when they were replaced by short single-fact cards. Kept for reference; not published.

Click a question to reveal its answer, or press **Study** to drill the whole set. Cards marked as exam questions are meant to be answered out loud or on paper first, then checked against the points listed.

> [!exam]- Why can mBERT transfer a task from one language to another when nothing in its training is cross-lingual? Argue from the experimental evidence.
> 1. **What mBERT lacks.** Same architecture and loss as BERT (MLM + NSP) on 104 Wikipedias; no parallel data, no language-ID embedding, no language-specific parameters, no alignment loss, NSP pairs always within one language. Languages share only the **parameters** and the **subword vocabulary**.
> 2. **Vocabulary overlap is a minor cause.** Pires et al. (2019): mBERT transfers across scripts (Hindi to Urdu POS 85.9, English to Bulgarian 87.1), and its zero-shot NER F1 is roughly flat in entity-wordpiece overlap, while English BERT's F1 is near 0 at low overlap and rises with it. K et al. (2020): fake English removes all subword overlap and costs only **0.5 to 1.4** XNLI points.
> 3. **Structure matters.** Transfer is best within the same word-order type (SVO to SVO 81.55, SVO to SOV 66.52) and rises with the number of shared WALS features. Permuting all words during pre-training drops XNLI by 8.4 (Spanish), 16.5 (Hindi) and 12.1 (Russian), yet transfer stays well over chance.
> 4. **Depth matters.** At roughly constant parameter count, the gap between fake-English and Russian XNLI shrinks from 21.6 (1 layer) to 11.3 (24 layers): deeper networks learn more language-independent representations.
> 5. **Bottom line:** no single factor explains transfer, or the lack of it.
> 6. **The limit.** When premise and hypothesis are in different languages, accuracy falls under both monolingual settings, so the shared space is not truly language-neutral. Rajaee and Monz hypothesise that transfer of heuristics (such as premise-hypothesis word overlap) contributes to cross-lingual generalisation; mixed-language inputs remove that shortcut.
>
> Losing marks: calling shared wordpieces the main cause; claiming mBERT saw parallel data or an alignment objective.

> [!exam]- Explain why static word embeddings are inadequate and how contextual embeddings fix this. What design tension do contextual models face, and how can it be managed?
> - **Static embeddings**, whether a by-product of a task (embedding layer of a classifier or MT system) or trained directly (Word2Vec with negative sampling, limited window), give one vector per word type: $E(\text{train}_1)=E(\text{train}_2)=E(\text{train}_3)$ for the vehicle, the verb "to practise" and "train of thought". Three senses, two parts of speech, one vector.
> - In the by-product route the network did see the whole sentence, but only the input embedding layer was kept, so the context computed in deeper layers was thrown away.
> - **Contextual embeddings** keep it: a deep network reads the whole sentence and every occurrence gets its own vector (a hidden state in that word's column).
> - **Tension:** integrate enough context to distinguish occurrences of the same word, without capturing so much that the word's own contribution becomes unclear (every position becomes "the meaning of the sentence").
> - **Managed by:** (1) choosing the right layer(s): lower layers are closer to the word, higher closer to the sentence; (2) choosing the right training task(s): per-token outputs keep each top state about its word, a single pooled output pushes the top layer toward the sentence; (3) tightening connections between layers, e.g. residual connections, which carry the word's own information upward alongside the added context.

> [!exam]- Describe how BERT is pre-trained and fine-tuned: input representation, both pre-training objectives, the role of `[CLS]`, and how task heads attach.
> - **Model:** encoder part of the Transformer, bidirectional (every position attends left and right). Devlin et al., NAACL 2019.
> - **Input:** `[CLS] A [SEP] B [SEP]`, with $x_i = E_{\text{token}(i)} + E_{\text{segment}(i)} + E_i$ (token + segment A/B + position embedding). WordPiece tokens.
> - **MLM:** select 15% of tokens; replace 80% with `[MASK]`, 10% with a random token, leave 10% unchanged; loss on the selected positions only, predicting the original token.
> - **NSP:** 50% B really follows A (IsNext), 50% B is random (NotNext); classified from $C$, the final-layer output at `[CLS]`.
> - **`[CLS]`:** never masked, not tied to a word, represents the whole sequence, trained through a classification loss.
> - **Fine-tuning:** small randomly initialised head; connect it to the token outputs $T_i$ for tagging or span extraction, to $C$ for classification; train on the task with or without updating BERT. Batch 16 or 32, 2 to 4 epochs.
> - Self-supervised pre-training can use huge amounts of task-irrelevant data; supervised fine-tuning then works with little task data.
>
> Losing marks: saying the loss covers all tokens, or only the positions showing `[MASK]`; saying BERT-base has 16 attention heads (it has 12).

> [!exam]- Describe how mBERT is trained and what it deliberately lacks. Explain its language sampling scheme and the problems a single shared multilingual vocabulary causes.
> - Released on GitHub in 2018 with no paper. Exact BERT architecture and loss (MLM + NSP), on the concatenated Wikipedias of **104 languages**.
> - **Absent:** parallel data (at least intentionally), language-ID embedding, language-specific parameters such as adapters, any alignment loss. NSP pairs are always in one language; no language mixing within a sequence (a batch can mix languages).
> - **Sampling:** Wikipedia sizes are extremely skewed (English alone exceeds dozens of small languages combined), so languages are drawn with $p_l = n_l^\alpha / \sum_{l'} n_{l'}^\alpha$. $\alpha=1$: proportional, big languages dominate; $\alpha=0$: uniform, tiny languages are memorised; $\alpha=0.7$: up-weights low-resource and down-weights high-resource languages.
> - **Vocabulary:** one shared WordPiece vocabulary of 120k (English BERT 30k), which is why mBERT has about 178M parameters against 110M. CJK characters are split into single characters before WordPiece. The sampling smoothing is also applied to the counts used to build the vocabulary.
> - **Problems:** even smoothed, the vocabulary favours high-resource languages and Latin script; low-resource and morphologically rich languages get much higher fertility (more pieces per word); the uncased variant lowercases and strips accents, which damages languages that depend on diacritics.

> [!exam]- Calculate: English has 1000 units of training text and Swahili 10. Using $p_l = n_l^\alpha/\sum_{l'} n_{l'}^\alpha$, give both sampling probabilities for $\alpha = 1$, $0.7$ and $0$, and interpret each.
> - $\alpha = 1$: $p_{sw} = 10/1010 = 0.0099$, $p_{en} = 0.990$. Proportional to size; English dominates.
> - $\alpha = 0.7$: $1000^{0.7} = 125.9$, $10^{0.7} = 5.01$, so $p_{sw} = 5.01/130.9 = 0.038$ and $p_{en} = 0.962$. Swahili gets about 3.9 times its raw share; English still gets 96%.
> - $\alpha = 0$: $p_{sw} = p_{en} = 0.5$. Swahili's share grows about 50-fold over its raw share, and each unit of Swahili text is drawn 100 times as often as each unit of English ($0.5/10$ against $0.5/1000$): the model would memorise the small language.
> - General rule: the ratio between two languages goes from $n_1/n_2$ to $(n_1/n_2)^\alpha$, because $x^\alpha$ with $\alpha<1$ is concave and compresses large values more. Here 100:1 becomes about 25:1 at $\alpha=0.7$.

> [!exam]- Compare XLM-R with mBERT: what changed in training, and how do the two compare within a language and across languages on XNLI and cross-lingual QA?
> - **XLM-R** (Conneau et al. 2020) is a RoBERTa-style extension of mBERT: **MLM only** (NSP dropped: contributes little, sometimes hurts), **dynamic masking**, **CC-100** (2.5TB filtered CommonCrawl, 100 languages, two orders of magnitude more data than Wikipedia), **250k** Unigram LM vocabulary via SentencePiece (better fertility, bigger embedding matrix).
> - **Within** a language XLM-R is better on both tasks: XNLI 74.2 against 65.7 (+8.5), XSQuAD F1 72.0 against 64.4 (+7.6).
> - **Across** languages on XNLI it is also better: 64.8 against 54.5 (+10.3); gap within to across 9.4 against 11.2.
> - **Across** languages on QA it is **worse**: 36.8 against 44.2 F1, a within-across gap of 35.2 against 20.2. Thai: within QA rises from 40.0 to 66.5, across barely moves (26.1 to 29.6).
> - XLM-R's QA matrix is "own language, or English question": strong diagonal and English-question column (58.2 to 75.0), little in between. mBERT degrades more smoothly.
> - Conclusion: more data and a bigger vocabulary made each language better on its own without making the model better at relating two languages inside one input.

> [!exam]- What happens when the two inputs of one task are in different languages? Give the evidence for B-BERT, mBERT and XLM-R, and a possible explanation.
> - Standard zero-shot transfer is English fine-tuning, then monolingual testing in another language. Cross-lingual inference within a task means e.g. an English premise with a Spanish hypothesis.
> - **K et al. (2020), B-BERT on XNLI:** fake English pairs 78.5 to 79.3, target-language pairs 59.6 to 70.9, mixed pairs 45.7 to 61.1, under **both** monolingual conditions (Hindi 45.7, 13.9 points under its own zero-shot score). A fake-English hypothesis beats a fake-English premise.
> - **Rajaee and Monz (2024), mBERT on XNLI, 15 languages:** within 65.7 against across 54.5 on average. A Swahili hypothesis is near chance (40 to 42.5).
> - **XLM-R:** better within and across on XNLI, but across-language QA falls to 36.8 F1 against mBERT's 44.2.
> - **Interpretation:** the model can do the same task in another language but is poor at relating two languages inside one input, so the space is not language-neutral.
> - **Heuristics hypothesis:** transfer of heuristics can contribute to cross-lingual generalisation. In SNLI full word overlap means entailment 94.7% of the time; a model can learn "high overlap means entailment", which works in any single language but fails when premise and hypothesis are in different languages, where overlap is near zero. (How exactly this explains the drop is an interpretation, not an established result.)

> [!exam]- What do layer-wise analyses reveal about where mBERT holds language-neutral information? Use translation retrieval and parameter freezing, and link to the choice of layer.
> - **Translation retrieval (Pires et al.):** per layer, mean-pool hidden states (excluding `[CLS]`, `[SEP]`), shift English vectors by the mean EN-to-DE difference, retrieve the nearest German sentence by $\ell_2$. Accuracy follows an **inverted U**: 20 to 33% at layer 1, peak 71 to 76% at layers 6 to 8, falling in the last layers.
> - Reading: low layers are close to language-specific surface tokens; middle layers are the most language-neutral (one constant offset maps one language onto the other); top layers are shaped by predicting words in a specific language.
> - **Freezing (Wu and Dredze 2019):** fine-tune on English with the lowest $k$ layers frozen. Feature-based use without fine-tuning is always worst (loses 1.9 to 14.8 average points). Freezing the embeddings up to layer 3 or 6 helps or is neutral (best at layer 0/3 for NER, 3 for POS, 6 for MLDoc and XNLI), since English-only fine-tuning would pull lower layers toward English. Freezing up to layer 9 hurts (NER 67.3 against 74.3): upper layers must adapt to the task.
> - Both measure the "choose the right layer" strategy for balancing word and context information.

> [!exam]- Explain how subword segmentation interacts with masked language modelling and with multilingual models: WordPiece, whole-word masking, and the vocabulary choices of mBERT and XLM-R.
> - **WordPiece** (BERT, mBERT): bottom-up merges by $\text{count}(a,b)/(\text{count}(a)\,\text{count}(b))$; `##` marks non-initial pieces; inference is greedy longest match; a word with an unmatchable character becomes `[UNK]` as a whole.
> - **Masking problem:** pieces carrying morphology (tense, number, case) are easy to predict from their visible stem, so per-token masking wastes many masked positions on trivially recoverable suffixes.
> - **Whole-word masking:** mask all pieces of a word together at the same overall rate (about 15%); improves BERT-Large on SQuAD 1.1 (uncased F1 91.0 to 92.8) and MultiNLI (86.05 to 87.07).
> - **mBERT:** 120k shared WordPiece vocabulary with smoothed counts, CJK split into characters, still biased toward high-resource languages and Latin script, higher fertility for low-resource and morphologically rich languages; the uncased variant strips accents.
> - **XLM-R:** 250k Unigram LM vocabulary via SentencePiece, fewer pieces per word, at the cost of a larger embedding matrix.
> - Wordpiece overlap itself contributes little to transfer (fake English costs 0.5 to 1.4 XNLI points).

> [!card]- What are the two ways of obtaining static word embeddings, and how do they differ in model and context used?
> - **As a by-product of an actual task:** the model can be arbitrarily complex and deep and uses the **whole sentence**; e.g. the embedding layer of a text classifier or an MT system.
> - **Directly as the training objective:** a simple model (Word2Vec, skip-gram or CBOW, trained with negative sampling) using a **limited context window**.
> - Either way the result is a table with one row per word type, the same vector in every sentence.

> [!card]- In a stacked left-to-right RNN, how does the representation in the column of one word (e.g. *train*) change from the input up to the top layer, and how does the output configuration affect it?
> - Input: the static embedding $E(\text{train})$. First hidden layer: already mixes in the words to its left. Top layer: three rounds of mixing.
> - With an **output on every position** (tagging, next-word prediction) the top state is trained to stay useful for a prediction about its own word.
> - With **one pooled vector** fed by all top states (sentence encoder for a classifier or decoder) no position is supervised alone, so the top layer is pushed to be about the sentence and the word's own contribution can be diluted.
> - Any hidden state in the column is a candidate contextual embedding; which is useful depends on how the network was trained.

> [!card]- State the pre-training phase of the contextual representation workflow, with its notation.
> - **Model:** $M_{C,\theta}$, a model that can model wider contexts (LSTM, Transformer, CNN), with parameters $\theta$; $C$ for contextual.
> - **Task:** a general task $T_P$ with very large amounts of training data, e.g. word prediction in language modelling.
> - **Train** $M_{C,\theta}$ on $T_P$, updating $\theta$.

> [!card]- State the fine-tuning phase of the contextual representation workflow. What is the difference between feature extraction and full fine-tuning?
> - Choose a real task $T_R$ (e.g. QA, POS tagging) and a task model (head) $M_{R,\theta'}$.
> - Combine into $M_{F,\theta\cup\theta'}$, whose parameters are the union of both.
> - Train it on the **real task $T_R$**, updating $\theta'$ and maybe also $\theta$. (Training on $T_P$ again would just be more pre-training; a version stating $T_P$ here is a known typo.)
> - Updating only $\theta'$: the pre-trained model is a frozen **feature extractor**. Updating both: full **fine-tuning**, BERT's default.

> [!card]- Name the three Transformer-based architecture families, what each does, and a prominent example of each.
> - **Encoder-decoder:** encoder builds a representation of the input, decoder generates the output the task needs; the original Transformer (Vaswani et al. 2017); summarisation, QA, MT.
> - **Encoder-only:** predicts its own input scrambled by a noise function, in its purest form an auto-encoder; **BERT**.
> - **Decoder-only:** classical next-word-prediction language model; **all current LLMs** are decoder-only.

> [!card]- How can an encoder-decoder task be framed in a decoder-only model?
> Concatenate input and output into one sequence (e.g. `translate to German: <source> <target>`) and train a next-word predictor on it. The encoder's job is absorbed into the decoder's processing of the prefix.

> [!card]- What does BERT stand for, what part of the Transformer is it, what makes it "bidirectional", and why is its pre-training called self-supervised?
> - **Bidirectional Encoder Representations from Transformers** (Devlin et al., NAACL 2019).
> - The **encoder** part of the Transformer; every position attends to every other position, left and right (unlike a left-to-right RNN).
> - Pre-training is missing-word prediction: the labels are the words themselves, so huge amounts of data can be used and the data need not be relevant to any downstream task. Fine-tuning on the real task is **supervised** and works with small amounts of data.

> [!card]- In the standard BERT pre-training and fine-tuning diagram, what are $E$, $T_i$ and $C$, and what input pair is used for MNLI, NER and SQuAD?
> - $E_\cdot$: input embeddings; $T_i$: final-layer contextual outputs; $C$: final-layer output at `[CLS]`.
> - Pre-training: $C$ feeds the NSP classifier, the $T_i$ at masked positions feed the MLM predictor. The same pre-trained parameters initialise every fine-tuning model.
> - **MNLI:** premise and hypothesis. **NER:** a single sentence. **SQuAD:** question and paragraph, output a start/end span over the paragraph tokens.

> [!card]- How is BERT's input vector at position $i$ formed? Illustrate with *my dog is cute* / *he likes playing*.
> $$x_i = E_{\text{token}(i)} + E_{\text{segment}(i)} + E_i$$
> - Token embedding + segment embedding ($E_A$ or $E_B$, telling the model which sentence the token belongs to; the first `[SEP]` belongs to A) + positional embedding $E_0, E_1, \dots$
> - Sequence: `[CLS] my dog is cute [SEP] he likes play ##ing [SEP]`, positions 0 to 10. *playing* is split by WordPiece into `play ##ing`, `##` marking a word-internal piece.

> [!card]- State BERT's masking rule exactly, including where the loss is computed.
> - Select tokens with probability $p_{\text{mask}} = 0.15$ (special tokens `[CLS]`, `[SEP]` are not masked).
> - A selected token is replaced **80%** of the time by `[MASK]`, **10%** by a random word, **10%** left as the original word.
> - **The loss is computed only at the selected positions**, whatever they show (mask, random word or original), and the target is the original token.

> [!card]- Why does BERT not simply replace every selected token by `[MASK]`?
> `[MASK]` never occurs at fine-tuning time. A model that only ever sees `[MASK]` at prediction positions could learn good representations only where `[MASK]` appears. Random and unchanged tokens mean it cannot tell which positions will be scored, so it must build a good representation of every input token. (This rationale is Devlin et al.'s; the 80/10/10 split itself is the core fact.)

> [!card]- Write BERT's MLM loss for one sequence and say what each term means.
> $$\mathcal{L} = -\sum_{i \in S} \log \operatorname{softmax}(W T_i)[x_i]$$
> - $S$: the selected (scored) positions; $T_i$: final-layer output of BERT on the corrupted input at position $i$; $W$: output projection to the vocabulary; $x_i$: the **original** token at $i$.

> [!card]- What is the `[CLS]` class embedding, and what are its four defining properties?
> 1. Not tied to a specific word and **never masked**.
> 2. Not meant to learn context-specific word representations.
> 3. Captures a **general representation of the entire input sequence**.
> 4. Its loss is computed with respect to a **classification task** (next sentence prediction in pre-training, the task label in fine-tuning).

> [!card]- How is a next sentence prediction training example built, and from which output is it classified?
> - Take a document and a segment A from it.
> - 50%: B is the segment that follows A (IsNext); 50%: B is from a random document (NotNext).
> - Input `[CLS] A [SEP] B [SEP]`; loss $-\log P(\text{label} \mid C)$, with $C$ the final-layer output at `[CLS]`.

> [!card]- Give BERT's pre-training configuration: batch, sequence content, corpus and tokenisation.
> - Batch of 256 sequences of 512 tokens, quoted as "128,000 tokens per batch" (exactly $256 \times 512 = 131{,}072$).
> - Sentences A and B are in practice segments of running text, much longer than single sentences.
> - BookCorpus (800M tokens) plus English Wikipedia (2.5B tokens).
> - WordPiece tokenisation.

> [!card]- Give the sizes of BERT-base and BERT-large: parameters, layers, hidden size, attention heads, per-head dimension.
> - **BERT-base:** 110M parameters, 12 layers, hidden 768, **12 heads**, so $768/12 = 64$ dimensions per head.
> - **BERT-large:** 340M parameters, 24 layers, hidden 1024, 16 heads, $1024/16 = 64$ per head.
> - A figure of 16 heads for BERT-base is an error.

> [!card]- List the steps of fine-tuning BERT and the typical hyperparameters. Why is fine-tuning kept short?
> 1. Choose a small-ish task-specific model, randomly initialised.
> 2. Connect it to the top-layer token outputs $T_i$ for tasks needing word representations (tagging, span extraction).
> 3. Connect it to the top-layer `[CLS]` output $C$ for classification (sentiment, entailment).
> 4. Train on the task data, with or without updating BERT's parameters.
>
> Batch size 16 or 32, 2 to 4 epochs. The encoder starts out already good, and long fine-tuning on small data overfits and erodes what pre-training learned.

> [!card]- What is mBERT? Give its origin, architecture, data and how NSP and language mixing are handled.
> - Released **on GitHub in 2018, with no paper**.
> - Exactly the BERT architecture and loss (MLM + NSP): no cross-lingual loss, nothing language-specific.
> - Trained on the concatenation of **Wikipedia in 104 languages**, sampled with a smoothing parameter.
> - NSP sentence pairs are **always within one language**; no mixing of languages inside a sequence, though a batch can contain several languages.
> - One shared WordPiece vocabulary for all languages.

> [!card]- List what mBERT's training deliberately does not contain, and say what the languages therefore share.
> - No parallel data (at least not intentionally).
> - No language-ID embedding.
> - No language-specific parameters (e.g. adapters).
> - No alignment loss rewarding translation equivalents for being close.
> - NSP pairs only within one language.
>
> Nothing tells the model that *Hund* and *dog* mean the same thing. Languages share only the **parameters** and the **subword vocabulary**, so any cross-lingual ability must come from these.

> [!card]- Why does mBERT need weighted language sampling? Give the formula and define each term.
> Wikipedia sizes vary enormously; English alone is larger than dozens of smaller languages combined.
> $$p_l = \frac{n_l^{\alpha}}{\sum_{l'} n_{l'}^{\alpha}}$$
> - $p_l$: probability of drawing training data from language $l$.
> - $n_l$: size of language $l$'s resource (characters, tokens, etc.).
> - $\alpha$: controls how strongly raw sizes are respected.
> - The sum over all languages normalises the probabilities.

> [!card]- What happens with language sampling at $\alpha = 1$, $\alpha = 0$ and $\alpha = 0.7$, and why does $\alpha < 1$ shrink the gap between languages?
> - $\alpha = 1$: proportional to raw size; English and a few big Wikipedias dominate.
> - $\alpha = 0$: uniform ($n^0 = 1$); tiny Wikipedias are oversampled so much that the model mostly memorises them.
> - $\alpha = 0.7$: up-weights low-resource and down-weights high-resource languages, a compromise.
> - $x^\alpha$ with $\alpha<1$ is concave, compressing large values more than small ones: a size ratio $n_1/n_2$ becomes $(n_1/n_2)^\alpha$, e.g. 100:1 becomes about 25:1 at 0.7.

> [!card]- How big is mBERT's vocabulary compared to English BERT's, and how does this explain the parameter counts?
> - mBERT: **one shared vocabulary of 120k** entries for 104 languages; English BERT: 30k.
> - mBERT has about **178M parameters against BERT-base's 110M** with the same architecture; the difference is due to the vocabulary. Each extra entry adds a 768-dimensional embedding row: $90\text{k} \times 768 \approx 69\text{M}$.

> [!card]- Describe four special properties or problems of mBERT's shared vocabulary.
> 1. **CJK:** Chinese characters, Japanese kanji and Korean hanja are split into individual characters before WordPiece.
> 2. **Smoothed counts:** the language-sampling smoothing also applies to the frequencies used to build the vocabulary, giving small languages more entries.
> 3. **Remaining bias:** even so, the vocabulary favours high-resource languages and Latin script; low-resource and morphologically rich languages are split into more pieces per word (**higher fertility**).
> 4. **Uncased variant:** lowercases and strips accents, damaging languages that depend on diacritics.

> [!card]- What is WordPiece, who proposed it, how does it relate to BPE historically and in direction, and what is its pair score?
> - Schuster and Nakajima (2012); very similar to BPE, and predates BPE as an NLP segmentation method (Sennrich et al. 2016), though BPE as a compression algorithm is older.
> - Bottom-up like BPE (start from characters, merge); Unigram LM is top-down.
> $$\text{score}(a,b) = \frac{\text{count}(a,b)}{\text{count}(a)\cdot\text{count}(b)}$$
> - A pair scores high when it occurs together more often than the frequencies of its parts suggest (association), where BPE merges the raw most frequent pair.

> [!card]- Give the steps of WordPiece training.
> 1. Assume word boundaries; collect word types with frequencies.
> 2. Split each word into characters, prefixing every **non-initial** character with `##` (*hug* becomes `h ##u ##g`); the vocabulary starts as all such symbols.
> 3. While the vocabulary is smaller than the target size $K$: score every adjacent pair with $\text{count}(a,b)/(\text{count}(a)\,\text{count}(b))$.
> 4. Merge every occurrence of the best pair into one symbol (the `##` of the second part is dropped) and add it to the vocabulary.

> [!card]- Give the steps of WordPiece inference for one word. What happens to a word containing an unknown character?
> 1. Start at the first character.
> 2. Try the **longest** remaining substring first, shortening from the right until a candidate is in the vocabulary; non-initial candidates carry `##`.
> 3. Append the match and continue after it.
> 4. If no candidate matches at some point, the **whole word** becomes `[UNK]`.
>
> E.g. *playing* becomes `play ##ing`, *tallest* becomes `tall ##est`. Inference does not replay merges, so it can segment differently from replaying the training merges.

> [!card]- Worked WordPiece step: types *hug* 10, *pug* 5, *pun* 12, *bun* 4, *hugs* 5. Which pair is merged first, and which would BPE merge?
> - Symbol counts: `h` 15, `##u` 36, `##g` 20, `p` 17, `##n` 16, `b` 4, `##s` 5.
> - Pairs: `##u ##g` 20, `p ##u` 17, `##u ##n` 16, `h ##u` 15, `b ##u` 4: all score $1/36 \approx 0.028$ (co-occurring with the very common `##u` is unsurprising).
> - `##g ##s`: $5/(20 \cdot 5) = 0.05$, the highest, because `##s` only ever follows `##g`. WordPiece merges `##gs`, the rarest pair.
> - BPE would merge `##u ##g`, the most frequent pair (20).

> [!card]- Compare BPE, WordPiece and Unigram LM on direction, merge/keep criterion, inference, output and unknown handling, and name typical models for each.
> - **BPE:** bottom-up; raw pair frequency; replay merges in order; deterministic; character/byte fallback; GPT family, RoBERTa, original XLM.
> - **WordPiece:** bottom-up; association score (likelihood gain); greedy longest match; deterministic; whole word becomes `[UNK]` (the harshest); BERT, mBERT, DistilBERT.
> - **Unigram LM:** top-down pruning; loss in corpus likelihood under EM; Viterbi or sampling; deterministic or sampled; character fallback; T5, XLM-R, ALBERT (via SentencePiece).

> [!card]- Why are some masked tokens or subword pieces easier to predict than others, and what does this do to MLM?
> - Content words (nouns, adjectives, verbs) are harder to predict than function words (determiners, prepositions).
> - The same holds for BPE pieces: masking the stem (`[MASK] ed`) leaves a hard question (which verb?), while masking the suffix after a visible stem (`view@@ [MASK]`) is nearly certain.
> - Pieces carrying mostly **morphological** information (tense, number, case) are easier than pieces carrying root/stem information.
> - Consequence: with per-token random masking many masked positions are trivially recoverable suffixes and continuation pieces, so the model gets loss signal without learning much and MLM becomes effectively easier.

> [!card]- What is whole-word masking, how is it implemented, and what results did it give for BERT-Large?
> - Always mask **all subword tokens of the same word** together; the masking rate is unchanged (about 15% of tokens). `view@@ ed` becomes `[MASK] [MASK]`.
> - Implementation: group tokens into words, shuffle the words, add whole words to the masked set until the token budget of about 15% is reached, then apply 80/10/10 and score only those positions.
> - BERT-Large uncased: SQuAD 1.1 F1/EM 91.0/84.3 to 92.8/86.7, MultiNLI 86.05 to 87.07. Cased: 91.5/84.8 to 92.9/86.7, MultiNLI 86.09 to 86.46. Improves every column; the model can no longer complete a word from its own visible pieces.

> [!card]- What is zero-shot cross-lingual transfer, and how is it read off a fine-tune by evaluate table (Pires et al. 2019)?
> Fine-tune on task data in language A only, evaluate on the same task in language B with no task data in B. In the table (rows fine-tuning language, columns evaluation language) the **diagonal** is ordinary in-language performance and every **off-diagonal** cell is zero-shot. Pires et al. (2019) studied mBERT this way over 16 languages.

> [!card]- What did Pires et al. find for mBERT zero-shot NER and POS between English, German, Dutch, Spanish and Italian?
> - Performance is decent off the diagonal, showing generalisation beyond the fine-tuning language.
> - NER: English-trained reaches 77.36 F1 on Dutch against 89.86 in-language (about 86%).
> - POS transfers better in absolute terms (English to German 89.40 against 93.99), and closely related pairs best (Spanish to Italian 93.71, Italian to Spanish 91.28).
> - Transfer is asymmetric: English to Dutch NER 77.36, Dutch to English 65.46.

> [!card]- How did Pires et al. test whether mBERT's transfer depends on shared script, and what did they find?
> - POS between languages with different scripts, which share almost no wordpieces.
> - Hindi (Devanagari) to Urdu (Perso-Arabic) 85.9, Urdu to Hindi 91.1; English (Latin) to Bulgarian (Cyrillic) 87.1. mBERT generalises well across scripts.
> - Scenarios involving **Japanese** drop sharply (English to Japanese 49.4, Bulgarian to Japanese 51.6, Japanese to English 57.4), possibly due to larger typological differences.

> [!card]- Give the entity-wordpiece overlap measure of Pires et al. and define its terms.
> $$\text{overlap} = \frac{|E_{l,t} \cap E_{l',e}|}{|E_{l,t} \cup E_{l',e}|}$$
> A Jaccard similarity over wordpieces occurring in labelled named entities only. $E_{l,t}$: entity wordpieces of the fine-tuning data in language $l$; $E_{l',e}$: entity wordpieces of the evaluation data in language $l'$. 0 for disjoint sets, 1 for identical.

> [!card]- What does the scatter plot of zero-shot NER F1 against entity-wordpiece overlap show for mBERT and English BERT, and what follows?
> - **English BERT:** near 0 F1 under about 10% overlap, rising roughly linearly to 40 to 70 F1 at 25 to 38% overlap. It transfers only through shared surface strings.
> - **mBERT:** overlap only 0 to about 27%, F1 between roughly 40 and 82 across the whole range, already 40 to 70 near 0 overlap: essentially flat.
> - mBERT's transfer is not string matching; it has a representation shared across languages beneath the level of surface wordpieces.

> [!card]- What is WALS? Give its size, coverage, structure and use in NLP.
> - The **World Atlas of Language Structures**, a typological database: a large catalogue of how languages are structured (phonological, grammatical, lexical properties), compiled from reference grammars by 55 authors.
> - **192 features**, each with a small set of values (e.g. *Tone*: no tones, simple, complex).
> - **2,662 languages**, but sparse: no language has every feature, no feature covers every language; English has the most (159), still missing 33.
> - NLP use: a measure of structural similarity between languages (word order, number of cases), with languages represented as feature vectors.

> [!card]- What did Pires et al. find about typological similarity and mBERT's zero-shot POS transfer?
> - **Subject/verb/object order:** SVO to SVO 81.55, SVO to SOV 66.52 (a 15-point gap for the same fine-tuning data); SOV to SVO 63.98, SOV to SOV 64.22.
> - **Adjective/noun order:** smaller effect; AN to AN 73.29, AN to NA 70.94; NA to NA 79.64.
> - **Shared WALS features:** accuracy rises with the number of common features for both models (mBERT about 58 at 1 feature to 77 at 6; English BERT about 31 to 50), mBERT 27 to 40 points higher, with wide error bars.
> - Transfer is best between structurally similar languages.

> [!card]- Describe the translation-retrieval method Pires et al. use to test whether mBERT's sentence representations align across languages.
> 1. Sample $M = 5\text{k}$ translation pairs from WMT16.
> 2. For each layer $l$, represent a sentence by the mean of its hidden activations, excluding `[CLS]` and `[SEP]`.
> 3. Compute one mean offset for the language pair:
> $$\bar v^{(l)}_{\text{EN}\to\text{DE}} = \frac{1}{M}\sum_i \left(v^{(l)}_{\text{DE}_i} - v^{(l)}_{\text{EN}_i}\right)$$
> 4. "Translate" each English sentence as $v^{(l)}_{\text{EN}_i} + \bar v^{(l)}_{\text{EN}\to\text{DE}}$.
> 5. Find the nearest German sentence by **$\ell_2$ distance** (cosine is not used); a hit is the true translation. Accuracy = hits / $M$.

> [!card]- What layer-wise pattern does mBERT translation retrieval show for EN-DE, EN-RU and UR-HI, and why?
> - All three follow an **inverted U**: 20 to 33% at layer 1, peak 71 to 76% in layers 6 to 8, then falling.
> - EN-RU (different scripts) starts lowest (about 20%) but catches up by layer 8; UR-HI peaks earliest (layer 6) and falls fastest; EN-DE peaks at about 76% at layer 8.
> - Lower layers are close to language-specific surface tokens; middle layers are the most language-neutral, so one constant shift maps one language onto the other; top layers are shaped by predicting words in a given language and become language-specific again.

> [!card]- What did Wu and Dredze (2019) set out to test, and with which five tasks?
> The desired property: a common, aligned embedding space, where training on language A for task X improves performance on language B for task X with no task data in B (zero-shot), English being the fine-tuning language. Tasks: document classification (**MLDoc**, document level), entailment (**NLI**/XNLI, sentence pairs), **NER** and **POS** (tokens), dependency **parsing** (token pairs). 8, 15, 5, 15 and 31 languages respectively.

> [!card]- Describe the MLDoc task and mBERT's results on it (Wu and Dredze).
> - Document classification into four classes: CCAT (Corporate/Industrial), ECAT (Economics), GCAT (Government/Social), MCAT (Markets). Only the first two sentences are used, due to memory constraints.
> - In-language mBERT averages 91.2, beating Schwenk and Li (2018, 89.5) everywhere except German.
> - Zero-shot mBERT averages 74.5, just under Artetxe and Schwenk's 74.9, which was trained with parallel text.
> - Zero-shot costs a lot: 91.2 to 74.5 on average; Japanese drops from 88.4 to 56.5.

> [!card]- Define natural language inference and describe SNLI and XNLI, including their sizes.
> - **NLI:** given a premise and a hypothesis, the premise entails, contradicts or is neutral to the hypothesis: 3-way classification.
> - **SNLI** (Bowman et al. 2015): 500k+ pairs, English only, commonly used for training; gold label is the majority of five annotator judgements, which can disagree in all three directions.
> - **XNLI:** crowd-sourced 5,000 test and 2,500 dev pairs, translated into 14 languages (15 in total, 112.5k annotated pairs); pairing any premise language with any hypothesis language gives more than 1.5M combinations ($7{,}500 \times 15 \times 15$).

> [!card]- How does mBERT do on XNLI under pseudo-supervision and zero-shot transfer, and what is the fair comparison (Wu and Dredze)?
> - **Pseudo-supervision** (machine-translate the English training set into each target language): 71.6 average, better than zero-shot 66.3.
> - Zero-shot mBERT (66.3) beats the older X-LSTM (65.6) but loses to systems using parallel data or bilingual signal: Artetxe and Schwenk 70.2, Lample and Conneau (2019) MLM+TLM 75.1.
> - Fair comparison: Lample and Conneau (2019) **MLM only**, also trained without cross-lingual signal: 71.5 against mBERT's 66.3.
> - Worst languages are low-resource and distant: Swahili 50.4, Thai 55.8, Urdu 58.0, Hindi 60.0, Turkish 61.6.

> [!card]- How does zero-shot mBERT compare with dedicated systems on NER and POS (Wu and Dredze)?
> - **NER:** zero-shot mBERT averages 74.03 F1 (nl, es, de) against 67.13 for the cross-lingual system of Xie et al. (2018), about 7 points better. Chinese collapses to 51.90 against 93.17 in-language.
> - **POS:** zero-shot mBERT averages 84.3, under Kim et al. (2017) with only 320 target sentences (89.9) or 1280 (93.3). A small amount of target-language data is worth more than mBERT's cross-lingual knowledge. Weakest: Persian 72.8, Dutch 75.9.

> [!card]- Describe the parameter-freezing experiment of Wu and Dredze and the meaning of "Feat" and "Lay $k$".
> - Fine-tune mBERT on English while keeping the $k$ lowest layers frozen, $k \in \{0, 3, 6, 9\}$, where 0 is the embedding layer; "Lay $k$" means layers up to $k$ are frozen. Evaluate zero-shot on other languages for MLDoc, XNLI, NER and POS.
> - "Feat": the feature-based setting, where mBERT is not fine-tuned at all and only the task head is trained.

> [!card]- What were the findings of Wu and Dredze's layer-freezing experiment?
> - **Feat is always worst**: loses 1.9 (POS), 3.5 (NER), 4.0 (NLI) and 14.8 (MLDoc) average points against the best setting. Fine-tuning is necessary.
> - **Freezing lower layers (embeddings up to layer 3 or 6) helps or is neutral**: best at Lay 0/3 for NER (74.3), Lay 3 for POS (85.2), Lay 6 for MLDoc (77.4) and XNLI (67.1). Frozen lower layers keep their multilingual representation instead of being pulled toward English.
> - **Freezing up to layer 9 hurts**, most for NER (67.3 against 74.3): upper layers need to adapt to the task.
> - The effect is largest for MLDoc and small for POS.

> [!card]- Give Wu and Dredze's observed-wordpiece percentages $p_{\text{type}}$ and $p_{\text{token}}$ and define each term.
> $$V^{\ell}_{\text{obs}} = V^{\text{en}}_{\text{train}} \cap V^{\ell}_{\text{test}}$$
> $$p^{\ell}_{\text{type}} = \frac{|V^{\ell}_{\text{obs}}|}{|V^{\ell}_{\text{test}}|}\cdot 100 \qquad p^{\ell}_{\text{token}} = \frac{\sum_{w \in V^{\ell}_{\text{obs}}} c^{\ell}_w}{\sum_{w \in V^{\ell}_{\text{test}}} c^{\ell}_w}\cdot 100$$
> - $V^{\text{en}}_{\text{train}}$: wordpiece types in English training data; $V^{\ell}_{\text{test}}$: types in language $\ell$'s test data; $V^{\ell}_{\text{obs}}$: test types seen in English training; $c^{\ell}_w$: frequency of $w$ in the test set.
> - $p_{\text{type}}$: % of test types seen; $p_{\text{token}}$: % of test tokens whose wordpiece was seen.

> [!card]- How strongly does wordpiece overlap with English training data correlate with zero-shot performance per task (Wu and Dredze), and what is the caveat?
> - **NER:** almost perfect, type $R = 0.99$, token $R = 0.98$ (the most lexical task, entities often copied verbatim; only 5 languages).
> - **XNLI:** no type-level correlation, $R = -0.036$ (token 0.36, not significant): sentence-level inference does not depend on shared wordpieces.
> - **MLDoc, POS, parsing:** in between, $R$ 0.5 to 0.8; low-overlap languages score lowest but with wide spread.
> - Caveat: correlation is not cause. Languages with high overlap with English are also typologically closer to English, which needs a separate experiment (fake English) to disentangle.

> [!card]- What is fake English (K et al. 2020), how is it made, and why does it isolate the effect of subword overlap?
> - K et al. train their own bilingual BERT (**B-BERT**) on (fake) English plus one other language, so factors can be switched off one at a time.
> - Fake English shifts every English Unicode codepoint by a large constant, so no character overlaps with the other language: a bijective mapping.
> - It is English in every respect except its characters: same words, grammar, word order and frequencies. Shared vocabulary with the other language becomes exactly zero, so any drop in transfer measures the contribution of shared wordpieces.

> [!card]- What were the fake-English XNLI results of K et al. (2020), and what is the conclusion?
> - en-es 72.3 against enfake-es 70.9 (contribution 1.4); en-hi 60.1 against 59.6 (0.5); en-ru 66.4 against 65.7 (0.7).
> - B-BERT on English and fake English, fine-tuned on fake English: 78.0 on fake English, 77.5 on real English (0.5).
> - Removing all subword overlap costs only **0.5 to 1.4** points: shared wordpieces are not what makes multilingual BERT multilingual.

> [!card]- How did K et al. (2020) test the role of word order, and what did they find?
> - Randomly permute a fraction of words (0, 0.25, 0.5, 1.0) during **pre-training** only; no permutation in fine-tuning. Fine-tune on English, evaluate XNLI in the target language.
> - Spanish: 70.9 to 62.5 at full permutation (drop 8.4); Hindi: 59.6 to 43.1 (16.5); Russian: 65.7 to 53.6 (12.1).
> - Significant drop, but transfer remains reasonable, well over the 33.3% chance level. Hindi, SOV and furthest from English word order, suffers most.

> [!card]- What did K et al. find when the premise and hypothesis of XNLI were in different languages (B-BERT)?
> - Fake English on both sides: 78.5 to 79.3; target language on both sides: 59.6 to 70.9 (the zero-shot results).
> - Mixed: enfake-target 57.9 (es), 45.7 (hi), 51.1 (ru); target-enfake 61.1, 55.6, 57.9.
> - Mixed pairs score under **both** monolingual conditions; Hindi falls 13.9 points under its own zero-shot score. A fake-English hypothesis is consistently better than a fake-English premise.
> - A truly language-neutral space would make mixed pairs as easy as monolingual ones; they are not.

> [!card]- What did K et al. (2020) find when varying B-BERT's depth, and why does it matter for transfer?
> - Depth varied from 1 to 24 layers with the parameter count held roughly constant (132.78M to 139.33M), 12 attention heads.
> - Fake-English (in-language) XNLI saturates fast: 66.6 at depth 1, 76.9 at 4, about 79 from 6 on.
> - Zero-shot Russian keeps rising: 45.0 (1), 63.1 (6), 67.6 (24).
> - The gap $\Delta$ shrinks from 21.6 to 11.3. **Depth has the largest impact of the architecture factors** and buys cross-lingual transfer more than in-language performance: deeper networks learn more language-independent representations.

> [!card]- What do "within" and "across" mean in Rajaee and Monz's (2024) XNLI evaluation of mBERT, and what are the averages?
> - **Within** for language $L$: premise and hypothesis both in $L$ (diagonal of the premise by hypothesis matrix).
> - **Across** for $L$: the mean of the cells where exactly one side is in $L$ and the other in a different language. The definition is reconstructed from the accuracy matrix rather than stated explicitly:
> $$\text{across}(L) = \frac{1}{2(N-1)}\Big(\sum_{L' \ne L} A_{LL'} + \sum_{L' \ne L} A_{L'L}\Big)$$
> - mBERT: within 65.7, across 54.5 on average (11.2 lower); Swahili across 45.6.

> [!card]- State the heuristics hypothesis (Rajaee and Monz) and the SNLI word-overlap evidence behind it.
> - Hypothesis: **transfer of heuristics can contribute to cross-lingual generalisation**.
> - Rajaee et al. (2022) binned SNLI pairs by premise-hypothesis word overlap: full overlap is entailment 94.7% of the time (17,364 against 963); [0.8, 1.0) 58.2%; falling to 13.9% at (0, 0.2). So high overlap is a strong cue for entailment.
> - A model fine-tuned on such data can learn "high overlap means entailment"; that shortcut works within any single language but cannot fire when premise and hypothesis are in different languages. How much of the cross-lingual drop this explains is an interpretation, not a settled result.

> [!card]- Name four patterns in mBERT's full XNLI premise by hypothesis accuracy matrix.
> 1. The diagonal is the maximum of every **column**: for any hypothesis language a same-language premise is best. Not along rows: for German, Urdu, Hindi, Swahili and Thai premises an English hypothesis beats the in-language one (sw-en 55.0 against sw-sw 50.3).
> 2. An **English hypothesis helps**: English column averages 64.9, English row 57.7.
> 3. A **Swahili or Thai hypothesis** is near-hopeless: Swahili column 40 to 42.5, Thai 43.8 to 46.7.
> 4. **Related languages pair well**: ru-bg 62.0, bg-ru 64.9, es-fr 68.3, fr-es 69.3, hi-ur 55.6, ur-hi 53.5.

> [!card]- List the four changes from mBERT to XLM-R and the reason for each.
> - **Objective:** MLM only, NSP dropped, because NSP contributes little and sometimes hurts.
> - **Masking:** dynamic instead of static.
> - **Data:** CC-100, 2.5TB of filtered CommonCrawl in 100 languages, two orders of magnitude more than Wikipedia.
> - **Vocabulary:** 250k entries (double mBERT's 120k), Unigram LM via SentencePiece, for better fertility (fewer pieces per word, especially for languages starved in mBERT's vocabulary), at the cost of a bigger embedding matrix.
> - XLM-R is by Conneau et al. (2020), a RoBERTa-style extension of mBERT.

> [!card]- Distinguish static from dynamic masking. Why is dynamic masking better?
> - **Static** (BERT, mBERT): masking is done once at the data level during pre-processing; every epoch reuses the same masked version.
> - **Dynamic** (RoBERTa, XLM-R): a fresh mask is drawn each time a sequence is fed to the model.
> - With static masking a token not selected in pre-processing is never a prediction target however many epochs run; dynamic masking gives new training signal on every pass at no extra data cost.

> [!card]- Give the three GLUE-style example tasks (sentiment, Winograd, reading comprehension) and say what makes each hard.
> - **Sentiment:** *Skip the film and buy the Philip Glass soundtrack CD* is negative, though it contains no negative word.
> - **Winograd schema:** *The trophy doesn't fit into the brown suitcase because it is too large*: *it* = the trophy; change *large* to *small* and *it* = the suitcase. Same syntax, so it needs world knowledge.
> - **Reading comprehension:** question *At what pressure is water heated in the Rankine cycle?* over a paragraph; the answer is the span from word 46 to 47, *high pressure*. Nothing is generated.

> [!card]- How is BERT fine-tuned for SQuAD span prediction? Give the new parameters, the probabilities and the decision rule.
> - Input `[CLS] question [SEP] paragraph [SEP]`. The only new parameters: a start vector $S \in \mathbb{R}^H$ and an end vector $E \in \mathbb{R}^H$.
> $$P_{\text{start}}(i) = \frac{\exp(T_i \cdot S)}{\sum_k \exp(T_k \cdot S)} \qquad P_{\text{end}}(j) = \frac{\exp(T_j \cdot E)}{\sum_k \exp(T_k \cdot E)}$$
> $$\hat{(i,j)} = \arg\max_{i \le j}\; T_i \cdot S + T_j \cdot E$$
> - $T_i$: final-layer output at paragraph token $i$; $H$: hidden size (768 for base); $i \le j$ forbids spans that end before they start.

> [!card]- Give the steps of decoding a SQuAD answer span from fine-tuned BERT.
> 1. Run BERT on `[CLS] q [SEP] p [SEP]` to get final-layer outputs $T$.
> 2. For each paragraph position compute a start score $T_i \cdot S$ and an end score $T_i \cdot E$.
> 3. Over all pairs $i \le j$ (in practice capped at a maximum answer length) pick the pair maximising start score + end score.
> 4. Return paragraph tokens $i$ to $j$.

> [!card]- Give the average within and across scores for mBERT and XLM-R on XNLI and XSQuAD (Rajaee and Monz 2024), and the result to remember.
> - mBERT XNLI: 65.7 / 54.5 (gap 11.2). XLM-R XNLI: 74.2 / 64.8 (gap 9.4).
> - mBERT XSQuAD F1: 64.4 / 44.2 (gap 20.2). XLM-R XSQuAD: 72.0 / **36.8** (gap **35.2**).
> - XLM-R wins within on both tasks and across on XNLI, but is **worse across languages on QA** than mBERT (36.8 against 44.2) despite being 7.6 better within.

> [!card]- How do the XSQuAD context by question matrices of mBERT and XLM-R differ?
> - **mBERT** degrades smoothly: an English question works with every context (strongest column), a Thai question works with none (18.8 to 23.4 off-diagonal), Thai context is weak for every question.
> - **XLM-R**: strong diagonal and strong English-question column (58.2 to 75.0), almost everything else collapses (Arabic context with a non-English, non-Arabic question 14.6 to 36.7; Chinese context with Thai or Turkish question 17.6). Structure: own language, or English question.
> - In XLM-R an English context with a foreign question is much weaker than a foreign context with an English question (en-ar 38.1 against ar-en 59.2).
> - On XNLI, by contrast, XLM-R lifts the whole matrix (off-diagonal mean 65.1 against 54.5); the Swahili hypothesis column stays weakest (44.5 to 51.7).

> [!card]- Summarise the conclusions on what drives mBERT's cross-lingual transfer, and what could be missing.
> - **Shared (sub)word vocabulary** plays a role but is not the main cause (fake English costs 0.5 to 1.4 XNLI points; transfer works across scripts).
> - **Structure** (grammar, word order): permuting words hurts significantly without eliminating transfer; typologically closer languages transfer better.
> - **Architecture:** deeper networks learn more language-independent representations.
> - **No single factor explains transfer** (or the lack of it).
> - What is missing is left as an open question; the obvious candidates are what mBERT lacks: parallel data and an alignment loss pulling translation equivalents together, since neither mBERT nor XLM-R handles mixed-language inputs well.
