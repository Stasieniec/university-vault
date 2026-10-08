# Archived flashcards: MNLP-L07 - Crosslingual NLP

The long-form cards this note carried until 2026-10-08, when they were replaced by short single-fact cards. Kept for reference; not published.

Click a question to reveal its answer, or press **Study** to drill the whole set. Cards marked as exam questions are meant to be answered out loud or on paper first, then checked against the points listed.

> [!exam]- Why does multilingual pretraining (mBERT, XLM-R) not by itself make a model crosslingual, and what evidence shows that explicit crosslingual objectives fix it?
> Must hit:
> 1. **The mechanism.** Multilingual models train jointly on many languages with one model and one subword vocabulary, but MLM predicts a word only from **same-language context**. When predicting word $W$ in language $L$, unrelated context in $L'$ is of no use, so nothing in the objective links languages. Any alignment is an **emergent side effect**.
> 2. **Why it looked fine.** Standard zero-shot benchmarks (fine-tune on English, test on another language) keep all parts of the input in one language, and mBERT does reasonably there.
> 3. **The harder test.** Put premise and hypothesis (XNLI) or context and question (XSQuAD) in **different languages** and compare **within** (both parts in one language) with **across** (mixed).
> 4. **The numbers.** QA F1 within/across: mBERT 64.4/44.2, XLM-R 72.0/**36.8**, InfoXLM 73.8/64.5. XLM-R is better than mBERT in every single language but **worse across languages**.
> 5. **The fix.** InfoXLM adds TLM and XLCo on parallel data: within improves only +1.2 (XNLI) and +1.8 (QA) over XLM-R, across improves **+5.5 and +27.7**. The QA gap shrinks from 35.2 to 9.3.
>
> Losing marks: treating "multilingual" and "crosslingual" as the same thing, or citing only standard zero-shot scores as proof of crosslingual ability.

> [!exam]- You have English-only training data for NLI and need a system for 14 other languages. Compare translate-train, translate-test and zero-shot transfer, using the XLM results on XNLI.
> - **Translate-train:** machine-translate the English training set into each language and fine-tune on the translation. XLM (MLM+TLM) average **76.7**, the best of the three, but needs an MT system and a translated training set for every language.
> - **Translate-test:** machine-translate each test example into English and apply an English model. XLM (MLM+TLM) average **74.2**.
> - **Zero-shot crosslingual transfer:** fine-tune on English only, test directly on each language. XLM (MLM) **71.5**, XLM (MLM+TLM) **75.1**.
> - Key point: a single zero-shot model that never saw non-English NLI data **beats the translate-test pipeline** (75.1 against 74.2), and TLM is what gets it there (+3.6 over MLM alone).
> - Zero-shot XLM (MLM+TLM) also beats mBERT and LASER in every language where they are reported.
>
> Losing marks: mixing up which side gets translated (train set into the target language, or test set into English).

> [!exam]- Explain translation language modeling (TLM): its input, its loss, why it forces crosslingual alignment, and how it differs from MLM and CLM.
> - **XLM** (Conneau and Lample, 2019) has three objectives: **CLM** (predict the next word from a prefix), **MLM** (predict masked words from monolingual context, as in BERT), **TLM** (predict masked words in a **sentence concatenated with its translation**). TLM is always paired with MLM or CLM; NSP is dropped.
> - **Loss:** TLM is MLM applied to $[\mathbf{x}; \mathbf{y}]$:
> $$\mathcal{L}_{\text{TLM}} = -\sum_{i \in M_x} \log p_\theta(x_i \mid \mathbf{x}_{\setminus M_x}, \mathbf{y}_{\setminus M_y}) - \sum_{j \in M_y} \log p_\theta(y_j \mid \mathbf{x}_{\setminus M_x}, \mathbf{y}_{\setminus M_y})$$
> The loss is the same as MLM's; only the input changes.
> - **Why it aligns:** in *"the [MASK] [MASK] blue" / "[MASK] rideaux étaient [MASK]"*, the English context barely constrains *curtains*, but the unmasked French *rideaux* gives it away. The cheapest way to lower the loss is to attend across languages and learn *rideaux* ↔ *curtains*. Context in $L'$ becomes useful for predicting $W$ in $L$.
> - **Two input details:** position embeddings **restart at 0** in the second sentence (so position cannot separate the halves, and corresponding words get similar position signals); **language embeddings** (en, fr) mark which half is which.
> - **Evidence:** zero-shot XNLI average 71.5 (MLM) to 75.1 (MLM+TLM).

> [!exam]- How would you build a parallel corpus from the web, and how do LASER embeddings make sentence alignment possible?
> - **Sources first:** natural by-products (multilingual news such as Xinhua, the UN and EU, websites of multilingual countries such as Canada and Belgium) and **OPUS**, a large research collection of parallel corpora.
> - **Crawl pipeline:** (1) document alignment (find parallel documents), (2) sentence alignment inside them. Or mine sentences directly from raw crawls such as **Common Crawl**.
> - **Why sentence alignment is separate:** segments do not follow paragraph boundaries (in the NHK swine fever example, two English paragraphs map into one Chinese paragraph, which must be split at a sentence boundary).
> - **Measuring equivalence:** old methods use dictionary overlap and relative length; the modern method compares **LASER** sentence embeddings (Artetxe and Schwenk, 2019).
> - **LASER:** BPE embeddings, stacked BiLSTM encoder, max pooling into one sentence vector; an LSTM decoder translates from that vector alone and is told the output language by a language ID embedding. So the vector must encode meaning and has no reason to encode the input language. Translations end up as near neighbours.
> - **vecalign** uses LASER similarities to match sentences, with an efficient search that scales to massive data sets.

> [!exam]- Explain neural machine translation as conditional language modeling, from the RNN encoder-decoder to the Transformer encoder-decoder.
> - Seq2seq drops two assumptions of sequence labeling: that $x_t$ corresponds to $y_t$, and that $|X| = |Y|$. MT needs both dropped (*Hiermit hörte sie nicht auf* / *She did not stop with this*: different lengths, reordering, one-to-many, many-to-one).
> - It is **conditional language modeling**: $p(y_t \mid \mathbf{Y}_{<t}, \mathbf{X})$, with $\mathbf{X}$ the encoder output and $\mathbf{Y}_{<t}$ the decoder's prefix. The sequence probability is the product of these terms over $t$.
> - **Data:** source ids $\mathbf{z}$; decoder input $\mathbf{x}$ = `<s>` + sentence; target $\mathbf{y}$ = sentence + `</s>` ($\mathbf{x}$ shifted by one).
> - **RNN encoder-decoder** (Sutskever et al., 2014, LSTMs): $\mathbf{h}^{dec}_0 = \mathbf{h}^{enc}_n$. Everything about the source must pass through one fixed-size vector: an **information bottleneck**.
> - **Transformer:** each decoder layer has a target context layer (masked self-attention over the target prefix), a **source-target context layer** (attention over all top-layer encoder outputs), and a feed-forward layer, each with a residual connection. Every decoder position in every layer can read every source position, which removes the bottleneck.

> [!exam]- Describe parent-child transfer learning for low-resource NMT. What should be frozen, and which factors decide whether transfer pays off?
> 1. **Procedure** (Zoph et al., 2016): train a parent on a high-resource pair (French→English, 300M English tokens); copy all parameters into the child (Uzbek→English, 1.8M tokens); child source words take over rows of the parent's source embedding matrix; freeze the English embeddings; continue training with strong regularisation (dropout 0.5).
> 2. **Gains:** Hausa +4.5, Turkish +5.6, Uzbek +3.7, Urdu +8.6 BLEU; the smallest corpus (Urdu) gains most.
> 3. **Freezing:** train everything up to and including attention (Uz→En dev 15.0), but keep the **target embeddings frozen** (unfreezing them drops to 14.7, then 13.7). The general rule "freeze the decoder" overstates it: training the target RNN raises 11.8 to 14.2.
> 4. **Relatedness:** a Spanish child gets 31.0 with a French parent, 29.8 with German, 16.4 with none. But French' (scrambled vocabulary) still gains 13.3 to 20.0, so structure transfers and shared words are not the only factor.
> 5. **Shared vocabulary** (Kocmi and Bojar, 2018): unrelated parents (Czech, Russian) help English→Estonian as much as related Finnish.
> 6. **Direction:** transfer must flow from the larger corpus to the smaller; reversed, it helps little or hurts.
> 7. **What transfers** (Aji et al., 2020): inner layers carry most of the benefit; parent embeddings alone are worse than training from scratch.

> [!exam]- Describe BART: its architecture, its pretraining noise functions, which noise works best, and how it is fine-tuned for classification, span prediction and translation.
> - **Architecture:** bidirectional encoder (as BERT) plus autoregressive decoder (as GPT). The encoder reads a **corrupted** document, the decoder reconstructs the **original** through cross-attention. Suited to tasks needing both, such as translation and summarisation.
> - **Noise functions:** token masking, token deletion, text infilling (a span replaced by one `[MASK]`), sentence permutation, document rotation.
> - **Best:** **text infilling** (SQuAD 90.8, best XSum and ConvAI2 perplexity); deletion beats masking on all generation tasks; rotation and sentence shuffling alone are poor (SQuAD 77.2 and 85.4). Infilling plus shuffling gives the best CNN/DM perplexity (5.41).
> - **Classification:** same input to encoder and decoder; label predicted from the last decoder hidden state. **Span prediction (SQuAD):** label each token, predict start and end of the answer.
> - **MT:** a randomly initialised source encoder replaces BART's embedding layer; BART can be frozen or updated; the source vocabulary can differ from BART's. Ro→En: baseline 36.80, Fixed BART 36.29, **Tuned BART 37.96**.
> - Results: matches RoBERTa on SQuAD (94.6 F1 on 1.1), best on all ROUGE columns for CNN/DM and XSum.

> [!exam]- Why does crosslingual QA expose the weakness of purely multilingual models much more than XNLI does? Use the XLM-R and InfoXLM results.
> - **XNLI** compares two sentences; a coarse sentence-level gist in a shared space is often enough.
> - **Extractive QA** requires finding the exact answer span, which means matching the question's words to specific context words. With a Hindi question and Arabic context that is **word-level matching between two non-English languages**, never asked for by monolingual MLM, and exactly what TLM trains.
> - **XLM-R QA:** diagonal strong (63.7 to 84.2), English-question column 58.2 to 75.0, but most other off-diagonal cells collapse (Arabic context with non-English question 14.6 to 36.7; Chinese context 16.0 to 32.0). Across average 36.8, under mBERT's 44.2.
> - **InfoXLM QA:** every cell at least 51.7; Arabic and Chinese contexts with non-English questions now 51.7 to 63.8.
> - Gain of InfoXLM over XLM-R in the across score: **+5.5 on XNLI, +27.7 on QA**.

> [!exam]- Which factors predict whether crosslingual transfer will succeed? Support each with evidence.
> - **Language relatedness:** Nepali perplexity 157.2 alone, 140.1 with English, **115.6 with Hindi**; Spanish NMT child 31.0 with a French parent against 29.8 with German.
> - **Being close to English:** English dominates pretraining and fine-tuning data, so the English column is the brightest in every XNLI heatmap and English cells are the darkest in the WikiMatrix BLEU grid; directions between two non-English languages are mostly 5 to 25 BLEU.
> - **Representation of the language and its script:** Swahili hypotheses leave mBERT barely over chance (40.2 to 42.5); mBERT cannot handle Thai questions in QA (18.8 to 23.4), while XLM-R has no such Thai problem.
> - **An explicit crosslingual training signal:** TLM and XLCo on parallel data close most of the within/across gap (InfoXLM).
> - **Amount and direction of data in NMT transfer:** transfer helps when it flows from a large parent to a small child; reversed it helps little or hurts. With a shared vocabulary and a large parent, relatedness matters less (Kocmi and Bojar).
> - **Which parameters transfer:** inner layers carry most of the benefit (Aji et al.).

> [!card]- In which four senses are models such as mBERT and XLM-R "multilingual", and which one is the weakness for crosslingual ability?
> 1. They train jointly on multiple languages.
> 2. They use one model for all languages.
> 3. They use one (subword) vocabulary for all languages.
> 4. Their predictions are based on context **within the same language** (MLM).
>
> The fourth is the weakness: predicting a masked German word, every visible token is German, so nothing rewards knowing its English translation.

> [!card]- Define crosslingual knowledge transfer and zero-shot crosslingual transfer, and say why this is the common scenario in practice.
> - **Crosslingual knowledge transfer:** given fine-tuning data in language A for task X, how well does the model generalise to test data for task X in language B?
> - **Zero-shot crosslingual transfer:** the fine-tuning and test languages differ and no task data in B was seen at all.
> - Common because fine-tuning requires annotated data, which is scarce outside high-resource languages. It is a crosslingual capability: the model must map what it learned in A onto B.

> [!card]- Can crosslingual capabilities emerge without any crosslingual training signal? Give the evidence on each side.
> - **For:** earlier zero-shot results; mBERT fine-tuned on English works to a degree on other languages.
> - **Against:** crosslingual tasks where the input itself mixes languages (premise in one, hypothesis in another) make purely multilingual models degrade sharply.
> - Verdict from the within/across results: much of the capability does **not** emerge without a crosslingual signal, and adding one fixes most of it.

> [!card]- Why does training on many languages not link them, and what is the fix?
> - Training on multiple languages makes a model multilingual, but nothing explicitly links information across languages. When predicting word $W$ in language $L$, unrelated context in $L'$ gives no benefit: the languages share parameters but never share **context**.
> - Fix: **context alignment across languages**. Give the model text in two languages known to correspond (parallel or comparable data), so context in $L'$ actually helps predict $W$ in $L$.

> [!card]- Distinguish parallel data from comparable data, with examples and degree of parallelism.
> - **Parallel:** sentence pairs in different languages that are **meaning equivalent** (translations). Fully parallel by construction. Used by data-driven MT for decades.
> - **Comparable:** sentence or document pairs **on the same topic**. Not fully parallel; parallelism is a **sliding scale**. Examples: news articles on the same event, Wikipedia articles on the same entity in different languages.

> [!card]- What does a Chinese-English parallel corpus excerpt with 新加坡 / *Singapore* highlighted in every row illustrate?
> Once sentence pairs are aligned, recurring co-occurrences (新加坡 always paired with *Singapore*) let a model learn that the two correspond without anyone writing a dictionary. Statistical MT used this historically; TLM exploits the same signal.

> [!card]- Where does parallel data come from naturally, what is OPUS, and what are the two ways to crawl your own?
> - **Natural by-products:** multilingual news (Xinhua); international organisations (UN, EU translate every official document); websites and documents from countries with two or more languages (Canada, Belgium, US).
> - **OPUS:** a website with a very large collection of parallel corpora for research, covering many languages. The first place to look.
> - **Crawling:** (1) find parallel documents (document aligning), then (2) parallel sentences within them (sentence aligning). Or directly align sentences from vast raw web crawls such as Common Crawl.

> [!card]- What does the NHK World swine fever example (English and Chinese news pages) show about parallel documents and sentences?
> - The two pages are a **parallel document pair**: same story, same photo, published a day apart (time zone).
> - Inside, segments align as **parallel sentences**, but not paragraph to paragraph: English paragraphs 2 and 3 both map into Chinese paragraph 2, which must be split at a sentence boundary.
> - So sentence alignment is a separate step after document alignment and must allow alignments other than one paragraph to one paragraph.

> [!card]- How can degrees of meaning equivalence between sentences in two languages be measured? Give the old-fashioned and the recent approach.
> - **Old-fashioned:** **dictionary overlap** (how many words in sentence 1 have a dictionary translation in sentence 2) and **relative length distribution** (translations have predictable length ratios).
> - **Recent:** compute and compare **sentence embeddings**, specifically **LASER** (Artetxe and Schwenk, 2019): embed all sentences in one shared space, where translations are nearest neighbours.

> [!card]- Describe the LASER encoder, step by step.
> 1. Input tokens $x_1, x_2, \dots, \texttt{</s>}$ are looked up in a **BPE embedding** table shared by all languages.
> 2. They pass through a **stack of BiLSTM layers**.
> 3. The top layer's hidden states are **max-pooled** over time (element-wise maximum across positions).
> 4. The result is one fixed-size vector, the **sentence embedding**.
>
> After training on translation, only this encoder is kept.

> [!card]- In LASER, how does the sentence embedding reach the decoder, and what else does the decoder receive at each step?
> - Through a linear map $W$ that **initialises the decoder LSTM**.
> - By being **concatenated to the input at every decoder step**, together with the BPE embedding of the previous output token ($\texttt{<s>}, y_1, \dots$) and a **language ID embedding** $L_{id}$ saying which language to produce.
> - Each output step ends in a softmax over the vocabulary.

> [!card]- Why does LASER's training produce language-independent sentence vectors?
> The decoder sees nothing of the source except the single sentence vector, and it is told the output language by $L_{id}$, not by the encoder. So the encoder has no reason to encode the input language and every reason to encode only the meaning, which is all the decoder needs to translate into any target language. After training the decoder is discarded, and the encoder maps all training languages into one space where translations land close together.

> [!card]- What is vecalign?
> A sentence alignment approach that (1) uses **LASER** to match sentences, scoring a source and target sentence by the similarity of their LASER embeddings, and (2) uses an **efficient search that scales to massive data sets**.

> [!card]- What problem with mBERT motivated XLM, and what does XLM do about it?
> - mBERT **never sees an explicit translation pair** during pretraining; any crosslingual alignment is an **emergent side effect**.
> - For many language pairs at least some parallel text exists (OPUS, self-crawled corpora).
> - **XLM** (Conneau and Lample, 2019) uses parallel data for an **explicit crosslingual loss** (TLM).

> [!card]- List XLM's three pretraining objectives (input and prediction for each), how they are combined, and where its parallel data comes from.
> - **CLM** (causal LM): prefix in, next word out. Not crosslingual.
> - **MLM** (masked LM): monolingual context with masked words, predict them as in BERT. Not crosslingual.
> - **TLM** (translation LM): a sentence and its translation, predict masked words in either language. Crosslingual.
> - Next sentence prediction is **dropped**. TLM is always combined with a monolingual objective: **MLM + TLM** or **CLM + TLM**.
> - Parallel corpora from **OPUS** (e.g. MultiUN, OpenSubtitles, EUbookshop).

> [!card]- Write the CLM and MLM losses and define every term.
> $$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{n} \log p_\theta(x_t \mid x_1, \dots, x_{t-1})$$
> $$\mathcal{L}_{\text{MLM}} = -\sum_{i \in M} \log p_\theta(x_i \mid \mathbf{x}_{\setminus M})$$
> - $\mathbf{x} = (x_1, \dots, x_n)$ is the token sequence.
> - $M \subseteq \{1, \dots, n\}$ is the set of masked positions; $\mathbf{x}_{\setminus M}$ is the sequence with those positions replaced by `[MASK]`.
> - $p_\theta$ is the Transformer's softmax output over the shared vocabulary.

> [!card]- Write the TLM loss for a parallel pair and explain how it relates to MLM.
> $$\mathcal{L}_{\text{TLM}} = -\sum_{i \in M_x} \log p_\theta(x_i \mid \mathbf{x}_{\setminus M_x}, \mathbf{y}_{\setminus M_y}) - \sum_{j \in M_y} \log p_\theta(y_j \mid \mathbf{x}_{\setminus M_x}, \mathbf{y}_{\setminus M_y})$$
> - $\mathbf{x}$ is a sentence in language 1, $\mathbf{y}$ its translation; $M_x$, $M_y$ are the masked positions in each.
> - Every masked word, in either language, is predicted from both (partially masked) sentences.
> - TLM is literally MLM on the concatenation $[\mathbf{x}; \mathbf{y}]$: only the input changes.

> [!card]- In XLM's MLM training example, what does the input stream look like, and what three embeddings make up each input position?
> - A continuous stream of sentences separated by `[/s]`, cut into a fixed-length window: *"[/s] take a seat [/s] have a drink [/s] now relax and"*, with *take*, a `[/s]`, *drink* and *now* masked as targets. A sentence separator can itself be masked.
> - Each position is the **sum of token, position and language embeddings**; in MLM all language embeddings are the same (en).

> [!card]- In XLM's TLM example (*the curtains were blue* / *les rideaux étaient bleus*), which words are masked, and what are the two input details that make crosslingual prediction work?
> - Masked targets: *curtains*, *were* (English) and *les*, *bleus* (French).
> - *curtains* is hard from *"the ___ ___ blue"* but easy from French *rideaux*; *les* and *bleus* can be read off English *the* and *blue*.
> - **Positions restart at 0** for the French sentence (both halves use positions 0 to 5), so position cannot tell the halves apart and roughly corresponding words get similar position signals.
> - **Language embeddings** (en, fr) mark which half is which language, since positions no longer do.

> [!card]- What does XNLI measure, how is it scored, and what is the $\Delta$ column in the XLM results?
> - Crosslingual natural language inference: given a premise and a hypothesis, a **three-way classification** (chance 33.3%), evaluated in **15 languages**.
> - Metric: **accuracy**.
> - $\Delta$: the plain average accuracy over the 15 languages.

> [!card]- Give XLM's average XNLI accuracy in each setting, and the main conclusions.
> - Translate-train, XLM (MLM+TLM): **76.7** (best overall).
> - Translate-test, XLM (MLM+TLM): **74.2**.
> - Zero-shot, XLM (MLM): **71.5**; zero-shot XLM (MLM+TLM): **75.1**; LASER (Artetxe and Schwenk) 70.2; Conneau et al. (2018b) 65.6. mBERT was reported for only six languages.
> - Conclusions: TLM is worth **+3.6** on average and every language gains; zero-shot XLM beats translate-test; translate-train is still best but needs MT for every language; zero-shot XLM beats mBERT and LASER in every reported column.

> [!card]- Which languages gain most and least from adding TLM to MLM in zero-shot XNLI, and what pattern does that show?
> - **Largest gains:** vi 71.2 → 76.1 (+4.9), tr 67.8 → 72.5 (+4.7), ar 68.5 → 73.1 and zh 71.9 → 76.5 (+4.6 each).
> - **Smallest:** English (+1.8), fr and ru (+2.2 each).
> - Pattern: TLM helps most for languages distant from English in script or structure.

> [!card]- Worked example: compute the zero-shot XNLI average of XLM (MLM+TLM) from its 15 per-language scores.
> Scores: 85.0, 78.7, 78.9, 77.8, 76.6, 77.4, 75.3, 72.5, 73.1, 76.1, 73.2, 76.5, 69.6, 68.4, 67.3.
> $$\frac{1126.4}{15} = 75.09 \approx 75.1$$

> [!card]- What do XLM's Nepali language modeling results show? Give the perplexities.
> Nepali perplexity (lower is better): Nepali alone **157.2**; + English **140.1**; + Hindi **115.6**; + English + Hindi **109.3**.
> Any second language helps, a **related** language helps far more (Hindi: same Devanagari script, closely related; cuts 41.6 points against English's 17.1), and both together are best. Transfer works best between related languages.

> [!card]- How do XLM's word embeddings compare with MUSE and Concat on crosslingual word alignment, and what are the baselines?
> - Baselines use **fastText** embeddings. **Concat** = joint training of embeddings over all languages; **MUSE** = the mapping-based method for crosslingual static embeddings. XLM's word embeddings are its input embedding table.
> - Cosine similarity (higher better): MUSE 0.38, Concat 0.36, **XLM 0.55**. L2 distance (lower better): 5.13, 4.89, **2.64**. SemEval'17 (higher better): 0.65, 0.52, **0.69**.
> - XLM's embeddings are better aligned on all three, though never trained to align word embeddings directly.

> [!card]- What is InfoXLM built on, and what are its three objectives?
> **InfoXLM** (Chi et al., 2021) builds on **XLM-R** (a purely multilingual masked LM) and adds XLM's parallel-data ideas.
> 1. **MMLM** (multilingual masked LM): described as similar to MLM but using negative sampling instead of directly optimising the ground-truth loss.
> 2. **TLM**, as in XLM.
> 3. **XLCo** (crosslingual contrastive learning): take the `[CLS]` representations of the two sentences of a parallel pair, classify whether the pair is parallel, with **random sentences as negatives**.

> [!card]- What does the "negative sampling" in InfoXLM's MMLM description actually mean, according to the InfoXLM paper?
> MMLM is ordinary masked LM on monolingual text in many languages, the same loss as XLM-R. The paper **interprets** the softmax cross-entropy over the vocabulary as a contrastive (InfoNCE) loss: the correct token is the positive, every other vocabulary entry acts as a negative. Nothing extra is sampled. (This reading comes from the paper; the short description of MMLM suggests a replaced, sampled loss.)

> [!card]- Write the XLCo contrastive loss and define every term.
> $$\mathcal{L}_{\text{XLCo}} = -\log \frac{\exp(f(\mathbf{x})^\top f(\mathbf{y}))}{\exp(f(\mathbf{x})^\top f(\mathbf{y})) + \sum_{\mathbf{y}' \in \mathcal{N}} \exp(f(\mathbf{x})^\top f(\mathbf{y}'))}$$
> - $(\mathbf{x}, \mathbf{y})$: a parallel pair (the positive). $f(\cdot)$: the `[CLS]` representation. $\mathcal{N}$: negatives, random sentences that are not translations of $\mathbf{x}$.
> - The fraction is a softmax over "which candidate is the translation of $\mathbf{x}$?"; minimising it pulls translations together and pushes non-translations apart.
> - This is the standard InfoNCE formalisation of the description; the exact formula is not given in the original source.

> [!card]- Contrast token-level and sentence-level crosslingual alignment, and say which of TLM, XLCo and LASER does which, and how.
> - **Token level: TLM.** A masked word is predicted from its translation's words.
> - **Sentence level: XLCo.** A sentence's whole representation and its translation's must be closer to each other than to anything else (contrastive loss inside masked-LM pretraining).
> - **Sentence level: LASER.** Gets sentence alignment from a translation decoder that sees only the sentence vector.

> [!card]- Compare XLM and InfoXLM on starting point, data, vocabulary, language embeddings, objectives and type of crosslingual signal.
> - **Start:** XLM from scratch; InfoXLM initialised from XLM-R, then further pretrained (150K steps base, 200K large).
> - **Monolingual data:** Wikipedia; CC-100 rebuilt (94 languages).
> - **Parallel data:** XLM, the 15 XNLI languages; InfoXLM, 14 English-centric pairs, about 42 GB (MultiUN, IIT Bombay, OPUS, WikiMatrix).
> - **Vocabulary:** shared BPE; XLM-R's 250k SentencePiece.
> - **Language embeddings:** yes; no.
> - **Objectives:** MLM (or CLM) + TLM; MMLM + TLM + XLCo, equally weighted: $\mathcal{L}_{\text{InfoXLM}} = \mathcal{L}_{\text{MMLM}} + \mathcal{L}_{\text{TLM}} + \mathcal{L}_{\text{XLCo}}$.
> - **Signal:** token level only; token and sentence level.

> [!card]- What does "English-centric" parallel data mean for InfoXLM, how does WikiMatrix connect to LASER, and why does InfoXLM have no language embeddings?
> - English-centric: every parallel pair has English on one side (en-fr, en-de, ...); there is no direct fr-de data.
> - WikiMatrix is parallel sentences mined from Wikipedia **with LASER** (Schwenk et al., 2019), so the LASER mining pipeline feeds InfoXLM.
> - InfoXLM starts from XLM-R's weights and cannot change the architecture without losing them, so it inherits the 250k SentencePiece vocabulary and the lack of language embeddings. TLM must then tell the two halves apart from the tokens alone.

> [!card]- Define the within and across scores for a language $\ell$ in mixed-language XNLI and QA evaluation, with the formula.
> - **Within**$(\ell)$: score with both input parts in $\ell$, the diagonal cell $S(\ell, \ell)$.
> - **Across**$(\ell)$: mean over all mixed combinations involving $\ell$ in **either** position (row and column without the diagonal):
> $$\text{across}(\ell) = \frac{1}{2(K-1)} \Big( \sum_{\ell' \neq \ell} S(\ell, \ell') + \sum_{\ell' \neq \ell} S(\ell', \ell) \Big)$$
> - $K$ = number of languages (15 for XNLI, 11 for QA); $S(a, b)$ = score with the first part (premise or context) in $a$ and the second (hypothesis or question) in $b$.
> - The definition is not stated in the source; it is recovered from the heatmaps, which it reproduces (row only or column only does not).

> [!card]- Worked example: mBERT's XNLI row for English (premise English, 14 other hypothesis languages) sums to 808.5 and its English column to 908.4. Compute across(en) and interpret it.
> $$\text{across}(\text{en}) = \frac{808.5 + 908.4}{2 \cdot 14} = \frac{1716.9}{28} = 61.3$$
> Within(en) = 81.5, so mBERT loses about 20 points once one sentence is not English. Asymmetry: column mean 64.9 against row mean 57.7, so an English **hypothesis** with a foreign premise is easier than the reverse.

> [!card]- Give the average within, across and gap for mBERT, XLM-R and InfoXLM on XNLI (accuracy) and QA (F1).
> - **mBERT:** XNLI 65.7 / 54.5, gap 11.2; QA 64.4 / 44.2, gap 20.2.
> - **XLM-R:** XNLI 74.2 / 64.8, gap 9.4; QA 72.0 / 36.8, gap **35.2**.
> - **InfoXLM:** XNLI 75.4 / 70.3, gap **5.1**; QA 73.8 / 64.5, gap **9.3**.

> [!card]- What is paradoxical about XLM-R against mBERT in the within/across evaluation?
> XLM-R is a much better multilingual model (within +8.5 on XNLI, +7.6 on QA), but on mixed-language QA it is **worse than mBERT**: across 36.8 against 44.2. Being better at each language separately did not make it better at relating two languages to each other.

> [!card]- In the XNLI heatmaps, which hypothesis language is easiest and which hardest, and why? Give the numbers.
> - **English column brightest** in every model (premise in any language, English hypothesis): mean 64.9 (mBERT), 74.1 (XLM-R), 77.8 (InfoXLM). English dominates pretraining and fine-tuning data, so every language is best aligned to it.
> - **Swahili column darkest:** mBERT 40.2 to 42.5 whatever the premise (barely over chance, 33.3%), XLM-R 44.5 to 51.7, InfoXLM 56.0 to 64.9.

> [!card]- Name three further patterns in the XNLI premise/hypothesis heatmaps.
> - **Not symmetric:** swapping which language holds premise and hypothesis changes the score. mBERT with a Swahili premise (row mean 49.5) does better than with a Swahili hypothesis (column mean 41.6).
> - **Mixing two related high-resource languages costs little:** mBERT (fr premise, en hypothesis) 73.4 against fr-fr 73.5; XLM-R (fr, en) 78.8, higher than fr-fr 78.3.
> - **InfoXLM is uniformly brighter off the diagonal:** its worst off-diagonal cell (56.0) beats every cell in XLM-R's Swahili column (at most 51.7); its best off-diagonal cell is (es, en) 81.6.

> [!card]- What does mBERT's crosslingual QA heatmap show for Thai?
> mBERT cannot handle Thai questions: the Thai column is 18.8 to 23.4 for every non-Thai context, and even Thai-Thai is only 40.0. XLM-R does not share the problem (Thai-Thai 66.5). A plausible reason is poor coverage of Thai script in mBERT's WordPiece vocabulary.

> [!card]- Describe XLM-R's crosslingual QA heatmap and InfoXLM's, cell by cell pattern.
> - **XLM-R:** works across languages only when the **question is English** (en column 58.2 to 75.0). With an Arabic context, non-English questions score 14.6 to 36.7; with Chinese, 16.0 to 32.0. Diagonal strong (63.7 to 84.2).
> - **InfoXLM:** fills the matrix in. Every cell at least 51.7 (zh context, hi question); English context gives 70.6 to 79.3 for any question language; the Arabic and Chinese cells XLM-R left at 15 to 37 are now 51.7 to 63.8.

> [!card]- Give the three paradigms of machine translation with their periods, and explain the overlap.
> - **1950s to 1990s:** rule-based, symbolic.
> - **1990s to 2016:** statistical, data-driven.
> - **2014 to now:** neural, deep learning, data-driven.
>
> The 2014 to 2016 overlap is deliberate: neural MT appeared in research in 2014, statistical systems stayed in production until around 2016. MT has been active AI research since the start and illustrates AI's paradigm shifts.

> [!card]- How does sequence-to-sequence modeling differ from sequence labeling, and how is it written as conditional language modeling?
> - It does **not** assume an isomorphic relationship between $x_t$ and $y_t$, and does **not** assume $|X| = |Y|$. It models the complex mapping between $X$ and $Y$.
> $$p(y_t \mid \mathbf{Y}_{<t}, \mathbf{X})$$
> - $\mathbf{X}$: the encoder's representation of the input. $\mathbf{Y}_{<t}$: the decoder's representation of the output before $t$. $y_t$: the next output token.
> - By the chain rule, $p(\mathbf{Y} \mid \mathbf{X}) = \prod_{t=1}^{|\mathbf{Y}|} p(y_t \mid \mathbf{Y}_{<t}, \mathbf{X})$; training minimises $-\log p(\mathbf{Y} \mid \mathbf{X})$ over parallel pairs.

> [!card]- What was Sutskever et al.'s (2014) neural MT model, and how did it connect encoder and decoder?
> MT as sequence to sequence with an **LSTM encoder** and an **LSTM decoder** (two-layer stacks in the figure). The encoder reads $A\,B\,C$; the decoder starts from a start symbol, emits $W$, feeds it back, emits $X$, and so on until `<eos>`. Connection: the decoder LSTM is **initialised with the last state of the encoder LSTM**.

> [!card]- Using *Hiermit hörte sie nicht auf.* → *She did not stop with this.*, name the four complex mappings that make MT hard.
> 1. **Different lengths** (6 tokens against 7).
> 2. **Word order differences** (*hörte ... auf* brackets the clause; *sie* moves from position 3 to 1).
> 3. **One-to-many:** *Hiermit* → *with this*, *nicht* → *did not*.
> 4. **Many-to-one:** *hörte ... auf* → *stop* (the separable verb *aufhören*).
>
> In sequence labeling every input has one output directly over it; in MT alignment links cross.

> [!card]- How are a parallel sentence pair $(f, e)$ represented as vectors $\mathbf{z}$, $\mathbf{x}$, $\mathbf{y}$ for NMT training?
> - Both are tokenized ($n$ and $m$ tokens); each token is its index in vocabulary $V_f$ or $V_e$.
> - Foreign sentence: $\mathbf{z} \in \mathbb{N}^n$.
> - Target as two vectors: $\mathbf{x}_1 = \text{id}_e(\texttt{<s>})$, $\mathbf{y}_i = \mathbf{x}_{i+1}$, last $\mathbf{y} = \text{id}_e(\texttt{</s>})$.
> - $\mathbf{x}$ (`<s>` She did not stop with this .) is what the decoder **reads**; $\mathbf{y}$ (She did not stop with this . `</s>`) is what it must **predict**. At step $i$ it has read the start symbol and first $i-1$ words and predicts word $i$.
> - Strictly, with $m$ English tokens $\mathbf{x}$ and $\mathbf{y}$ have $m+1$ entries (7 tokens, 8 entries).

> [!card]- How does the RNN encoder-decoder connect its two halves, and what problem does that cause?
> - Encoder represents $\mathbf{z}$ as a whole; decoder reads $\mathbf{x}$ token by token and learns to predict $\mathbf{y}$.
> - Connection: $\mathbf{h}^{dec}_0 = \mathbf{h}^{enc}_n$ (decoder's initial state = encoder's state after the last source token).
> - Problem: an **encoder-decoder information bottleneck**. Everything about the source must fit through one fixed-size vector, however long the sentence. Attention removes it by letting every decoder step look at all encoder states.

> [!card]- Give the steps for training an RNN encoder-decoder on one sentence pair.
> 1. $h \leftarrow 0$.
> 2. For $t = 1..n$: $h \leftarrow \text{EncoderRNN}(h, \text{Emb}_f[z_t])$.
> 3. $s \leftarrow h$ (the bottleneck, $\mathbf{h}^{dec}_0 = \mathbf{h}^{enc}_n$); loss $\leftarrow 0$.
> 4. For $i = 1..m$: $s \leftarrow \text{DecoderRNN}(s, \text{Emb}_e[x_i])$ with the **gold** previous word; $p \leftarrow \text{softmax}(W_{out} s)$ over $V_e$; loss $\leftarrow$ loss $- \log p[y_i]$.
> 5. Update parameters by gradient descent on the loss.
>
> At test time there is no gold $\mathbf{x}$: start from `<s>` and feed each predicted word back in until `</s>`.

> [!card]- Describe one decoder layer of the Transformer encoder-decoder, component by component.
> 1. Each target token is turned into **two vectors** (drawn cyan and green and unlabelled; their role fits the keys and values of attention).
> 2. **Target context layer:** masked self-attention; a position attends only to itself and earlier positions. Residual "+".
> 3. **Source-target context layer:** attends over **all** top-layer encoder outputs (cross-attention). Residual "+". This replaces the RNN's single-vector bottleneck.
> 4. **Feed-forward layer**, per position. Residual "+".
> 5. Output split again into vector pairs for the next layer; after layer $n$, the output layer's softmax over the target vocabulary.

> [!card]- What does NMT quality depend on, and what are two typical low-resource NMT problems?
> - Quality depends on: **the amount of data**, **the amount of variation within the data**, and **the relevance of the data for the actual task**.
> - Low-resource problems (some of these conditions unmet): **domain adaptation** (data in the wrong domain) and **NMT for low-resource language pairs** (little data at all).

> [!card]- How far are we from universal machine translation (Schwenk et al., 2019), and what does the BLEU grid behind the claim show?
> - **86% of all language directions are of poor quality.** Core problem: limited parallel training data for most directions, and current MT models do not generalise very well beyond the training data, and not at all beyond specific language directions.
> - Grid (26 languages, from the WikiMatrix paper): cells involving English darkest (e.g. en with da, fr, no 41.2); non-English directions mostly 5 to 25 BLEU, except related or high-resource pairs (da and no 27.5 and 30.4); Korean and Japanese weakest (ko column 1.2 to 4.1); many cells empty for lack of mined data.

> [!card]- What does Zoph et al. (2016) show about how data hungry NMT is? Give the numbers.
> BLEU into English, syntax-based SMT against NMT trained on child data only:
> - Hausa (1.0M tokens): 23.7 / 16.8
> - Turkish (1.4M): 20.4 / 11.4
> - Uzbek (1.8M): 17.9 / 10.7
> - Urdu (0.2M): 17.9 / 5.2
>
> With 0.2M to 1.8M tokens NMT loses by 6.9 to 12.7 BLEU; the smallest corpus has the worst gap.

> [!card]- What do Koehn and Knowles's (2017) learning curves show about NMT against phrase-based MT?
> - Corpus size from about $4 \times 10^5$ to $4 \times 10^8$ words, doubling each step.
> - **Neural starts catastrophically low:** 1.6 BLEU at about 0.4M words, against 16.4 phrase-based and 21.8 phrase-based with a big LM.
> - **Neural improves faster:** overtakes phrase-based between about $1.2 \times 10^7$ and $2.4 \times 10^7$ words (22.4 < 23.5, then 25.7 > 24.7), and phrase-based with big LM around $10^8$ to $2 \times 10^8$ words. At the largest size: 31.1 against 28.6 and 30.4.
> - Under tens of millions of words statistical MT wins, which is where most language pairs are.

> [!card]- What is transfer learning for NMT, and what four conditions make it work best?
> - Learning tens of millions of parameters from small data is hard, so initialise with "reasonable" parameters. Basically fine-tuning: train on a large-data pair (**parent**), continue training on the low-resource pair (**child**).
> - Works best if: (1) the **target language** of parent and child is identical; (2) the **source languages are related**; (3) decoder parameters are frozen (overstated: only the target embeddings should be); (4) child and parent **share the same vocabulary**.

> [!card]- Give the steps of parent-child transfer learning for low-resource NMT, with Zoph et al.'s concrete setup.
> 1. Train the parent from random initialisation on the large corpus (French→English, 300M English tokens, WMT'15, 5 epochs, about 26 BLEU dev).
> 2. Copy every parameter into the child.
> 3. Map each child source word (Uzbek) to a row of the parent's source embedding matrix; that row is its initial embedding.
> 4. Freeze the target-language (English) embeddings.
> 5. Continue training on the child corpus (Uzbek→English, 1.8M English tokens) with strong regularisation (dropout 0.5), updating only non-frozen parameters.

> [!card]- In parent-child NMT transfer, what is shared, what is not, and why are the target-language parameters frozen?
> - **Not shared:** the source vocabulary and source embeddings. Child rows are initialised from parent rows, but nothing about that French word means anything for the Uzbek word.
> - **Shared:** the target language and all other parameters (target embeddings, attention, ...).
> - **Frozen target parameters:** English is the same in parent and child, and the parent saw 300M English tokens against 1.8M, so its English embeddings are far better than the child data could make them.

> [!card]- Give Zoph et al.'s (2016) transfer results per child language, and what they show.
> NMT / Xfer / Final / SBMT:
> - Hausa 16.8 / 21.3 / **24.0** / 23.7
> - Turkish 11.4 / 17.0 / 18.7 / **20.4**
> - Uzbek 10.7 / 14.4 / 16.8 / **17.9**
> - Urdu 5.2 / 13.8 / 14.5 / **17.9**
>
> Transfer alone: +4.5, +5.6, +3.7, +8.6 (Urdu nearly triples). With further improvements (Final, whose exact definition is not given in the source), NMT overtakes SBMT for Hausa and is 1.1 to 3.4 BLEU short for the others.

> [!card]- Give the Uzbek→English freezing experiment results and the rule they support.
> Dev BLEU as parameter groups are unfrozen one by one: none **0.0**; + source embeddings **7.7**; + source RNN **11.8**; + target RNN **14.2**; + attention **15.0**; + target input embeddings **14.7**; + target output embeddings **13.7**.
> - With nothing trained the output is garbage: source embeddings must be retrained (biggest jump).
> - Training helps up to and including attention.
> - Unfreezing target embeddings hurts (learned from 300M tokens, overfit on 1.8M).
> - Rule: **freeze the target embeddings, train everything else**. The general claim "decoder parameters are frozen" is contradicted by the target RNN's +2.4.

> [!card]- What do Zoph et al.'s relatedness experiments, including French', show about what NMT transfer transfers?
> - Spanish→English child: no parent 16.4, German parent 29.8, **French parent 31.0**. A related parent helps more.
> - **French'** = French with random vocabulary reshuffling (consistent but arbitrary word forms; same grammar and word order, no shared surface forms). French parent lifts it from 13.3 to **20.0**: structure (word order, syntax, attention shape) transfers, not only shared words.
> - Uzbek with a French parent (unrelated): 15.0, the smallest benefit.

> [!card]- What does Kocmi and Bojar (2018) find about NMT transfer with a shared parent-child vocabulary?
> - Notation "enFI - enET": parent English→Finnish, child English→Estonian.
> - Every transfer from a larger parent helps, by 1.6 to 3.4 BLEU.
> - **Unrelated parents help as much as related ones:** English→Estonian 19.74 (Finnish parent), 20.41 (Czech), 20.09 (Russian), against 17.03 child-only.
> - "Only parent" (parent applied directly) is near zero except Czech/Slovak (up to 11.62), which are mutually intelligible.
> - **Reversed direction** (smaller parent, larger child) helps little (enET - enFI +0.57) or hurts (ETen - FIen 23.95 against 24.40; SKen - CSen 28.20 against 29.61). Transfer must flow from the larger corpus to the smaller.

> [!card]- What do Aji et al. (2020) find about which parts of a parent NMT model carry the transfer benefit?
> Children My, Id, Tr → English; embeddings and inner layers each transferred (Y) or randomly initialised (N). Reported averages:
> - Y/Y **21.7** (best); N/Y (inner only) 18.3; Y/N (embeddings only) 13.7; N/N (scratch) 14.5.
> - Inner layers carry most of the benefit; **embeddings alone are worse than nothing** (no inner layers to interpret them); embeddings on top of inner layers add more, so the parts work together.
> - Burmese shows it most: 4.0 from scratch, 17.8 with full transfer.

> [!card]- What does Aji et al.'s comparison of a German→English and an English→German parent suggest about the "same target language" condition?
> With full transfer into X→English children, the En→De parent (English on the **source** side) transfers about as well as the De→En parent (English on the target side): mean 21.7 against 21.8. This sits awkwardly with the rule that parent and child should share the target language.

> [!card]- Contrast BERT, GPT and BART on architecture and pretraining task.
> - **BERT:** encoder only, bidirectional; masked tokens (A _ C _ E → B, D) predicted independently.
> - **GPT:** decoder only, autoregressive; position $t$ sees only earlier positions and predicts the next token.
> - **BART:** bidirectional encoder reads a **corrupted** document, autoregressive decoder reconstructs the **original**, attending to the encoder through cross-attention. A denoising autoencoder; covers tasks needing both (translation, summarisation).

> [!card]- Give BART's five noise functions with what each does to `A B C . D E .`
> - **Token masking:** `A _ C . _ E .` (B, D replaced by `[MASK]`; predict them).
> - **Token deletion:** `A . C . E .` (B, D removed; predict them and where they were).
> - **Text infilling:** `A _ . D _ E .` (span B C replaced by a **single** `[MASK]`; a `[MASK]` also inserted between D and E where nothing was removed; predict the masked sequence).
> - **Sentence permutation:** `D E . A B C .` (restore original sentence order).
> - **Document rotation:** `C . D E . A B` (identify where the document really starts).

> [!card]- What does each of token masking, token deletion and text infilling hide from the BART encoder?
> - **Masking** marks the gap, so the model only has to fill it.
> - **Deletion** hides **where** the gap is, since nothing marks it.
> - **Infilling** hides **how long** the gap is, since one `[MASK]` can stand for any number of tokens (including zero).

> [!card]- Define document rotation in BART, and why is "rotation" the better name than "document permutation"?
> A random position of the document is used as the first token, followed by the rest from that position, followed by the actual beginning up to the random position. The task is to predict the actual first position. "Rotation" is accurate because order is preserved cyclically; nothing is shuffled.

> [!card]- Summarise BART's comparison of pretraining objectives (SQuAD, MNLI, perplexity on generation tasks).
> - **Text infilling** is the best single noise: SQuAD **90.8** F1, best XSum (6.61) and ConvAI2 (11.05) perplexity.
> - **Deletion beats masking** on all four generation tasks (it is harder: locate the gaps).
> - **Document rotation and sentence shuffling alone are poor:** SQuAD 77.2 and 85.4, ELI5 PPL 53.69 and 41.87; too weak a token-level signal.
> - Infilling + shuffling: SQuAD 90.8, best CNN/DM perplexity **5.41**.
> - Plain left-to-right LM: best ELI5 (21.40), worst SQuAD (76.7), since it lacks bidirectional context.
> - BERT Base keeps the best MNLI (84.3).

> [!card]- How is BART fine-tuned for classification and for span prediction such as SQuAD?
> - **Classification:** the same uncorrupted input goes into encoder and decoder; the label is predicted from the **last hidden time step** of the decoder, which has attended to the whole input.
> - **Span prediction:** each token is labeled, and the model learns to predict the **beginning and end** labels of the answer span.

> [!card]- How is BART fine-tuned for machine translation, and what were the Romanian→English results?
> - A **randomly initialised source-language encoder replaces BART's embedding layer**. It maps foreign tokens into vectors BART can treat like (noisy) English embeddings, which BART then denoises into English.
> - BART's parameters can stay **frozen** or be **updated**; the source vocabulary can differ from BART's.
> - Ro→En BLEU: baseline 36.80, **Fixed BART 36.29** (worse), **Tuned BART 37.96** (+1.16). Pretraining helps only when BART's parameters may adapt.
> - Same idea as parent-child transfer: keep a large English-trained model, train a small new component to connect a new source language.

> [!card]- How does large BART compare with BERT, XLNet and RoBERTa on SQuAD, and on CNN/DM and XSum summarisation?
> - SQuAD 1.1 EM/F1: BERT 84.1/90.9, XLNet 89.0/94.5, RoBERTa 88.9/94.6, **BART 88.8/94.6**. SQuAD 2.0: RoBERTa 86.5/89.4, BART 86.1/89.2. Adding a decoder costs nothing on understanding tasks.
> - Summarisation ROUGE-1/2/L: BART best on every column (CNN/DM 44.16/21.28/40.90; XSum 45.14/22.27/37.25). Largest lead on XSum: +3.69 R1 and +3.48 R2 over RoBERTaShare.
> - Lead-3 is strong on CNN/DM (40.42 R1) and useless on XSum (16.30 R1, 1.60 R2).

> [!card]- Name the benchmarks and metrics used to evaluate XLM, InfoXLM, the within/across tests, NMT transfer and BART, and what each measures.
> - **XNLI:** crosslingual NLI, premise/hypothesis three-way classification in 15 languages; accuracy.
> - **XSQuAD:** extractive QA with context and question possibly in different languages, 11 languages; F1.
> - **Within/across scores:** same-language against mixed-language input, per language.
> - **Perplexity:** language modeling quality (Nepali; BART's ELI5, XSum, ConvAI2, CNN/DM); lower is better.
> - **Cosine similarity, L2 distance, SemEval'17:** alignment of crosslingual word embeddings.
> - **BLEU:** MT quality. **ROUGE-1/2/L:** summarisation. **SQuAD EM/F1, MNLI accuracy:** understanding tasks.

> [!exam]- Describe multilingual NMT as in Johnson et al. (2017): training data, how the output language is chosen, what is shared, and how zero-shot translation arises.
> Must hit:
> 1. **One single system for all translation directions**, trained jointly: knowledge is transferred **in parallel**, as opposed to parent-child transfer **in stages**.
> 2. **Data:** a collection of parallel corpora, **English-centric**: only $xx \rightarrow en$ and $en \rightarrow xx$ directions, because that is where the resources are. Directions are **mixed during training, even within a batch**.
> 3. **Target language tag:** every training example gets a tag such as `<2es>` prepended to the **source** as fed to the encoder. It names the target language; the source language is never named.
> 4. **Shared everything:** all parameters shared for all language combinations, one WordPiece vocabulary for all languages (32k, 64k), shared source, target and output embeddings. The architecture is a standard encoder-decoder.
> 5. **Zero-shot:** training covers many-to-one and one-to-many; asking for a non-English pair (many-to-many, testing only) such as De→Fr is a direction never seen in training and relies entirely on transfer.
> 6. **Balancing:** languages are sampled with weights, between proportional (high-resource dominate) and uniform (high-resource suffer).
>
> Losing marks: saying the tag names the source language, or that the model trains on non-English pairs.

> [!card]- What are the two ways of transferring knowledge between language pairs in NMT?
> - **In stages:** parent-child transfer learning. Train on a high-resource pair, then continue training on the low-resource pair.
> - **In parallel:** multilingual joint training (Johnson et al., 2017). One single system trained on all translation directions at once.

> [!card]- Define many-to-one, one-to-many and many-to-many multilingual NMT. Which are used in training?
> - **Many-to-one:** many source languages into one single target language. Training.
> - **One-to-many:** one source language into many target languages. Training.
> - **Many-to-many:** many source languages into many target languages. **Testing only**: with English-centric training, non-English pairs are never trained, so testing them is zero-shot.

> [!card]- What are the two indicators that knowledge transfer happens within multilingual NMT?
> - **Performance on zero-shot directions**: any quality there can only come from transfer.
> - **Whether the multilingual model outperforms a baseline trained only on the direction-relevant data** (a bilingual system for that pair).

> [!card]- In multilingual NMT, what does a tag like `<2ja>` mean and where is it placed?
> It names the **target** language ("to Japanese"). It is prepended to the **source sentence as fed to the encoder**. The same English sentence goes to Spanish with `<2es>` and to Japanese with `<2ja>`. No tag names the source language.

> [!card]- Which parts of Johnson et al.'s multilingual NMT system are shared across languages?
> Everything: one system with all parameters shared for all language combinations, one WordPiece vocabulary for all languages (32k, 64k, ...), and shared source, target and output embeddings.

> [!card]- What happens to high- and low-resource languages under proportional and under uniform language sampling in multilingual NMT, and what is the compromise?
> - **Proportional:** high-resource languages dominate; low-resource directions perform poorly.
> - **Uniform:** high-resource languages suffer; low-resource languages can see strong performance.
> - **Compromise:** in between. Over-sample low-resource languages, under-sample high-resource ones.

> [!card]- What do Johnson et al.'s many-to-one results (into English) show?
> Multi beats the bilingual Single baseline in every row, by +0.05 to +1.27 BLEU, on WMT data and on in-house production data 10 to 100 times larger (e.g. Pt→En 44.40 → 45.19). Largest gain WMT Fr→En without oversampling (+1.27). Sharing the English target side helps consistently.

> [!card]- What do Johnson et al.'s one-to-many results (out of English) show, and what does oversampling do?
> - **Mixed:** four of eight directions get worse, by up to 2.11 BLEU. The decoder has to produce several languages that compete for the same parameters.
> - **Oversampling moves the loss around:** with it, En→De +0.30 and En→Fr −2.11; without it, En→De −2.06 and En→Fr −0.79.
> - Production En→Es (+0.90) and En→Pt (+0.23) still gain.

> [!card]- Define transfer and interference in multilingual NMT.
> - **Transfer:** a direction improves because it shares a model with other languages.
> - **Interference:** a direction gets worse because the other languages compete with it for the same model capacity.

> [!card]- What do Arivazhagan et al.'s (2019) 103-language results show about transfer and interference?
> Languages grouped as High 25, Medium 52, Low 25; compared with bilingual baselines:
> - **Low-resource gains:** Any→En low 21.63 → **30.56** (+8.93); En→Any low 11.72 → 12.98 (+1.26).
> - **High-resource losses:** no multilingual model beats bilingual on the high group (Any→En 37.61 → 36.61; En→Any 29.34 → 28.75).
> - **Into English transfers far more than out of English** (many-to-one over one-to-many).
> - **All→All** (one model for both directions) is the weakest multilingual option in every column: high-resource Any→En 33.85.

> [!card]- In temperature-based language sampling, what do T = 1 and T = 100 mean, and how does temperature relate to the smoothing exponent alpha used for multilingual pretraining?
> - **T = 1:** proportional to data size. **T = 100:** uniform, in effect.
> - $p_l \propto (n_l / \sum_{l'} n_{l'})^{1/T}$, so $\alpha = 1/T$; T = 5 is $\alpha = 0.2$.

> [!card]- What do Arivazhagan et al.'s temperature results show for T = 1, T = 100 and T = 5?
> - **T = 1 (proportional):** best multilingual result on high-resource languages (28.63 En→Any, 34.60 Any→En), but En→Any low collapses to **6.24**, about half the bilingual 11.72.
> - **T = 100 (uniform):** best on low-resource (12.87, 27.32), worst on high-resource (27.20, 33.25).
> - **T = 5:** the compromise. Within 0.12 and 0.36 BLEU of uniform on low-resource, better than uniform on high-resource, best on medium in both directions. Its numbers equal the All→All rows, so that model used T = 5.

> [!card]- What does zero-shot translation in a massively multilingual NMT model look like when going from 10 to 102 languages?
> - Zero-shot directions have no data and rely entirely on transfer.
> - More languages improve five of six directions: De→Fr 11.15 → 14.24, Be→Ru 36.28 → **50.26**, Yi→De 8.97 → 20.00, Hi→Fi 2.98 → 8.76, Ru→Fi 6.02 → 9.06. Fr→Zh drops (15.07 → 11.83).
> - Results probably depend on which languages are included (closely related Be→Ru is high even with 10 languages).
> - Zero-shot translation is still lagging behind: unrelated pairs stay in single digits.

> [!card]- What is mBART, and does its pretraining contain a crosslingual signal?
> - Liu et al. (2020): one BART model trained on a **concatenation of monolingual data in multiple languages**. BART is to mBART as BERT is to mBERT.
> - **No crosslingual signal:** the BART denoising objective is always applied within the same language (input and output in the same language). Any crosslingual ability must emerge from the shared model and vocabulary, as with mBERT and XLM-R.

> [!card]- How is mBART's pretraining input built, and what role do the language ID tokens play?
> - Noise is **text infilling plus sentence permutation**: spans replaced by a mask, sentence order shuffled (`Where did __ from ? </s> Who __ I __ </s> <En>` for *Who am I? Where did I come from?*).
> - `</s>` separates sentences, so instances can span several sentences.
> - A language ID token (`<En>`, `<Ja>`) **ends the encoder input and starts the decoder input**; the decoder reconstructs the original text in the same language.

> [!card]- How is mBART fine-tuned for machine translation, and what are Sent-MT and Doc-MT?
> - The whole pretrained encoder-decoder is fine-tuned directly on parallel data. The **source** language tag ends the encoder input, the **target** language tag starts the decoder input, so the first decoder token selects the output language.
> - **Sent-MT:** sentence-level (`Who am I ? </s> <En>` → `私 は 誰 ？ </s> <Ja>`).
> - **Doc-MT:** document-level, several sentences at once (Japanese *Well then. See you tomorrow.* → English).

> [!card]- Why does translating with mBART not need the extra source encoder that BART needs?
> BART was pretrained on English only, so MT fine-tuning puts a new randomly initialised source encoder in front of it, and the gain was about one BLEU (Ro→En 36.80 → 37.96 tuned). mBART has seen every language it translates during pretraining, so the whole model is fine-tuned as is.

> [!card]- What do the mBART25 against random-initialisation results show across language pairs and data sizes?
> - mBART25 wins for **every pair in both directions**; average +7.2 and +4.4 BLEU for the two directions.
> - **Largest gains at a few hundred thousand pairs:** En-Vi (133K) +12.5/+10.6, En-Tr (207K) +10.3/+8.3, En-Ar (250K) +10.1/+4.7; also En-Hi (1.56M) +12.6/+6.6.
> - **Almost no data cannot be rescued:** En-Gu (10K) 0.0 → 0.3/0.1. En-Kk (91K) is where it starts to work: 0.8 → 7.4.
> - **Gains shrink with more data:** En-Ro (608K) +3.8/+3.4, En-Lv (4.5M) +3.7/+3.0.

> [!card]- Multiple selection. Which are true of multilingual NMT as in Johnson et al. (2017)? (A) Each language gets its own encoder. (B) A tag naming the target language is added to the source sentence. (C) Source, target and output embeddings are shared. (D) Training uses every language pair, including pairs without English.
> **B and C.** (A) is false: all parameters are shared across all language combinations. (D) is false: training is English-centric, $xx \rightarrow en$ and $en \rightarrow xx$ only; non-English pairs are zero-shot at test time.

> [!card]- Multiple selection. Which are true of language sampling in multilingual NMT? (A) Proportional sampling lets high-resource languages dominate. (B) Uniform sampling hurts low-resource languages. (C) A temperature of T = 100 approximates uniform sampling. (D) The best compromise over-samples low-resource languages.
> **A, C and D.** (B) is false: uniform sampling hurts **high**-resource languages; low-resource ones can see strong performance (T = 100 gives the best low-resource BLEU, 12.87 and 27.32).

> [!card]- Multiple selection. Which are true of Arivazhagan et al.'s results on 103 languages? (A) Low-resource languages gain most when translating into English. (B) Multilingual models beat bilingual baselines on the high-resource group. (C) All→All is weaker than the dedicated Any→En and En→Any models. (D) One-to-many gains more than many-to-one for low-resource languages.
> **A and C.** (B) is false: every multilingual setting scores under bilingual on the High 25 group. (D) is false: low-resource gains are +8.93 BLEU into English (many-to-one) against +1.26 out of English (one-to-many).

> [!card]- Multiple selection. Which are true of mBART? (A) It is pretrained on parallel data. (B) Its pretraining objective has no crosslingual signal. (C) A language ID token starts the decoder input. (D) Translation fine-tuning requires a new randomly initialised source encoder.
> **B and C.** (A) is false: it is trained on concatenated monolingual data. (D) is false: that is how English-only BART is used for MT; mBART is fine-tuned as is.

> [!card]- Multiple selection. Which are true of parent-child transfer learning for low-resource NMT? (A) The child model is initialised with the parent's parameters. (B) It works best when parent and child share the target language. (C) Unfreezing the target embeddings during child training improves Uzbek→English. (D) Transfer from a smaller parent to a larger child works as well as the reverse.
> **A and B.** (C) is false: letting the target embeddings train too lowers Uzbek→English dev BLEU from 15.0 to 14.7 and then 13.7; they overfit on the small child data. (D) is false: Kocmi and Bojar find reversed transfer helps little or hurts; it must flow from the larger corpus to the smaller.
