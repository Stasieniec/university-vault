# Archived flashcards: MNLP-L04 - Subword Segmentation

The long-form cards this note carried until 2026-10-08, when they were replaced by short single-fact cards. Kept for reference; not published.

Click a question to reveal its answer, or press **Study** to drill the whole set. Cards marked as exam questions are meant to be answered out loud or on paper first, then checked against the points listed.

> [!exam]- Compare BPE and Unigram LM as subword segmentation methods: mechanism, segmentation quality and downstream effect.
> Must hit three layers:
> 1. **Mechanism.** BPE (Sennrich et al., 2016) is **bottom-up**: start from characters and greedily merge the most frequent adjacent pair until the merge budget is spent. A merge is never undone, and each word has exactly one segmentation. Unigram LM (Kudo, 2018) is **top-down**: start from a huge seed vocabulary of substrings, fit a unigram LM by EM, and prune the tokens whose removal costs the least likelihood until the target size $k$ is reached. It scores whole segmentations, $P(\mathbf{x}) = \prod_i p(x_i)$, takes the global best by dynamic programming, and can return n-best or sampled segmentations.
> 2. **Segmentation quality** (Bostrom and Durrett, 2020, same data and same vocabulary size): F1 against CELEX2 English morphology is **19.3% (BPE) against 30.3% (Unigram LM)**; against MeCab Japanese word segmentation, 73.8% against 77.2%. Unigram LM recovers suffixes (`ly`, `ed`, `s`); BPE leaves word-initial single capitals and frequency artefacts such as `n|an|ote|chn|ology`.
> 3. **Downstream** (identical pretraining and model, only the tokenizer differs): Unigram LM wins the English tasks by roughly one point (SQuAD 1.1 EM 80.6 to 81.8) and Japanese TyDi QA by **12.3 EM and 12.3 F1** (EM 41.4 to 53.7).
> 4. **Vocabulary use:** Unigram LM produces longer segments on average and uses its vocabulary more effectively; BPE spends the bottom of its vocabulary on near-useless tokens.
>
> Losing marks: calling SentencePiece the algorithm (it is the toolkit that implements both); reading the low absolute English F1 as "tokenizers fail at English" when the meaningful number is the ratio between the methods.

> [!exam]- Why does neural machine translation need subword segmentation? Argue from the weaknesses of word-level, character-level and rule-based (FST) alternatives.
> - The vocabulary is one of the main bottlenecks of training an NMT system. How well a word is represented depends on its **frequency** and on the **number of different contexts** it occurs in, both properties of the training corpus.
> - Most new surface forms are **word formations**: inflection (`tall`, `taller`, `tallest`) and compounding (`Donaudampfschifffahrtsgesellschaft`). A word-level vocabulary treats `tall` and `tallest` as unrelated integers.
> - **Word-level:** 300K to 500K entries and still OOVs, huge embedding and softmax matrices, every unseen word collapses to `<UNK>` ("Kyrgyzstan" in a test sentence becomes an unspecified place).
> - **Character-level:** near-zero OOV, but each symbol carries almost no meaning and sequences are long and slow.
> - **Hand-built FSTs:** accurate, but extremely laborious, totally language dependent and give no preference among ambiguous analyses, so they do not scale to a hundred languages.
> - **Subwords** (typically 30K to 128K): learned from frequencies with no annotation, vocabulary size tunable, near-zero OOV, moderate sequence length. The price is that pieces are not clean morphemes.

> [!exam]- Train BPE on the corpus `the tall man is taller than the tallest man` for 10 merges, breaking ties by first occurrence. Give the merges and comment on the result.
> 1. Word types: `the` 2, `tall` 1, `man` 2, `is` 1, `taller` 1, `than` 1, `tallest` 1. Split each into characters and append `</w>`.
> 2. Initial counts: six pairs tie at 3: `t+h` (2 from `the`, 1 from `than`), `t+a`, `a+l`, `l+l`, `a+n`, `n+</w>`. `t+h` is encountered first, so it wins.
> 3. Merges in order: `t+h` (3), `t+a` (3), `ta+l` (3), `tal+l` (3), `a+n` (3), `an+</w>` (3), `th+e` (2), `the+</w>` (2), `m+an</w>` (2), `tall+e` (2).
> 4. Final corpus: `the</w>`, `tall </w>`, `man</w>`, `i s </w>`, `talle r </w>`, `th an</w>`, `the</w>`, `talle s t </w>`, `man</w>`.
> 5. Comment: the frequent words `the` and `man` became single tokens, which is desired. But `taller` and `tallest` ended as `talle + r` and `talle + s + t` where the morphological split is `tall + er` and `tall + est`. The greedy `tall + e` merge was locally the most frequent move and destroyed the morpheme boundary, and BPE can never undo a merge.
>
> Losing marks: counting `than` towards `t+a` (in `than` the `t` is followed by `h`).

> [!exam]- Segment `cats` by bottom-up DP with p(c)=.02, p(a)=.04, p(t)=.03, p(s)=.05, p(ca)=.002, p(at)=.0015, p(ts)=.0004, p(cat)=.003, p(ats)=.00005, p(cats)=.0001.
> 1. **Length 1:** $m[i,i]$ is the character probability: 0.02, 0.04, 0.03, 0.05.
> 2. **Length 2:** `ca` stays whole, 0.002 (split gives 0.0008); `at` stays whole, 0.0015 (split 0.0012); `ts` splits, $0.03 \times 0.05 = 0.0015 > 0.0004$, so $s[3,4]=3$.
> 3. **Length 3:** `cat` stays whole, 0.003 (splits give 0.00003 and 0.00006); `ats` is best as `at|s`, $0.0015 \times 0.05 = 0.000075$, beating unsplit 0.00005 and `a|ts` 0.00006, so $s[2,4]=3$.
> 4. **Length 4:** unsplit `cats` 0.0001; $k=1$: $0.02 \times 0.000075 = 1.5\times10^{-6}$; $k=2$: $0.002 \times 0.0015 = 3\times10^{-6}$; $k=3$: $0.003 \times 0.05 = 1.5\times10^{-4}$. Winner $k=3$, $s[1,4]=3$.
> 5. **Read back:** $s[1,4]=3$ splits into $c[1..3]$ and $c[4]$; $s[1,3]$ is unset, so the result is `cat | s` with $P = 1.5\times10^{-4}$.
> 6. **Points to make:** the rare whole word `cats` (0.0001) loses to two common pieces, which BPE cannot do once it has merged `cats`. The morphologically correct split appears without any morphology in the model, because plural `s` is very frequent. Combining the best scores of two halves is valid only because the unigram model treats segments as independent.

> [!exam]- Explain fertility and why one subword vocabulary serves languages unequally. Include byte-level BPE in your answer.
> - **Fertility** = subword tokens / words, the average number of segments per word. 1.0 means every word is one token. English: 1.343 (BPE) and 1.318 (Unigram LM).
> - **Drivers:** merge count or target vocabulary size (the one factor you control), average surface word length, morphological richness. Isolating Latin-script languages are close to 1 for common words; agglutinative languages (Finnish, Turkish, Hungarian) have many more, individually rarer word forms, which get split further.
> - **Consequences:** (1) the model must reassemble meaning from many small parts; (2) no tokenization scheme dominates across all languages and scripts, and a multilingual vocabulary is a compromise for all of them, hence the push to 128K; (3) commercial APIs price by token, so high-fertility languages pay more for the same content; (4) fixed context windows fill up faster.
> - **Byte-level BPE:** the 256-byte base alphabet guarantees zero OOV, but UTF-8 uses 1 byte for ASCII, 2 for Latin with diacritics, Cyrillic and Greek, 3 for most CJK, Devanagari and Thai, and 4 for many emoji. Those scripts spend merges just getting from bytes back to characters and end up with shorter subsegments. The cause is UTF-8 encoding length, which has nothing to do with the language itself.
> - **Methodology:** when comparing across languages, measure and report fertility per language, because sequence length is a confound.

> [!exam]- Walk through Kudo's (2018) Unigram LM vocabulary construction algorithm and correct the pruning rule in the version printed as Algorithm 2 in MNLP.
> 1. Seed $V$ with all substrings that occur more than once in $D$ and do not cross word boundaries (millions of candidates: this is the top-down start).
> 2. While $|V| > k$: fit a unigram LM $\theta$ to $D$ by EM (segmentation is latent: the E-step computes expected subword counts over all segmentations, the M-step renormalises them into $p(x)$).
> 3. For each token $t$: $L_t = p_\theta(D) - p_{\theta'}(D)$, where $\theta'$ is the LM without $t$ and strings that used $t$ are re-segmented.
> 4. Remove $\min(|V|-k, \lfloor\alpha|V|\rfloor)$ tokens, $\alpha \in [0,1]$ a hyperparameter, then refit. Pruning is gradual because each $L_t$ is computed assuming all other tokens are still present.
> 5. Fit the final unigram LM and return $V$ and $\theta$.
>
> **The error:** the printed version removes the tokens with the **highest** $L_t$. Removing a token cannot raise the likelihood, so $L_t \ge 0$ and a large $L_t$ marks a valuable token: taken literally the printed rule prunes the most useful pieces first, and it is wrong. **Correct rule:** remove the tokens with the **smallest** $L_t$ (equivalently, keep the top-scoring pieces), which is Kudo's intent and what SentencePiece implements. Also, Kudo never prunes **single-character tokens**, so every string stays segmentable; the printed lines would allow a character to be removed.

> [!card]- What two properties of the training data determine how well a word is represented in NMT, and which word-formation processes produce most new surface forms?
> Its **frequency** in the training data and the **number of different contexts** it occurs in. Both belong to the corpus, so any fixed word list is a bet that future text looks like the training text.
> Many word occurrences are the result of word formation: **inflection** (`tall`, `taller`, `tallest`) and **compounding** (`Donaudampfschifffahrtsgesellschaft`). A word-level vocabulary that has seen `tall` a thousand times and `tallest` twice treats them as unrelated.

> [!card]- Define a subword, and compare word-level, character-level and subword vocabularies on size, OOV rate and sequence length.
> A subword is either a **character n-gram** or a **morphologically meaningful unit** (language dependent). Data-driven methods aim at the first and are judged against the second.
> - **Word-level:** 300K to 500K and still not enough; high OOV, unbounded on new text; short sequences; huge embedding and softmax matrices, no sharing between related forms, unseen words become `<UNK>`.
> - **Character-level:** tens to a few hundred symbols; essentially zero OOV; very long sequences; symbols carry almost no meaning.
> - **Subword:** typically 30K to 128K; near-zero OOV; moderate length; the vocabulary size is a tunable dial learned from data.

> [!card]- What is a finite-state transducer (FST) morphological analyzer, and what are its strengths and three weaknesses?
> A hand-built machine of **states** and **transitions that consume a string and emit a string**, written `input:output` (e.g. `a:x`). The input is a morpheme, the output an analysis such as "1st person, noun, singular".
> - **Strengths:** very powerful, efficient, handles cases and exceptions with great accuracy.
> - **Weaknesses:** (1) extremely laborious to produce; (2) totally language dependent; (3) no preference among ambiguous analyses, just an unranked set (probabilistic FSTs exist, but the plain formalism ranks nothing).
>
> Language dependence alone rules it out for multilingual NLP: one expert per language does not scale to a hundred languages.

> [!card]- What is Morfessor, what principle is it based on, and what two costs does that principle trade off?
> The first data-driven morphological analyzer, **Creutz and Lagus (2002)**, based on **Minimum Description Length (MDL)**. MDL asks (1) how much it costs the model to generate the data and (2) how much the model itself costs, and minimises the sum.
> A whole-word vocabulary makes the data cheap and the model enormous; single characters make the model tiny and the data expensive. The optimum lies where reusable pieces exist, roughly where morphemes are, because morphemes are the pieces a language reuses.

> [!card]- Current data-driven segmenters make three simplifications relative to morphological analysis. Name them and the two practical properties gained in exchange.
> **Simplifications:**
> 1. No labelling of the data: only a split, no morphological analysis.
> 2. No character substitutions or deletions: `run`/`ran` and `city`/`cities` get no special treatment.
> 3. Strictly concatenative: a word is exactly the concatenation of its segments.
>
> **Gains:** no data annotation or linguistic knowledge required; they can be fitted to a desired vocabulary size, which sets the size of the embedding and output layers.

> [!card]- Name the three frequency-based subword methods with their reference, direction and key property.
> - **Byte Pair Encoding**, Sennrich et al. (2016): bottom-up (merge); greedy and deterministic.
> - **WordPiece**, Schuster et al. (2012): bottom-up (merge); chooses merges by likelihood gain where BPE uses raw count.
> - **Unigram LM** (in SentencePiece), Kudo (2018): top-down (prune); probabilistic, supports multiple segmentations.
>
> All three split words using frequencies alone, with no linguistic knowledge.

> [!card]- Give the BPE training algorithm as numbered steps. When does it stop, and what is its only hyperparameter?
> 1. Split words into characters, keeping word boundaries, and collect word frequencies.
> 2. Consider all neighbouring symbol pairs and collect their frequencies.
> 3. Merge the most frequent pair and repeat from step 2.
>
> Stop when the maximum number of merge operations is reached. The number of merges is the only hyperparameter and controls the final vocabulary size.

> [!card]- What does the BPE output `The preside@@ nt of France arr@@ ived in K@@ yrg@@ yz@@ stan .` illustrate?
> `@@` marks a piece that continues into the next token. Frequent words (`The`, `of`, `France`, `in`) survive whole. `president` is cut at `preside|nt`, which is no morpheme boundary; `arrived` at `arr|ived`, close to one but on the wrong side. The unseen word `Kyrgyzstan` becomes four pieces and is therefore representable at all. The trade: clean morphology is given up in exchange for being able to write down any word.

> [!card]- In Sennrich et al.'s BPE reference code (Algorithm 1), what does `vocab` map, what do `get_stats` and `merge_vocab` do, and why is training cheap?
> - `vocab` maps a **space-separated symbol sequence** to a **word-type frequency**: `'l o w </w>': 5` means the type `low` occurred 5 times.
> - `get_stats` counts every adjacent pair weighted by the word's frequency (a pair inside a word seen 5000 times counts 5000 times).
> - `merge_vocab` rewrites `a b` as `ab` everywhere; the regex `(?<!\S)...(?!\S)` (not preceded or followed by a non-space) ensures it matches whole symbols only, never part of a longer symbol.
> - Training runs over the type vocabulary instead of the whole corpus, which is why it is cheap.

> [!card]- What is the end-of-word marker `</w>` for in BPE, and what does it not do?
> It lets a piece be word-final or word-internal: word-final `est` in `highest` and word-initial `est` in `establish` become different symbols. That makes it possible to see where one word ends and the next begins after segmentation, so detokenization works.
> It does **not** stop merges across words: each word type is a separate key in `vocab`, so no pair ever spans two words in the first place.

> [!card]- How does Sennrich et al.'s BPE code break ties between equally frequent pairs, and why does that matter for reproducibility?
> `max(pairs, key=pairs.get)` returns the **first** key attaining the maximum, and dicts iterate in insertion order, so the pair encountered first while scanning the vocabulary wins. Other implementations break ties lexicographically or by the frequency of the constituent symbols.
> So two tokenizers trained on the same corpus with the same merge count can produce different vocabularies. Pin the tokenizer library and its version; "BPE, 32k merges" does not identify a tokenizer.

> [!card]- Trace BPE for 10 merges on {`l o w </w>`: 5, `l o w e r </w>`: 2, `n e w e s t </w>`: 6, `w i d e s t </w>`: 3}, with counts.
> 1. `e+s` 9 (newest 6 + widest 3; three-way tie with `s+t` and `t+</w>`, `e+s` is seen first)
> 2. `es+t` 9
> 3. `est+</w>` 9
> 4. `l+o` 7 (low 5 + lower 2)
> 5. `lo+w` 7
> 6. `n+e` 6
> 7. `ne+w` 6
> 8. `new+est</w>` 6
> 9. `low+</w>` 5
> 10. `w+i` 3
>
> Final: `low</w>` 5, `low e r </w>` 2, `newest</w>` 6, `wi d est</w>` 3. `lower` is still four tokens because ten merges are not enough to reach `er</w>`: the vocabulary-size dial in miniature.

> [!card]- How is a trained BPE model applied to a new word, and why is BPE deterministic?
> Split the input into characters (with `</w>` appended) and apply the merge operations **in the order they were learned**:
> 1. For each merge $(a,b)$ in learned order,
> 2. scan the symbols left to right and replace every adjacent $a, b$ with $ab$.
> 3. Return the remaining symbols.
>
> The learned model is an **ordered list of merge rules**, and applying them out of order gives a different segmentation. Practical implementations keep a rank table, repeatedly apply the lowest-ranked pair present (equivalent and faster), and cache results per word type. The same word always gets the same segmentation, whatever sentence it is in.

> [!card]- What are the three major advantages of BPE reported by Sennrich et al., and for which language pairs do they hold?
> 1. Significantly reduces vocabulary size (less memory, better speed).
> 2. Better translation quality.
> 3. Significantly reduces the number of out-of-vocabulary items.
>
> They hold for **both high- and low-resource** language pairs.

> [!card]- In Sennrich et al.'s English to German results, how do word-level and subword systems compare, and what do WDict and BPE-J90k show?
> - **WUnk** and **WDict** carry 300 000 source and 500 000 target entries and score 20.6 and 22.0 BLEU (single system). Subword systems with 60k to 90k entries match or beat them: C2-50k 22.8, BPE-60k 21.5, BPE-J90k 22.8 (8-way ensembles: 25.3, 24.5, 24.7 against WDict's 24.2).
> - WUnk maps unknown words to `<UNK>`; WDict uses a back-off dictionary to copy or translate them, worth **1.4 BLEU** on its own, which shows how much damage the unknown-word problem does.
> - **BPE-J90k** is **joint** BPE, trained on the union of both languages so the same string is segmented identically on both sides, which helps the model copy names across.

> [!card]- What vocabulary range is typical for BPE today, and why is the upper end tied to multilinguality?
> **30K to 128K** merge operations. A vocabulary that must cover many scripts needs more room before any single language gets a decent share of it, so the 128K end is a multilingual concession.

> [!card]- State the WordPiece merge criterion next to BPE's, define the terms, and explain how it changes which pairs get merged.
> $$\text{BPE: } \arg\max_{(a,b)} \text{count}(ab) \qquad \text{WordPiece: } \arg\max_{(a,b)} \frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$$
> WordPiece merges the pair that most increases the likelihood of the training data under a unigram language model, which reduces to this ratio. $\text{count}(ab)$ is the frequency of the adjacent pair; $\text{count}(a)$ and $\text{count}(b)$ are the frequencies of each symbol alone.
> The denominator divides out pairs whose parts are common anyway (BPE happily merges `t + h`), so WordPiece prefers pairs that are **surprisingly** frequent together, a pointwise-mutual-information-style criterion. This pushes it slightly closer to morpheme-like pieces. Otherwise it is the same bottom-up merge procedure as BPE. It is the tokenizer of BERT.

> [!card]- How do WordPiece and BPE mark word structure differently, and why does it matter in practice?
> WordPiece marks **continuation**: `playing` becomes `play`, `##ing`, where `##` means "attaches to the left". BPE as in Sennrich et al. marks **word end** with `</w>`. The information content is the same but the strings differ, and mixing the two conventions is a classic way to break a pipeline.

> [!card]- What are BPE's two structural limitations, and what does each one prevent?
> - **Greedy:** once a sequence is merged it stays merged. No lookahead and no undo, so an early locally optimal merge can block a globally better segmentation (`tall + e` destroying `tall|er`).
> - **Deterministic:** a word is always segmented the same way. You cannot sample alternative segmentations, so segmentation cannot serve as training-time regularisation, and you cannot hedge at inference.
>
> Unigram LM (Kudo, 2018) addresses both.

> [!card]- State the four properties of Unigram LM (Kudo, 2018), and say how it relates to SentencePiece.
> 1. Considers **all possible segmentations** of a word.
> 2. Optimises for the **best global segmentation**.
> 3. Supports **multiple segmentations** (sampling and n-best lists).
> 4. **Top-down:** starts from a large existing vocabulary and shrinks it to the desired size, the mirror image of BPE, which merges upward from characters.
>
> It is part of Google's **SentencePiece** toolkit, whose other algorithm is BPE.

> [!card]- Give the Unigram LM probability of a segmentation and the rule for choosing the best one, defining every term.
> $$P(\mathbf{x}) = \prod_{i=1}^{M} p(x_i) \quad \text{subject to} \quad \sum_{x \in V} p(x) = 1, \qquad \mathbf{x}^{\ast} = \arg\max_{\mathbf{x} \in S(X)} P(\mathbf{x})$$
> - $\mathbf{x} = (x_1, \dots, x_M)$: one segmentation of the string into $M$ subwords, assumed **independent**
> - $p(x_i)$: unigram probability of subword $x_i$
> - $V$: the subword vocabulary
> - $S(X)$: the set of all segmentations of string $X$
>
> BPE picks a segmentation by replaying local decisions; Unigram LM scores whole segmentations and takes the best.

> [!card]- In Unigram LM vocabulary construction, what does the token loss $L_t$ measure, which tokens should be pruned, and what is wrong with Algorithm 2 as printed in MNLP?
> $L_t = p_\theta(D) - p_{\theta'}(D)$, with $\theta'$ the LM without token $t$: how much corpus likelihood drops if $t$ is removed and every string using it is re-segmented. A token always replaceable by a cheap split has a tiny $L_t$; one nothing else can express has a large $L_t$.
> **Correct rule:** remove the tokens with the **smallest** $L_t$ (keep the top-scoring pieces), as Kudo (2018) intends and SentencePiece implements. The printed line 12 says remove those with the **highest** $L_t$, which would prune the most valuable pieces first: that version is wrong and must not be implemented as printed.
> Kudo also never prunes **single-character tokens**, so every string stays segmentable and pruning cannot create OOV items; the printed algorithm omits this.

> [!card]- In Unigram LM vocabulary construction, what does fitting the LM involve, and why is pruning done gradually?
> - **Fitting** is EM: the segmentation of each string is a latent variable. The E-step computes expected counts of each subword over all segmentations of each string; the M-step renormalises them into $p(x)$. It is refitted every round, which makes the loop expensive.
> - **Gradual pruning:** remove at most $\min(|V|-k, \lfloor\alpha|V|\rfloor)$ tokens per round, $\alpha \in [0,1]$, then refit, so remaining pieces can absorb the work of deleted ones. Removing straight down to $k$ in one step would be much worse, since each $L_t$ was computed assuming all other tokens are still present.
> - The seed vocabulary (all substrings occurring more than once, not crossing words) is typically millions of candidates.

> [!card]- How many segmentations does an $n$-character string have, why, and what is the brute-force procedure? What two ingredients does scoring need?
> $2^{n-1}$: there are $n-1$ positions between characters and each is independently a cut or not.
> Brute force: (1) generate all $2^{n-1}$ segmentations, (2) score each, (3) select the best. A 15-character German compound has $2^{14} = 16\,384$ segmentations per word per occurrence, so this is unusable.
> Scoring needs **scores for segments** (units not further split) and a way to **combine scores of sub-segmentations**. In the unigram model these are $p(x_i)$ and multiplication, and the product over independent parts is what makes dynamic programming applicable.

> [!card]- Cut a sequence of length 4 to maximise value with segment prices 1, 5, 8, 9 for lengths 1 to 4. What is the best cut, and what does it show about segmentation?
> There are $2^3 = 8$ cuts. Uncut: 9. $1+3$ or $3+1$: 9. $2+2$: **10**. Three pieces ($1+1+2$ in any order): 7. Four pieces: 4.
> Best is $2+2$ with value 10. Neither extreme wins: the optimum is an interior split, and finding it requires comparing whole configurations instead of making one local decision. Subword segmentation is the same problem with unigram probabilities as prices and multiplication in place of addition.

> [!card]- Define dynamic programming, name its two strategies, and compare them.
> Solve each sub-problem **only once** and **store** its solution: extra memory buys computation time, a **time-memory trade-off**.
> - **Top-down with memoization:** write the recursion naturally and cache each result the first time it is computed.
> - **Bottom-up:** fill a table in an order that guarantees every sub-result is ready before it is needed (for segmentation, by increasing span length).
>
> Both have the **same asymptotic run time**. Bottom-up avoids recursion depth limits and has better cache behaviour; top-down computes only the sub-problems it actually needs.

> [!card]- Give the bottom-up DP segmentation algorithm as numbered steps, and say what $m[i,j]$ and $s[i,j]$ hold.
> 1. Let $n$ be the length of word $w$ and $c[1..n]$ its characters; create tables $m$ and $s$.
> 2. For $i = 1..n$: $m[i,i] = p(c[i])$.
> 3. For span length $l = 2..n$, for $i = 1..n-l+1$, set $j = i+l-1$ and:
> 4. initialise $m[i,j] = p(c[i..j])$, the no-split option (zero, or $-\infty$ in log space, if the substring is not in $V$);
> 5. for $k = i..j-1$: $t = m[i,k] \cdot m[k+1,j]$; if $t > m[i,j]$, set $m[i,j] = t$ and $s[i,j] = k$.
> 6. Return $m$ and $s$.
>
> $m[i,j]$ is the score of the best segmentation of $c[i..j]$; $s[i,j]$ is the split point that achieved it, unset when leaving the span whole is best. Filling by increasing length ensures both halves are final. Optimising halves separately is valid only because the unigram model is independent across segments.

> [!card]- How is the segmentation read back out of the DP split table $s$?
> Recursive procedure Segments$(i,j)$:
> 1. If $s[i,j]$ is unset, output $c[i..j]$ as one subword.
> 2. Otherwise let $k = s[i,j]$, call Segments$(i,k)$, then Segments$(k+1,j)$.
>
> Start with Segments$(1,n)$.

> [!card]- What is the complexity of span-table DP segmentation, and how do production implementations do better?
> $O(n^2)$ cells with $O(n)$ work each gives $O(n^3)$ time and $O(n^2)$ space in word length $n$: fine for words, not for sentences, one reason segmentation is done per word.
> Because segments are independent, one best score per end position suffices:
> $$\text{best}[j] = \max_{i<j} \text{best}[i] \cdot p(c[i{+}1..j])$$
> which is $O(n^2)$. Capping subword length at $L$ gives the Viterbi lattice in $O(nL)$, which production implementations use; SentencePiece runs it over whole sentences since it does no pre-tokenization.

> [!card]- Why must DP segmentation under a unigram model be done in log space, and what changes in the algorithm?
> Multiplying many probabilities underflows to zero in float32; every candidate then ties at zero and the argmax is meaningless. Replace `*` with `+`, replace $p(\cdot)$ with $\log p(\cdot)$, initialise unknown substrings to $-\infty$, and keep `>` as the comparison, since log is monotone. Nothing else changes.

> [!card]- How can you obtain more than the single best segmentation, and what is that used for?
> Viterbi returns only the most probable segmentation. For more:
> - **Yen's and Eppstein's algorithms** give n-best segmentations (general k-shortest-path algorithms; the segmentation lattice is a DAG);
> - or **disallow certain sub-solutions**: re-run the search with the winning split forbidden.
>
> Use: **subword regularization**. At training time, sample a segmentation from the n-best list, so the model sees `taller` as `tall|er` sometimes and `talle|r` other times and cannot overfit to one segmentation. It is free data augmentation, unavailable to plain BPE, which has no distribution to sample from; **BPE-dropout** later retrofitted a similar trick by randomly skipping merges.

> [!card]- How does SentencePiece's input handling differ from standard BPE implementations, and what are the two consequences?
> Standard BPE implementations **assume word boundary detection**: a whitespace or language-specific tokenizer has already split the input into words. SentencePiece treats the input as a **raw stream of Unicode characters, including spaces**, with the space escaped as a visible symbol `▁` (U+2581).
> Consequences: (1) no language-specific pre-tokenizer is needed, removing the last per-language engineering step and the thing that makes Japanese and Chinese awkward; (2) segmentation becomes **lossless**.

> [!card]- What is the desegmentation problem, why is it worse for languages like Japanese, and how does lossless segmentation solve it?
> `Hello world.` tokenized as `[Hello] [world] [.]` could be desegmented as `Hello world .`, `Helloworld.` or `Hello world.`: nothing records a space before `world` and none before `.`, so detokenizers guess with hand-written per-language rules. Japanese (`こんにちは世界。`) puts no spaces between words, so a rule that inserts spaces breaks Japanese and one that never does breaks English.
> Fix: a special whitespace character `▁`: `Hello▁world.` becomes `[Hello] [▁wor] [ld] [.]`. Desegmentation is joining the tokens and replacing `▁` with a space: no rules, no language knowledge, exactly invertible. Pieces may straddle a human word boundary because the boundary is encoded in the characters.

> [!card]- Contrast BPE and Unigram LM segmentations of English words and numbers from Bostrom and Durrett (2020).
> - Unigram LM: `▁fur ious ly`, `▁tri cycle s`, `▁nano technology`, `▁corrupt ed`, `▁Complete ly`, `▁pre post er ous`, `▁suggestion s`, `▁1848`.
> - BPE: `▁fur iously`, `▁t ric y cles`, `▁n an ote chn ology`, `▁cor rupted`, `▁Comple t ely`, `▁prep ost erous`, `▁184 8`.
>
> Unigram LM finds real morpheme boundaries; BPE produces frequency artefacts. In `nanotechnology`, BPE has spent its early merges on generic frequent clusters and has no symbol for `nano`, while Unigram LM chose its vocabulary by usefulness. Numbers: BPE splits `1848` as `184` + `8`, so the model must rebuild the year from a meaningless shared prefix. Check what a tokenizer does to numbers, dates and code.

> [!card]- How do BPE and Unigram LM segment 磁性は様々に分類がなされている。 ("Magnetism is classified in various ways"), and why does the difference matter?
> - BPE: `磁 | 性は | 様々 | に分類 | がなされている | 。`
> - Unigram LM: `磁 | 性 | は | 様々 | に | 分類 | がなされている | 。`
>
> BPE glues the topic particle は onto 性 and the particle に onto 分類. Absorbing grammatical function words into content words is worse than an arbitrary split inside a word, because it destroys a unit the model needs as a unit. Unigram LM separates them correctly.

> [!card]- What do token length and frequency-rank distributions show about BPE and Unigram LM vocabularies of the same size?
> - **English lengths:** identical at length 1 (both keep the full character alphabet); BPE has more vocabulary entries of length 3 to 5, Unigram LM more from length 7 upward.
> - **Conclusions:** Unigram LM produces **longer segments on average** and **uses its vocabulary space more effectively, with more tokens of moderate frequency**.
> - **Frequency against rank:** the curves coincide for roughly the first 15 000 ranks; then BPE's collapses while Unigram LM's holds near $10^4$ for a couple of thousand more ranks. BPE spends the bottom of its vocabulary on near-useless tokens, wasted embedding parameters.
> - **Japanese:** distribution peaks at length 2 and dies out by about 9 (a Japanese character carries far more information than a Latin one); Unigram LM again produces longer segments on average.

> [!card]- Which tokens does BPE over-produce relative to Unigram LM in English and vice versa, and how do their tokens per word compare?
> - **BPE:** word-initial single capitals (`▁H`, `▁L`, `▁M`, `▁T`, `▁B`, `▁P`, `▁C`, `▁K`, `▁D`, `▁R`), debris left when a capitalised word cannot be merged into anything.
> - **Unigram LM:** suffixes and punctuation (`s`, `.`, `,`, `ed`, `d`, `ing`, `e`, `ly`, `t`, `▁a`): morphology.
> - **Tokens per word type:** 4.721 (BPE) against 4.633. **Tokens per word:** 1.343 against 1.318, about a 2% difference in sequence length, in Unigram LM's favour but small.

> [!card]- Give BPE and Unigram LM precision, recall and F1 against CELEX2 (English) and MeCab (Japanese), and interpret them correctly.
> - English (CELEX2): BPE P 38.6%, R 12.9%, F1 19.3%; Unigram LM 62.2%, 20.1%, 30.3%.
> - Japanese (MeCab): BPE 78.6%, 69.5%, 73.8%; Unigram LM 82.2%, 72.8%, 77.2%.
>
> Low English recall is expected: neither method tries to find morphemes, and most English words are frequent enough to stay whole (`walked` as one token misses `walk|ed`). The meaningful figure is the **ratio**: about 1.6x on English F1, about 1.05x on Japanese. The references measure different things (morpheme boundaries inside spaced words against word boundaries in unspaced text), and the downstream gap runs the other way (about one point in English, 12.3 on Japanese TyDi QA), so a larger gold-standard gap does not predict a larger task gap.

> [!card]- What did Bostrom and Durrett find downstream when only the tokenizer differed, and what is the multilingual moral?
> - **English:** Unigram LM wins consistently by about one point: SQuAD 1.1 EM 81.8 against 80.6 (F1 89.3 against 88.2), MNLI matched 82.8 against 81.4, CoNLL NER test F1 90.4 against 90.2.
> - **Japanese TyDi QA:** EM 53.7 against 41.4, F1 54.4 against 42.1, a gap of **12.3** on both.
> - The BERT_BASE row was trained on different data and is no controlled comparison (which is why it wins MNLI and NER); it only shows the models are in a sensible range.
>
> Moral: the tokenizer is a hyperparameter invisible in English benchmarks and dominant outside them. A one-point English gain from a modelling change with an unreported tokenizer may be a tokenizer effect, so multilingual evaluation must control for segmentation.

> [!card]- Why is OOV handling a motivation for subword segmentation, and how can BPE and Unigram LM still produce OOVs?
> Handling low-frequency words, including zero-frequency (OOV) ones, is one of the main motivations for subword segmentation. BPE and Unigram LM have very low OOV rates, but not zero:
> - **typos** can sometimes still cause OOVs;
> - **unknown characters** result in OOVs. This is the real problem: a character vocabulary holds only the characters seen in training, so an emoji, a rare CJK character or an unseen script yields a genuine unknown symbol with nothing to fall back to.

> [!card]- What is byte-level BPE, why does it guarantee a zero OOV rate, and what does it cost in the multilingual case?
> It runs BPE on **raw UTF-8 bytes** instead of Unicode characters. The base alphabet is fixed at **256 symbols**, so any input in any script, including unseen scripts, can be encoded byte by byte and then merged where patterns emerge. It is used by GPT-2 and most decoder-only models since, and its base alphabet is smaller than a Unicode-character alphabet, freeing slots for merges.
> **Cost:** non-Latin, non-ASCII scripts need more bytes per character (1 for ASCII; 2 for Latin with diacritics, Cyrillic, Greek; 3 for most CJK, Devanagari, Thai; 4 for many emoji). Extra merges are spent getting from bytes back to characters, giving **shorter subsegments** for those scripts: the same nominal vocabulary size is a weaker tokenizer for, say, Hindi than for English.

> [!card]- Define fertility with its formula, and name the three factors that influence it for a language.
> $$\text{fertility} = \frac{\text{number of subword tokens}}{\text{number of words}}$$
> The average number of segments a word is split into; 1.0 means every word is one token, and higher is worse for the model.
> Factors:
> 1. the maximum number of merge operations (BPE) or target vocabulary size (Unigram LM), the one factor you control;
> 2. the average length of surface words in the language;
> 3. morphological richness: how many affix combinations are possible.
>
> Isolating Latin-script languages typically see fertility close to 1 for common words; agglutinative languages such as Finnish, Turkish or Hungarian get split much further.

> [!card]- Name the four consequences of high fertility for a language.
> 1. **Modelling:** a word chopped into many small, less meaningful parts forces the model to work harder to reassemble its meaning.
> 2. **No universal scheme:** no single tokenization dominates across all languages and scripts; a multilingual vocabulary is a compromise for all of them, which is why multilingual models push to 128K.
> 3. **Money:** commercial LLM APIs price by token count, so high-fertility languages pay more for the same content (English against Telugu).
> 4. **Capacity:** fixed context windows fill up faster, so a 128k-token context holds less document in a high-fertility language.
>
> Together these show the cost of English-centric NLP: pricing, context budget and per-token compute are all worse elsewhere, tracing back to merge tables learned from mostly English corpora.

> [!card]- Why is "we used SentencePiece" an under-specified description of a tokenizer?
> SentencePiece is Google's **toolkit**, and it implements both BPE and Unigram LM, so the phrase does not say which algorithm was used. What is specific to SentencePiece is the input handling: a raw Unicode stream including spaces, whitespace escaped as `▁`, no language-specific pre-tokenizer, lossless desegmentation. Report the tokenizer, the algorithm and the vocabulary size.

> [!card]- What methodological rules follow for experiments that compare tokenizers or compare results across languages?
> - Report the **tokenizer, algorithm and vocabulary size** as experimental settings.
> - Comparing across languages: **measure and report fertility per language**, because sequence length confounds anything that depends on it.
> - Comparing tokenizers: **hold the vocabulary size fixed**; 32k BPE against 64k Unigram LM measures the vocabulary size.
> - **Inspect actual segmentations** of real inputs before looking at metrics, to catch numbers split into digits and particles glued to nouns.
