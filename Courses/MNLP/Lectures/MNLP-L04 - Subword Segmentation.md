---
type: lecture
course: 5204MNLP6Y
week: 2
lecture: 4
status: complete
topics:
  - The vocabulary bottleneck
  - Out-of-vocabulary items
  - Rule-based morphological analyzers (FSTs)
  - Morfessor and Minimum Description Length
  - Byte Pair Encoding
  - WordPiece
  - SentencePiece Unigram LM
  - Viterbi segmentation by dynamic programming
  - Lossless segmentation
  - Byte-level BPE
  - Fertility and cross-lingual segmentation quality
---

# MNLP-L04: Subword Segmentation

> [!abstract] In short
> How can a model with a fixed vocabulary handle words it has never seen? Whole words turn every unseen word into `<UNK>`; single characters carry almost no meaning and make sequences long. The answer is a vocabulary of word pieces learned from frequency counts, at a size you choose.
>
> **BPE** builds it bottom-up, merging the most frequent adjacent pair. **WordPiece** merges too, but scores a pair against how common its parts are. **Unigram LM** works top-down, pruning a huge candidate list, and finds each word's most probable segmentation by dynamic programming, so it can also give alternatives. **SentencePiece** is the toolkit that runs BPE or Unigram LM on raw text with the space as a symbol; **byte-level BPE** starts from the 256 bytes, so nothing is out of vocabulary. Unigram LM's pieces are more morpheme-like, and that matters most outside English.
>
> The sample exam has no question on this lecture. Expect statements on which method does what: merging up or pruning down, greedy or not, deterministic or not, and who pays for each choice.

## The problem: a fixed vocabulary meets an open-ended language

The vocabulary is one of the main bottlenecks in training a neural machine translation (NMT) system. How well a word is represented depends on its **frequency** in the training data and the **number of different contexts** it occurs in, so rare words get poor embeddings and unseen words none.

Many word occurrences come from **word formation**: **inflection** (`tall`, `taller`, `tallest`) and **compounding** (German `Donaudampfschifffahrtsgesellschaft`, "Danube steamship company"). A word-level vocabulary treats `tall` and `tallest` as two unrelated entries.

The two obvious vocabularies both fail:

- **Whole words.** Translation systems carried 300,000 to 500,000 entries and still met unknown words, each of which becomes `<UNK>` (the unknown-word symbol): an unseen "Kyrgyzstan" leaves the president of France arriving somewhere unspecified. The embedding and output matrices are huge, and related forms share nothing.
- **Single characters.** Almost nothing is unknown, but each symbol carries almost no meaning and sequences become long and slow.

> [!definition] Subword
> Instead of whole words, use **sub-words**, where a subword is either
> - a **character n-gram** (a sequence of $n$ characters), or
> - a **morphologically meaningful unit** (language dependent).

A subword vocabulary sits between the extremes, at a size you set. **BPE is typically used with 30K to 128K merge operations**, the top of the range for multilingual models, since a vocabulary shared by many scripts needs more room.

## Doing it by hand: finite-state transducers

Historically, morphological analysers were hand-built **finite-state transducers (FSTs)**: states, with transitions that "digest" input and output a string (`a:x` reads `a`, writes `x`). The input is a morpheme, the output an analysis such as *1st person, noun, singular*.

- **Strength:** a very powerful, efficient framework that handles cases and exceptions with great accuracy.
- **Weaknesses:** **extremely laborious** to produce, **totally language dependent**, and given an ambiguous word they express **no preference** among its analyses (probabilistic FSTs, which do rank them, exist).

For multilingual NLP the language dependence is fatal: one expert-built analyser per language does not scale.

## Learning segments from data: Morfessor and description length

The alternative is to learn morphological regularities automatically from data. The first approach was **Morfessor** (Creutz and Lagus, 2002), based on the **Minimum Description Length (MDL)** principle, which adds up two costs:

- how much does it cost the model to **generate the data**?
- how much does the **model itself** cost?

Why this finds word pieces: whole words make the data cheap (one symbol per word) but the model enormous; characters make the model tiny but the data expensive. Minimising the sum favours pieces reused across many words, and the pieces a language reuses most are roughly its morphemes.

Current subword methods follow similar principles, with three deliberate simplifications:

- they **do not label** the data in any way: a split, with no actual analysis;
- they **do not account for character substitutions or deletions**, so `city`/`cities` or `run`/`ran` get no special treatment;
- they are **strictly concatenative**: a word is exactly its pieces joined together.

In return they need **no data annotation or linguistic knowledge**, and they **can be fitted to a desired vocabulary size** (computationally handy, since that size fixes the embedding and output layers). The three common methods, all based purely on frequencies, are **BPE** (Sennrich et al., 2016), **WordPiece** (Schuster and Nakajima, 2012) and **Unigram LM** (Kudo, 2018).

## BPE: merge the most frequent pair, again and again

Byte Pair Encoding (BPE) starts from characters and repeatedly fuses the most frequent pair of adjacent symbols into one new symbol. Frequent words end up as single symbols; rare words stay split into pieces that are themselves frequent.

> [!definition] BPE training
> 1. **Split words into characters** (keep word boundaries) and collect frequencies.
> 2. **Consider all neighbouring occurrences** (adjacent symbol pairs) and collect frequencies.
> 3. **Merge the most frequent sequence** and repeat from (2).
>
> Stop when a maximum number of merge operations has been reached.

The number of merges is the only hyperparameter, and it sets the final vocabulary size: the characters plus one new symbol per merge.

```pseudo
Algorithm: Learn BPE merges (Sennrich et al., 2016)
───────────────────────────────────────────────────
Input:  word types with their frequencies, number of merges m
Output: ordered merge list M

for each word type w:
    symbols(w) ← characters of w, then </w>
M ← empty list
repeat m times:
    for each adjacent pair (a, b):
        count(a, b) ← Σ over words w of freq(w) × (occurrences of a b in w)
    (a*, b*) ← the pair with the highest count       // ties: implementation-specific
    append (a*, b*) to M
    in every word, replace adjacent a* b* by the single symbol a*b*
return M
```

Training runs over **word types with their frequencies**, so it is cheap. Each word is counted on its own, so **no pair ever spans two words**. `</w>` marks the end of a word: word-final `est</w>` (*tallest*) differs from word-internal `est` (*establish*), which is what lets pieces be turned back into words.

> [!definition] BPE at inference time
> Split the input into characters and **apply the merge operations in the same order they were learned**.

So a trained BPE model is an **ordered list of merges**, replayed one after another on the word's characters plus `</w>`. The same word always gets the same pieces, whatever sentence it is in.

A segmented sentence, with `@@` marking a piece that continues into the next one:

```
The president of France arrived in Kyrgyzstan .
The preside@@ nt of France arr@@ ived in K@@ yrg@@ yz@@ stan .
```

Frequent words stay whole. `preside|nt` and `arr|ived` are not morpheme boundaries, but the unseen `Kyrgyzstan` is now four known pieces, so the model can represent it at all. That is the trade: clean morphology for the ability to write down any word.

### A worked merge example

Corpus: `the tall man is taller than the tallest man`. Word types: `the` 2, `tall` 1, `man` 2, `is` 1, `taller` 1, `than` 1, `tallest` 1, each split into characters plus `</w>` (`t a l l e r </w>`).

**Count the pairs.** Six pairs tie at 3: `t+h` (2 from `the`, 1 from `than`), `t+a` (`tall`, `taller`, `tallest`; in `than` the `t` is followed by `h`), `a+l`, `l+l`, `a+n` (`man` twice, `than`) and `n+</w>`. `h+e`, `m+a` and `l+e` occur twice; `i+s`, `e+r` and `e+s` once.

**Merge 1.** Breaking ties by the pair met first in the corpus gives `t+h → th`, so `the` becomes `th e </w>` and `than` becomes `th a n </w>`.

**Merge 2.** Recount: `t+a` is now the first pair at 3, so `t+a → ta`. Continuing the same way, merges 3 to 10 are `ta+l`, `tal+l`, `a+n`, `an+</w>` (count 3 each), then `th+e`, `the+</w>`, `m+an</w>` and `tall+e` (count 2 each). The corpus ends as:

```
the</w>  tall </w>  man</w>  i s </w>  talle r </w>
th an</w>  the</w>  talle s t </w>  man</w>
```

`the` and `man` became single symbols, as intended. But morphology says `tall+er` and `tall+est`, and BPE produced `talle+r` and `talle+s+t`: merging `tall+e` was the most frequent move at that point, it buried the morpheme boundary, and BPE never reconsiders a merge.

### What BPE bought in translation

**BPE has three major advantages:** it **significantly reduces the vocabulary size** (less memory, better speed), it **results in better translation quality**, and it **significantly reduces the number of out-of-vocabulary (OOV) items**. **These benefits apply to both high- and low-resource language pairs.**

In Sennrich et al.'s English to German experiment, the best scores came from subword systems with 60K to 90K entries (character bigrams and joint BPE) over word-level systems with 300K source and 500K target entries, though not every subword system beat the best word-level one. A word-level system with a dictionary fallback for unknown words clearly beat one that outputs `<UNK>`. **Joint BPE** learns one set of merges on both languages together, so a name is split the same way on both sides and is easy to copy. Scores are on the cheat sheet.

## WordPiece: the same merges, a different score

WordPiece is only named in the lecture; this section is from outside the course material (Schuster and Nakajima, 2012, and common implementations). WordPiece builds its vocabulary like BPE, bottom-up by merging adjacent pairs, but picks **a different pair**: the one whose merge most increases the likelihood of the training data, computed as a ratio:

> [!formula] Merge criteria
> $$\text{BPE:}\ \ \arg\max_{(a,b)} \text{count}(ab) \qquad \text{WordPiece:}\ \ \arg\max_{(a,b)} \frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$$
>
> where:
> - $\text{count}(ab)$ is how often symbol $a$ is directly followed by symbol $b$
> - $\text{count}(a)$, $\text{count}(b)$ are how often each symbol occurs in total

The denominator is the whole difference. Two individually common symbols, like `t` and `h`, score low even when `th` is common, because seeing them together is no surprise. WordPiece prefers pairs whose parts **rarely occur apart** (the idea behind pointwise mutual information).

> [!example] First merge on the same corpus
> Symbol counts in `the tall man is taller than the tallest man`: `t` 7, `a` 6, `l` 6, `e` 4, `h` 3, `n` 3, `m` 2, `s` 2, `i` 1, `r` 1, `</w>` 9.
>
> | pair | count | WordPiece score |
> |---|---|---|
> | `t+h` | 3 | 3 / (7 × 3) = 0.14 |
> | `a+n` | 3 | 3 / (6 × 3) = 0.17 |
> | `l+l` | 3 | 3 / (6 × 6) = 0.08 |
> | `e+r` | 1 | 1 / (4 × 1) = 0.25 |
> | `i+s` | 1 | 1 / (1 × 2) = 0.50 |
>
> **BPE** merges `t+h` (highest count, first of the six tied at 3). **WordPiece** merges `i+s`: the pair occurs only once, but `i` never occurs anywhere else. Its second merge is `e+r` (0.25 after the recount), so WordPiece reaches the suffix `er` before it builds anything like `tall`.

These counts use BPE's `</w>` convention so the two can be compared. Real WordPiece differs in two more ways:

- WordPiece marks **continuation**: `playing` becomes `play ##ing`, where `##` means "attaches to the piece before". BPE as given here marks the **end of a word** with `</w>`.
- WordPiece keeps only the final vocabulary and segments a new word by repeatedly taking the **longest vocabulary entry that matches from the left**. BPE replays its ordered merges.

WordPiece is BERT's tokenizer. Like BPE, it merges greedily and gives each word one segmentation.

## What BPE cannot do, and Unigram LM

BPE has two structural limitations:

- **greedy**: once a character sequence has been merged, it stays that way, so a locally best merge can block a better segmentation, as `tall+e` did;
- **deterministic**: a word is always segmented the same way, so there is nothing to sample and no second-best option.

**Unigram LM** (Kudo, 2018) addresses both. It is part of Google's **SentencePiece**, whose other part is BPE.

> [!definition] Unigram LM
> - **considers all possible segmentations** of a word
> - **optimizes for the best global segmentation**
> - **supports multiple segmentations** (allowing sampling and n-best lists)
> - is **top-down**: it **starts with an existing large vocabulary that is shrunk to the desired size**

Top-down is the mirror image of BPE: BPE merges upward from characters until the vocabulary is big enough; Unigram LM prunes downward from far too many candidates until it is small enough.

### The model

A unigram language model gives each subword $x$ in vocabulary $V$ a probability $p(x)$ and treats the pieces of a segmentation as independent:

$$P(\mathbf{x}) = \prod_{i=1}^{M} p(x_i), \qquad \sum_{x \in V} p(x) = 1, \qquad \mathbf{x}^{\ast} = \arg\max_{\mathbf{x} \in S(X)} P(\mathbf{x})$$

where:
- $\mathbf{x} = (x_1, \dots, x_M)$ is one segmentation of the string $X$ into $M$ subwords
- $p(x_i)$ is the probability of subword $x_i$
- $S(X)$ is the set of all segmentations of $X$, and $\mathbf{x}^{\ast}$ is the most probable one

This is what "global" means: BPE replays local decisions, while Unigram LM scores every complete segmentation and takes the best. Each extra piece multiplies in another probability below 1, so few probable pieces beat many.

### Building the vocabulary by pruning

```pseudo
Algorithm 2: Unigram LM (Kudo, 2018), as given in MNLP
──────────────────────────────────────────────────────
Input: set of strings D, target vocab size k
V ← all substrings occurring more than once in D (not crossing words)
while |V| > k:                                     // prune tokens
    fit unigram LM θ to D
    for t ∈ V:                                     // estimate token 'loss'
        L_t ← p_θ(D) − p_θ'(D),  θ' = the LM without token t
    remove min(|V| − k, ⌊α|V|⌋) of the tokens t with highest L_t from V
        (α ∈ [0,1] is a hyperparameter)
fit final unigram LM θ to D
return V, θ
```

Reading it:

- **The start** is every substring that occurs more than once within a word: typically millions of candidates.
- **Fitting** uses EM (expectation maximisation), because the segmentation of each string is hidden: expected counts of each subword over all segmentations, then renormalise them into probabilities.
- **$L_t$** is how much the corpus likelihood drops if token $t$ is removed and the strings that used it are re-segmented. A token a cheap split can replace has a small $L_t$; one nothing else can express has a large one.
- **Pruning is gradual**: at most a fraction $\alpha$ goes per round, then the model is refitted, because each $L_t$ assumed all other tokens stay.

> [!warning] Which tokens are removed
> As given in the course, the algorithm removes the tokens with the **highest** $L_t$. Removing a token cannot raise the likelihood, so $L_t \ge 0$, and a high $L_t$ marks a **valuable** token. Kudo's method, and SentencePiece, do the opposite: they **remove the tokens with the smallest $L_t$**, the ones the corpus misses least. Kudo also **never removes single characters**, so every string can still be segmented and pruning never creates unknown symbols.

## Finding the best segmentation: dynamic programming

Unigram LM needs the most probable segmentation of each word, and trying them all is hopeless:

> [!formula] Number of segmentations
> A string of $n$ characters has $2^{n-1}$ possible segmentations: each of the $n-1$ gaps between characters is either a cut or not.

The **brute-force approach** generates all of them, scores each and picks the best: exponential in word length. Any approach needs two ingredients: **scores for segments** (units that are not split further) and a way to **combine the scores of sub-segmentations**. In the unigram model these are $p(x)$ and multiplication.

**Sequence cutting** is the same problem in the abstract: cut a sequence into segments to maximise total value. If pieces of length 1, 2, 3 and 4 are worth 1, 5, 8 and 9, a length-4 sequence is best cut `2+2` (value 10): it beats leaving it whole (9), `1+3` (9), any three pieces (7) and four single pieces (4). Neither extreme wins, and only comparing whole cuttings finds the best one.

**Dynamic programming** avoids the exponential blow-up:

> [!definition] Dynamic programming
> - **Solve each sub-problem only once.**
> - **Store the solution to each sub-problem.**
>
> It uses extra memory to save time, a **time-memory trade-off**. Two strategies: **top-down with memoization** (recursion that caches each result) and **bottom-up** (fill a table from small sub-problems to large). **Both have the same asymptotic run time.**

The segmentation algorithm is bottom-up. For a word with characters $c[1..n]$, cell $m[i,j]$ holds the score of the best segmentation of the span $c[i..j]$, and $s[i,j]$ records where that best segmentation splits the span (unset if the span is best kept whole):

```pseudo
Input: word w (character string), probabilities p
Output: segmentation s and subsegment costs
n = length(w)
let m[1..n,1..n] and s[1..n,1..n] be new tables
let c[1..n] be the characters of w
for i = 1 to n
    m[i,i] = p(c[i])                  // single characters
for l = 2 to n                        // span length, short to long
    for i = 1 to n-l+1
        j = i+l-1
        m[i,j] = p(c[i:j])            // option 1: keep the span whole (0 if not in V)
        for k = i to j-1              // option 2: split after position k
            t = m[i,k] * m[k+1,j]
            if t > m[i,j]
                m[i,j] = t
                s[i,j] = k
return m and s
```

To read the answer out: if $s[1,n]$ is unset, the word is one piece; otherwise split at $k = s[1,n]$ and do the same for both halves.

Combining the best scores of the two halves is valid **only because segments are independent** in the unigram model: the best cut of the left half does not depend on the right half. Filling by increasing length means both halves are finished before they are used. $O(n^2)$ cells with up to $n$ split points each give $O(n^3)$ time and $O(n^2)$ space. Real implementations add log probabilities, because long products underflow to zero.

> [!example] Segmenting `cats`
> Say $p(\texttt{cats}) = .0001$, $p(\texttt{cat}) = .003$, $p(\texttt{s}) = .05$. The whole-word cell starts at .0001; the split after `cat` scores $.003 \times .05 = .00015$ and beats it and every other split, so the answer is **`cat | s`**. Two common pieces beat a rare whole word, which BPE cannot do once it has merged `cats`, and the morphologically right split appears only because plural `s` is frequent. The full table is in the exam questions.

### More than one segmentation

**The Viterbi segmentation** (the search above) **only returns the most probable segmentation.** For more, **Yen's and Eppstein's algorithms give n-best segmentations**, or you can **disallow certain sub-solutions** (forbid the winning split and search again). Because every segmentation has a probability, training can also sample a different segmentation each time a word is seen, so the model does not overfit to one arbitrary split (Kudo's subword regularization). BPE, being deterministic, has nothing to sample from.

## SentencePiece: segmenting raw text losslessly

**Standard BPE implementations assume word boundary detection**: something has already split the text into words. **SentencePiece instead treats the input as a raw stream of Unicode characters, including spaces**, and **the space character is escaped as a visible symbol**, `▁`.

Why that matters shows when you undo a segmentation:

```
Raw text:   Hello world.
Tokenized:  [Hello] [world] [.]
```

Should this become `Hello world .`, `Helloworld.` or `Hello world.`? The tokens do not record where the spaces were, so a detokenizer guesses with per-language rules. **Languages without clear word boundaries** make it worse: Japanese `[こんにちは] [世界] [。]` ("Hello world.") must be joined with no spaces, the opposite of the English rule.

With the space kept as a symbol:

```
Raw text:   Hello▁world.
Tokenized:  [Hello] [▁wor] [ld] [.]
```

To desegment, join the pieces and turn `▁` back into a space. No rules or language knowledge are needed, and the original text comes back exactly, which is what **lossless** means. No language-specific pre-tokenizer is needed either.

> [!warning] SentencePiece is a toolkit
> SentencePiece implements **both** BPE and Unigram LM. What belongs to SentencePiece itself is the input handling: a raw stream including spaces, `▁` for the space, lossless desegmentation. "Trained with SentencePiece" does not say which algorithm was used.

## BPE against Unigram LM in practice

Bostrom and Durrett (2020) trained both on the same data at the same vocabulary size, then pretrained identical models on each, so any difference comes from the tokenizer. Their examples (`▁` marks a word-initial piece):

| word | BPE | Unigram LM |
|---|---|---|
| furiously | `▁fur iously` | `▁fur ious ly` |
| tricycles | `▁t ric y cles` | `▁tri cycle s` |
| nanotechnology | `▁n an ote chn ology` | `▁nano technology` |
| corrupted | `▁cor rupted` | `▁corrupt ed` |
| 1848 and 1852, | `▁184 8 ▁and ▁185 2,` | `▁1848 ▁and ▁1852 ,` |

Unigram LM finds real morpheme boundaries (`corrupt|ed`, `cycle|s`, `ious|ly`); BPE produces frequency artefacts, has no symbol for `nano` (its merges went to generic letter clusters), and splits years into a meaningless prefix plus a digit. In Japanese, BPE glues grammatical particles onto content words (は onto 性, に onto 分類); Unigram LM keeps them apart:

```
Original     磁性は様々に分類がなされている。   ("Magnetism is classified in various ways.")
BPE          磁 | 性は | 様々 | に分類 | がなされている | 。
Unigram LM   磁 | 性 | は | 様々 | に | 分類 | がなされている | 。
```

How the two vocabularies differ at the same size:

- **The unigram LM produces longer segments on average**, in English and in Japanese. Both keep all single characters; BPE has more entries of length 3 to 5, Unigram LM more of length 7 and up.
- **The unigram LM uses its vocabulary space more effectively, with more tokens of moderate frequency.** Ranked by use, the two look alike until the last few thousand entries, where BPE's frequencies collapse: BPE fills the bottom of its vocabulary with near-unused tokens, each still costing an embedding.
- **Each over-produces something telling.** BPE has more **word-initial single capitals** (`▁H`, `▁K`), leftovers from capitalised words it could not merge. Unigram LM has more **suffixes and punctuation** (`s`, `ed`, `ing`, `ly`, `,`).
- **Unigram LM needs slightly fewer tokens per word** in English, a difference of about 2%.

**Against gold-standard segmentations** (CELEX2, a database of English morphology; MeCab, a standard Japanese segmenter), Unigram LM beats BPE on precision, recall and F1 in both languages, by far more in English. English recall is low for both, because neither looks for morphemes and frequent words stay whole (`walked` misses `walk|ed`).

**Downstream**, with only the tokenizer changed, Unigram LM wins every task: by about a point on English (SQuAD question answering, MNLI inference, CoNLL named-entity recognition), by more than ten on **Japanese TyDi QA** (question answering). So the larger gold-standard gap (English) did not predict the larger downstream gap (Japanese): the tokenizer is nearly invisible in English benchmarks and can dominate outside them. Figures are on the cheat sheet.

## The unknown words that remain: byte-level BPE

**Handling low-frequency words (including zero-frequency, that is, OOV) is one of the main motivations for subword segmentation.** **BPE and Unigram have very low OOV rates**, but not zero:

- **typos can still sometimes cause OOVs**;
- **unknown characters will result in OOVs**: the base vocabulary holds only characters seen in training, so an unseen emoji or script has nothing to fall back to.

> [!definition] Byte-level BPE: guaranteeing zero OOV rates
> - **operate on raw UTF-8 bytes instead of Unicode characters**
> - **the base alphabet is fixed at 256 symbols**, so any input, in any script, can always be represented
> - **even unseen scripts can be encoded**, byte by byte, and then merged where patterns emerge

From outside the course material: GPT-2 and many later decoder-only models use byte-level BPE.

The cost lands on non-Latin scripts. UTF-8 uses one byte for an ASCII character, two for Latin letters with diacritics, Cyrillic or Greek, three for most Chinese, Japanese, Devanagari and Thai, and four for many emoji. So **non-Latin, non-ASCII scripts cost more bytes per character**, and **additional merge operations are required just to go from bytes back to characters, resulting in shorter subsegments** for those scripts. The cause is UTF-8's encoding length, nothing about the language itself.

## Fertility: what segmentation costs a language

> [!definition] Fertility
> **Fertility measures the average number of segments a word is split into:**
> $$\text{fertility} = \frac{\text{number of subword tokens}}{\text{number of words}}$$
> A fertility of 1 means every word is a single token; higher means more pieces per word.

**Fertility for a language is influenced by:**

- the **maximum number of merge operations** (BPE) or **target vocabulary size** (Unigram LM): a bigger vocabulary means fewer splits;
- the **average length of the surface words** in the language;
- **morphological richness**: how many combinations of affixes are possible.

**Isolating, Latin-script languages typically see fertility close to 1 for common vocabulary words** (an isolating language has about one morpheme per word; see [[MNLP-L03 - Morphology and Word Formation|L03]]). Agglutinative languages such as Finnish or Turkish string many affixes into one word, so they have far more distinct, rarer word forms, each split further.

High fertility costs a language in four ways:

- **the model has to work harder** to reassemble a word's meaning from many small, less meaningful parts;
- **no single tokenization scheme dominates across all languages and scripts**, so a shared vocabulary is a compromise for each language;
- **commercial LLM APIs price by token count**, so **high-fertility languages pay more for the same amount of content**;
- **fixed context windows fill up faster** for high-fertility languages.

## The methods side by side

| | BPE | WordPiece | Unigram LM | byte-level BPE |
|---|---|---|---|---|
| builds vocabulary | bottom-up, merges | bottom-up, merges | top-down, prunes | bottom-up, merges |
| picks by | pair count | count ratio | likelihood loss | pair count |
| per word | one segmentation | one segmentation | best, n-best or sampled | one segmentation |
| starts from | characters | characters | all frequent substrings | 256 bytes |
| OOV left | unseen characters | unseen characters | unseen characters | none |

**Morfessor** came first: segments chosen by MDL (model cost plus data cost), no annotation. **SentencePiece** is the toolkit that runs BPE or Unigram LM on raw text with `▁` for the space.

## What to be able to answer

- Why word and character vocabularies fail; FST weaknesses; MDL's two costs; the three simplifications and two gains (exam question on why subwords).
- BPE training and inference, with a merge trace by hand (exam question on the `tall` corpus).
- The WordPiece score: why it merges `i+s` where BPE merges `t+h` (exam question).
- Unigram LM: the model, top-down pruning, which tokens go, single characters kept (exam question).
- DP segmentation: $2^{n-1}$, the table, why independence makes it valid (exam question on `cats`).
- SentencePiece: raw stream, `▁`, lossless, a toolkit for both algorithms.
- Bostrom and Durrett: the direction of each comparison, and where the gap is large (exam question).
- Byte-level BPE and fertility: who pays, and why (exam question).

## Exam questions

> [!exam]- Compare BPE and Unigram LM as subword segmentation methods: mechanism, segmentation quality and downstream effect.
> **Key points:** BPE bottom-up, greedy, one segmentation per word; Unigram LM top-down likelihood pruning, global best, n-best and sampling; Unigram LM closer to gold segmentations, far more so in English; downstream about a point in English, much larger on Japanese TyDi QA; SentencePiece is the toolkit that implements both.
>
> 1. **Mechanism.** BPE (Sennrich et al., 2016) is **bottom-up**: start from characters and greedily merge the most frequent adjacent pair until the merge budget is spent. A merge is never undone, and each word has exactly one segmentation. Unigram LM (Kudo, 2018) is **top-down**: start from every substring occurring more than once, fit a unigram LM by EM, and prune the tokens whose removal costs the least likelihood until the target size $k$ is reached. It scores whole segmentations, $P(\mathbf{x}) = \prod_i p(x_i)$, takes the global best by dynamic programming, and can return n-best or sampled segmentations.
> 2. **Segmentation quality** (Bostrom and Durrett, 2020, same data and same vocabulary size): Unigram LM has higher precision, recall and F1 than BPE against CELEX2 (English morphology) and MeCab (Japanese word segmentation), with a far larger gap in English. Unigram LM recovers suffixes (`ly`, `ed`, `s`); BPE over-produces word-initial single capitals and frequency artefacts such as `n|an|ote|chn|ology`.
> 3. **Vocabulary use:** Unigram LM produces longer segments on average and uses its vocabulary more effectively, with more tokens of moderate frequency; BPE spends the bottom of its vocabulary on near-unused tokens.
> 4. **Downstream** (identical pretraining and model, only the tokenizer differs): Unigram LM wins the English tasks by about a point and Japanese TyDi QA by more than ten. The larger gold-standard gap (English) does not predict the larger downstream gap (Japanese).
>
> Losing marks: calling SentencePiece the algorithm (it is the toolkit that implements both); reading the low absolute English recall as "tokenizers fail at English" when the point is the comparison between the methods.

> [!exam]- Why does neural machine translation need subword segmentation? Argue from the weaknesses of word-level, character-level and rule-based (FST) alternatives.
> **Key points:** vocabulary bottleneck, representation depends on frequency and contexts; word-level: huge vocabularies, no sharing, unseen words become `<UNK>`; character-level: meaningless units, long sequences; FSTs: laborious, language dependent, no preference among ambiguities; subwords: 30K to 128K, near-zero OOV, learned from frequencies.
>
> - The vocabulary is one of the main bottlenecks of training an NMT system. How well a word is represented depends on its **frequency** and on the **number of different contexts** it occurs in, both properties of the training corpus.
> - Many word occurrences come from **word formation**: inflection (`tall`, `taller`, `tallest`) and compounding (`Donaudampfschifffahrtsgesellschaft`). A word-level vocabulary treats `tall` and `tallest` as unrelated entries.
> - **Word-level:** 300K to 500K entries and still OOVs, huge embedding and output matrices, every unseen word collapses to `<UNK>` ("Kyrgyzstan" becomes an unspecified place).
> - **Character-level:** near-zero OOV, but each symbol carries almost no meaning and sequences are long and slow.
> - **Hand-built FSTs:** accurate and efficient, but extremely laborious, totally language dependent, and with no preference among ambiguous analyses, so they do not scale to many languages.
> - **Subwords** (typically 30K to 128K merges): learned from frequencies with no annotation, vocabulary size tunable, near-zero OOV, moderate sequence length. The price is that pieces are not clean morphemes.

> [!exam]- Train BPE on the corpus `the tall man is taller than the tallest man` for 10 merges, breaking ties by first occurrence. Give the merges and comment on the result.
> **Key points:** six-way tie at 3, `t+h` met first; `than` does not count towards `t+a`; merge order ending in `tall+e`; final `talle r` and `talle s t`; the greedy merge destroys `tall|er` and is never undone.
>
> 1. Word types: `the` 2, `tall` 1, `man` 2, `is` 1, `taller` 1, `than` 1, `tallest` 1. Split each into characters and append `</w>`.
> 2. Initial counts: six pairs tie at 3: `t+h` (2 from `the`, 1 from `than`), `t+a`, `a+l`, `l+l`, `a+n`, `n+</w>`. `t+h` is met first, so it wins.
> 3. Merges in order: `t+h` (3), `t+a` (3), `ta+l` (3), `tal+l` (3), `a+n` (3), `an+</w>` (3), `th+e` (2), `the+</w>` (2), `m+an</w>` (2), `tall+e` (2).
> 4. Final corpus: `the</w>`, `tall </w>`, `man</w>`, `i s </w>`, `talle r </w>`, `th an</w>`, `the</w>`, `talle s t </w>`, `man</w>`.
> 5. Comment: the frequent words `the` and `man` became single tokens, as desired. But `taller` and `tallest` ended as `talle + r` and `talle + s + t` where the morphological split is `tall + er` and `tall + est`. The `tall + e` merge was locally the most frequent move and destroyed the morpheme boundary, and BPE never undoes a merge.
>
> Losing marks: counting `than` towards `t+a` (in `than` the `t` is followed by `h`).

> [!exam]- On the same corpus, which pair does WordPiece merge first, and how does that differ from BPE? Show the computation.
> **Key points:** WordPiece score count(ab) / (count(a) count(b)); symbol counts `t` 7, `h` 3, `i` 1, `s` 2, `e` 4, `r` 1; `t+h` scores 3/21, `i+s` scores 1/2; WordPiece merges `i+s`, BPE merges `t+h`; WordPiece prefers pairs whose parts rarely occur apart.
>
> 1. **BPE** merges the pair with the highest raw count. Six pairs tie at 3, and `t+h` is met first.
> 2. **WordPiece** merges the pair with the highest $\frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$. Symbol counts: `t` 7, `a` 6, `l` 6, `e` 4, `h` 3, `n` 3, `m` 2, `s` 2, `i` 1, `r` 1, `</w>` 9.
> 3. Scores: `t+h` $3/(7 \cdot 3) \approx 0.14$; `a+n` $3/(6 \cdot 3) \approx 0.17$; `l+l` $3/36 \approx 0.08$; `e+r` $1/(4 \cdot 1) = 0.25$; `i+s` $1/(1 \cdot 2) = 0.5$. WordPiece merges **`i+s`**, then `e+r`.
> 4. Comment: `i+s` occurs once, but `i` occurs nowhere else, so the pair is maximally "surprising". `t` and `h` are each common, so `th` scores low despite its count. WordPiece therefore reaches the suffix `er` before building `tall`; both methods remain greedy, bottom-up and deterministic.
>
> (WordPiece is only named in MNLP; the score is how common implementations compute its likelihood criterion.)

> [!exam]- Segment `cats` by bottom-up DP with p(c)=.02, p(a)=.04, p(t)=.03, p(s)=.05, p(ca)=.002, p(at)=.0015, p(ts)=.0004, p(cat)=.003, p(ats)=.00005, p(cats)=.0001.
> **Key points:** each cell initialised with the unsplit probability; `ts` and `ats` split, `cat` stays whole; $s[1,4]=3$ gives `cat | s`, $P = 1.5\times10^{-4}$; a rare whole word loses to common pieces; valid because segments are independent.
>
> 1. **Length 1:** $m[i,i]$ is the character probability: 0.02, 0.04, 0.03, 0.05.
> 2. **Length 2:** `ca` stays whole, 0.002 (split gives 0.0008); `at` stays whole, 0.0015 (split 0.0012); `ts` splits, $0.03 \times 0.05 = 0.0015 > 0.0004$, so $s[3,4]=3$.
> 3. **Length 3:** `cat` stays whole, 0.003 (splits give 0.00003 and 0.00006); `ats` is best as `at|s`, $0.0015 \times 0.05 = 0.000075$, beating unsplit 0.00005 and `a|ts` 0.00006, so $s[2,4]=3$.
> 4. **Length 4:** unsplit `cats` 0.0001; $k=1$: $0.02 \times 0.000075 = 1.5\times10^{-6}$; $k=2$: $0.002 \times 0.0015 = 3\times10^{-6}$; $k=3$: $0.003 \times 0.05 = 1.5\times10^{-4}$. Winner $k=3$, $s[1,4]=3$.
> 5. **Read back:** $s[1,4]=3$ splits into $c[1..3]$ and $c[4]$; $s[1,3]$ is unset, so the result is `cat | s` with $P = 1.5\times10^{-4}$.
> 6. **Points to make:** the rare whole word `cats` loses to two common pieces, which BPE cannot do once it has merged `cats`. The morphologically correct split appears without any morphology in the model, because plural `s` is frequent. Combining the best scores of two halves is valid only because the unigram model treats segments as independent.

> [!exam]- Explain fertility and why one subword vocabulary serves languages unequally. Include byte-level BPE in your answer.
> **Key points:** fertility = subword tokens / words; drivers: vocabulary size, word length, morphological richness; four consequences: reassembly, no universal scheme, API cost, context; byte-level BPE: zero OOV but multi-byte scripts get shorter subsegments, because of UTF-8 encoding length.
>
> - **Fertility** = subword tokens / words, the average number of segments per word. 1 means every word is one token.
> - **Drivers:** merge count or target vocabulary size (the one you control), average surface word length, morphological richness. Isolating Latin-script languages are close to 1 for common words; agglutinative languages (Finnish, Turkish) have many more, individually rarer word forms, which get split further.
> - **Consequences:** (1) the model must reassemble meaning from many small parts; (2) no tokenization scheme dominates across all languages and scripts, so a shared vocabulary is a compromise for each language; (3) commercial APIs price by token, so high-fertility languages pay more for the same content; (4) fixed context windows fill up faster.
> - **Byte-level BPE:** the 256-byte base alphabet guarantees zero OOV, but UTF-8 uses 1 byte for ASCII and 2 to 4 for other scripts (Cyrillic, CJK, Devanagari, emoji). Those scripts spend merges just getting from bytes back to characters and end up with shorter subsegments. The cause is encoding length, nothing about the language itself.

> [!exam]- Walk through Kudo's (2018) Unigram LM vocabulary construction and correct the pruning rule in the version given in MNLP.
> **Key points:** seed with all substrings occurring more than once, not crossing words; fit unigram LM by EM, segmentation latent; $L_t$ = likelihood drop without $t$; prune at most a fraction $\alpha$ per round, refit; correction: remove the smallest $L_t$, never prune single characters.
>
> 1. Seed $V$ with all substrings that occur more than once in $D$ and do not cross word boundaries (millions of candidates: the top-down start).
> 2. While $|V| > k$: fit a unigram LM $\theta$ to $D$ by EM (segmentation is hidden: expected subword counts over all segmentations, then renormalise into $p(x)$).
> 3. For each token $t$: $L_t = p_\theta(D) - p_{\theta'}(D)$, where $\theta'$ is the LM without $t$ and strings that used $t$ are re-segmented.
> 4. Remove $\min(|V|-k, \lfloor\alpha|V|\rfloor)$ tokens, $\alpha \in [0,1]$ a hyperparameter, then refit. Pruning is gradual because each $L_t$ assumes all other tokens are still present.
> 5. Fit the final unigram LM and return $V$ and $\theta$.
>
> **The error:** the version given in MNLP removes the tokens with the **highest** $L_t$. Removing a token cannot raise the likelihood, so $L_t \ge 0$ and a large $L_t$ marks a valuable token: taken literally the rule prunes the most useful pieces first. **Correct rule:** remove the tokens with the **smallest** $L_t$, which is Kudo's method and what SentencePiece implements. Kudo also never prunes **single-character tokens**, so every string stays segmentable.

## Flashcards

> [!card]- Which two properties of the training data determine how well a word is represented in NMT?
> Its **frequency** and the **number of different contexts** it occurs in.

> [!card]- Which two word-formation processes produce many of the word occurrences a fixed word vocabulary handles badly?
> **Inflection** (`tall`, `taller`, `tallest`) and **compounding**.

> [!card]- What two kinds of unit can a subword be?
> A **character n-gram**, or a **morphologically meaningful unit** (language dependent).

> [!card]- What happens to an unseen word under a word-level NMT vocabulary?
> It collapses to **`<UNK>`**; related forms like `tall` and `tallest` share nothing either.

> [!card]- Why is a character-level vocabulary a poor fix for OOV words, even though its OOV rate is near zero?
> Each symbol carries almost **no meaning**, and sequences become **long and slow**.

> [!card]- What is the typical size range of a BPE vocabulary today?
> About **30K to 128K** merge operations.

> [!card]- Why do multilingual models sit at the 128K end of the BPE range?
> A vocabulary shared by **many scripts** needs more room.

> [!card]- What are the three weaknesses of hand-built FST morphological analyzers?
> Extremely **laborious** to produce, totally **language dependent**, **no preference** among ambiguous analyses.

> [!card]- Why are hand-built FST analyzers ruled out for multilingual NLP even though they are accurate?
> They are **totally language dependent**: one expert-built analyzer per language does not scale.

> [!card]- Which data-driven morphological segmenter is based on the Minimum Description Length principle?
> **Morfessor** (Creutz and Lagus).

> [!card]- Minimum Description Length minimises the sum of which two costs?
> The cost of the **model itself** plus the cost for the model to **generate the data**.

> [!card]- Why does Minimum Description Length push segmentation towards morpheme-like pieces?
> Whole words make the model huge, characters make the data expensive; the optimum is **reusable pieces**.

> [!card]- Which three simplifications do current data-driven subword segmenters make?
> No **labelling** (a split only), no character **substitutions or deletions**, strictly **concatenative**.

> [!card]- What two practical properties do data-driven subword segmenters gain from their simplifications?
> **No annotation or linguistic knowledge** needed, and they can be **fitted to a desired vocabulary size**.

> [!card]- How do BPE and Unigram LM differ in the direction of vocabulary construction?
> BPE starts from characters and **merges upward**; Unigram LM starts large and **prunes downward** to size $k$.

> [!card]- Which pair does BPE merge at each training step?
> The **most frequent** adjacent symbol pair, counted over word types weighted by frequency.

> [!card]- What is BPE's only hyperparameter?
> The **number of merge operations**, which sets the final vocabulary size.

> [!card]- Why is BPE training cheap?
> It runs over the **word types with their frequencies**, never over the whole corpus.

> [!card]- What does BPE's end-of-word marker `</w>` let it distinguish?
> **Word-final** pieces from word-internal ones, so pieces can be turned back into words.

> [!card]- Is the `</w>` marker what stops BPE from merging pairs across two words?
> **No.** Each word is counted on its own, so no pair ever spans two words.

> [!card]- What does a trained BPE model consist of?
> An **ordered list of merge operations**.

> [!card]- How is a trained BPE model applied to a new word?
> Split it into characters and apply the merges **in the order they were learned**.

> [!card]- When counting BPE pairs in `the tall man is taller than the tallest man`, does `than` count towards `t+a`?
> **No.** In `than` the `t` is followed by `h`.

> [!card]- Which pair does BPE merge first on `the tall man is taller than the tallest man`, breaking ties by first occurrence?
> **`t+h`**: one of six pairs at count 3, and met first.

> [!card]- Why does BPE trained on `the tall man is taller than the tallest man` end with `talle + r` where morphology says `tall + er`?
> The greedy `tall + e` merge was locally most frequent and **buried the boundary**; BPE never undoes a merge.

> [!card]- What are the three major advantages of BPE reported by Sennrich et al.?
> Smaller **vocabulary**, better **translation quality**, far fewer **OOV items**.

> [!card]- In Sennrich et al.'s English to German NMT, how did 300K/500K word-level systems compare with 60K to 90K subword systems?
> The best scores came from **subword** systems, despite their much **smaller vocabularies**.

> [!card]- Why does joint BPE, trained on both languages together, help translation?
> A string is **segmented identically on both sides**, which makes names easy to **copy**.

> [!card]- Which segmentation method merges the pair maximising $\frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$?
> **WordPiece**.

> [!card]- What is the difference between BPE's and WordPiece's choice of which pair to merge?
> BPE takes the highest **raw count**; WordPiece divides by the parts' counts, preferring parts that **rarely occur apart**.

> [!card]- Why would WordPiece merge `t + h` less eagerly than BPE does?
> `t` and `h` are each common, so the **denominator** shrinks the score of `th`.

> [!card]- Which pair does WordPiece merge first on `the tall man is taller than the tallest man`?
> **`i+s`**: it occurs once, but `i` occurs nowhere else, giving the top score.

> [!card]- How does WordPiece segment a new word at inference?
> By repeatedly taking the **longest vocabulary entry** matching from the left.

> [!card]- Which well-known encoder model uses WordPiece as its tokenizer?
> **BERT**.

> [!card]- What is the difference between WordPiece's `##` and BPE's `</w>` markers?
> WordPiece marks **continuation** (`##ing` attaches left); BPE marks **word end**.

> [!card]- What are BPE's two structural limitations?
> **Greedy** (a merge is never undone) and **deterministic** (one segmentation per word).

> [!card]- Why is BPE's greediness a problem for segmentation quality?
> A **locally best** merge can block a **better** segmentation, and it is never undone.

> [!card]- Why can BPE not sample alternative segmentations of a word?
> It is **deterministic**: one segmentation per word, so there is **no distribution** to sample from.

> [!card]- Is SentencePiece a segmentation algorithm?
> **No.** It is Google's **toolkit**, implementing both BPE and Unigram LM.

> [!card]- What are the four properties of Unigram LM (Kudo)?
> Considers **all segmentations**, optimises the **best global** one, supports **multiple segmentations**, **top-down**.

> [!card]- How does Unigram LM score a segmentation $\mathbf{x} = (x_1, \dots, x_M)$?
> $P(\mathbf{x}) = \prod_i p(x_i)$: subwords are assumed **independent**.

> [!card]- How does Unigram LM choose the segmentation of a string $X$?
> The $\arg\max$ of $P(\mathbf{x})$ over **all segmentations** $S(X)$: a global choice.

> [!card]- Which method starts from all substrings occurring more than once and prunes them to the target size?
> **Unigram LM**.

> [!card]- How is the unigram LM fitted during Unigram LM vocabulary construction?
> By **EM**, with each string's segmentation as a **hidden variable**.

> [!card]- What does the token loss $L_t$ measure in Unigram LM vocabulary construction?
> How much **corpus likelihood drops** if token $t$ is removed and its strings are re-segmented.

> [!card]- When pruning a Unigram LM vocabulary, should the tokens with the highest or the smallest loss $L_t$ be removed?
> The **smallest** $L_t$. The version given in MNLP says highest, which would prune the most valuable tokens.

> [!card]- Why does Unigram LM prune at most a fraction $\alpha$ of the vocabulary per round?
> Each $L_t$ assumes all other tokens stay, so the model is **refitted** between rounds.

> [!card]- Why does Kudo's Unigram LM never prune single-character tokens?
> So every string stays **segmentable** and pruning cannot create **OOVs**.

> [!card]- How many segmentations does a string of $n$ characters have?
> $2^{n-1}$: each of the $n-1$ gaps is either a **cut or not**.

> [!card]- What two ingredients are needed to score candidate segmentations?
> **Scores for segments** and a way to **combine** them; in the unigram model, $p(x)$ and multiplication.

> [!card]- Why can dynamic programming find the best Unigram LM segmentation?
> Segments are **independent**, so the best cut of the left part does not depend on the right part.

> [!card]- What does the sequence-cutting example show about finding the best segmentation?
> Neither extreme wins: the optimum is an **interior split**, found only by comparing **whole cuttings**.

> [!card]- What are the two principles of dynamic programming?
> Solve each sub-problem **only once** and **store** its solution: a time-memory trade-off.

> [!card]- How do top-down and bottom-up dynamic programming differ?
> Top-down is **recursion with memoization**; bottom-up **fills a table** small to large. Same asymptotic run time.

> [!card]- In bottom-up DP segmentation, what does the cell $m[i,j]$ hold?
> The score of the **best segmentation of the span** $c[i..j]$.

> [!card]- With what value is each DP cell $m[i,j]$ initialised before trying split points?
> $p(c[i..j])$, the **no-split** option; zero if the span is not in the vocabulary.

> [!card]- Why is the DP segmentation table filled by increasing span length?
> Both halves of any split are **shorter**, so their best scores are already final.

> [!card]- What is the complexity of span-table DP segmentation in word length $n$?
> $O(n^3)$ **time** and $O(n^2)$ **space**.

> [!card]- Why do implementations of DP segmentation add log probabilities instead of multiplying probabilities?
> Long products **underflow** to zero.

> [!card]- In DP segmentation of `cats`, why can `cat | s` win even though `cats` is in the vocabulary?
> $p(\text{cat}) \cdot p(\text{s})$ exceeds $p(\text{cats})$: a **rare whole word** loses to two **common pieces**.

> [!card]- How can you get more than the single Viterbi segmentation?
> **Yen's or Eppstein's** n-best algorithms, or **disallow** certain sub-solutions and search again.

> [!card]- What does Kudo's subword regularization do during training?
> **Samples** a different segmentation each time a word is seen, instead of always the best one.

> [!card]- What do standard BPE implementations assume about their input that SentencePiece does not?
> **Word boundary detection**: the text is already split into words.

> [!card]- How does SentencePiece treat its input?
> As a **raw stream of Unicode characters including spaces**, with the space escaped as `▁`.

> [!card]- What two consequences follow from SentencePiece's raw-stream input?
> **No language-specific pre-tokenizer**, and **lossless** desegmentation.

> [!card]- What is the desegmentation problem?
> Tokens like `[Hello] [world] [.]` do not record **where spaces were**, so detokenizers guess per language.

> [!card]- How is SentencePiece output desegmented?
> **Join the pieces** and turn `▁` back into a space: no rules, exactly invertible.

> [!card]- How did Bostrom and Durrett make their BPE against Unigram LM comparison controlled?
> **Same data, same vocabulary size**, identical pretraining and model; only the tokenizer differs.

> [!card]- Which of BPE and Unigram LM finds real morpheme boundaries in words like `corrupted` and `furiously`?
> **Unigram LM** (`corrupt|ed`, `ious|ly`); BPE produces **frequency artefacts**.

> [!card]- Why does BPE have no symbol for `nano` in `nanotechnology`?
> Its merges went to **generic frequent letter clusters**.

> [!card]- What goes wrong when BPE segments numbers such as years?
> It splits them like `184` + `8`, a **meaningless prefix** plus a digit.

> [!card]- What does BPE do wrong when segmenting Japanese, compared with Unigram LM?
> It **glues particles** (は, に) onto content words.

> [!card]- At the same vocabulary size, which produces longer segments on average: BPE or Unigram LM?
> **Unigram LM**, in both English and Japanese.

> [!card]- How do BPE and Unigram LM differ in how they use the bottom of their vocabulary?
> BPE fills it with **near-unused tokens**; Unigram LM keeps **more tokens of moderate frequency**.

> [!card]- Which extra tokens are characteristic of BPE versus Unigram LM in English?
> BPE: **word-initial single capitals**; Unigram LM: **suffixes** like `s`, `ed`, `ing`, `ly`.

> [!card]- Why are both tokenizers' recall scores against CELEX2 English morphology low?
> Neither looks for morphemes, and frequent words **stay whole** (`walked` misses `walk|ed`).

> [!card]- Where is Unigram LM's downstream advantage over BPE much larger: English tasks or Japanese TyDi QA?
> **Japanese TyDi QA**; on English tasks it is small.

> [!card]- Does a larger gold-standard F1 gap between tokenizers predict a larger downstream gap?
> **No.** The F1 gap is larger in English, the downstream gap larger in Japanese.

> [!card]- What is the multilingual moral of Bostrom and Durrett's tokenizer comparison?
> The tokenizer is nearly **invisible in English** benchmarks and can **dominate outside** them.

> [!card]- What two things can still cause OOVs under BPE or Unigram LM?
> **Typos** (sometimes) and **unknown characters**.

> [!card]- Why do unknown characters cause OOVs for character-based subword vocabularies?
> The base vocabulary holds only **characters seen in training**.

> [!card]- How does byte-level BPE guarantee a zero OOV rate?
> It runs on **UTF-8 bytes**; the base alphabet is fixed at **256 symbols**.

> [!card]- Which method starts from a fixed base alphabet of 256 symbols?
> **Byte-level BPE**.

> [!card]- Which segmentation variant do GPT-2 and many later decoder-only models use?
> **Byte-level BPE**.

> [!card]- Why does byte-level BPE give non-Latin scripts shorter subsegments?
> They take **more bytes per character**, so merges are spent getting **back to characters**.

> [!card]- What is the cause of byte-level BPE's shorter subsegments for scripts like Devanagari?
> **UTF-8 encoding length**, nothing about the language itself.

> [!card]- What is the formula for fertility?
> $\text{fertility} = \frac{\text{number of subword tokens}}{\text{number of words}}$; 1 means one token per word.

> [!card]- What three factors influence fertility for a language?
> **Merge count or target vocabulary size**, average **surface word length**, **morphological richness**.

> [!card]- Why do agglutinative languages like Finnish or Turkish have high fertility?
> They have many more distinct, **rarer word forms**, so each gets **split further**.

> [!card]- What are the four consequences of high fertility for a language?
> Harder **meaning reassembly**, no **universal scheme**, higher **API cost**, faster **context exhaustion**.

> [!card]- Why do high-fertility languages pay more on commercial LLM APIs?
> APIs **price by token**, and the same content takes **more tokens**.

> [!card]- True or false: Unigram LM builds its vocabulary bottom-up by merging the most likely pairs.
> **False.** It is top-down: it prunes a large seed vocabulary to the target size.

> [!card]- True or false: WordPiece merges the most frequent adjacent pair at each step.
> **False.** It divides the pair count by its parts' counts; BPE takes the most frequent pair.

> [!card]- True or false: Applying a trained BPE model replays the learned merges in the order they were learned.
> **True.** The model is an ordered merge list.

> [!card]- True or false: BPE's advantages in vocabulary size, OOV rate and translation quality apply only to low-resource language pairs.
> **False.** They apply to both high- and low-resource pairs.

> [!card]- True or false: Data-driven subword segmenters account for character substitutions such as `city`/`cities`.
> **False.** They do not account for substitutions or deletions; they are strictly concatenative.

> [!card]- True or false: Unigram LM returns only a single segmentation per word.
> **False.** It supports multiple segmentations: n-best lists and sampling.

> [!card]- True or false: In Unigram LM vocabulary construction, Kudo removes the tokens with the highest loss $L_t$.
> **False.** The smallest $L_t$; a high $L_t$ marks a valuable token.

> [!card]- True or false: Kudo's Unigram LM can prune single-character tokens.
> **False.** Single characters are never pruned, so every string stays segmentable.

> [!card]- True or false: DP segmentation under a unigram model is valid because segments are treated as independent.
> **True.** The best cut of one half does not depend on the other.

> [!card]- True or false: In DP segmentation, each cell is initialised with the probability of the unsplit span.
> **True.** Split points then replace it only if they score higher.

> [!card]- True or false: Top-down DP with memoization and bottom-up DP have the same asymptotic run time.
> **True.**

> [!card]- True or false: The `</w>` end-of-word marker is what prevents BPE from merging pairs across two words.
> **False.** Words are counted separately, so no pair spans two words; `</w>` marks word-final pieces.

> [!card]- True or false: SentencePiece requires the input to be split into words first.
> **False.** It treats the input as a raw stream of Unicode characters, including spaces.

> [!card]- True or false: Byte-level BPE guarantees a zero OOV rate.
> **True.** Its base alphabet is all 256 bytes.

> [!card]- True or false: Byte-level BPE gives non-Latin scripts shorter subsegments because those languages are morphologically richer.
> **False.** Because their characters take more UTF-8 bytes.

> [!card]- True or false: At the same vocabulary size, Unigram LM produces longer segments on average than BPE.
> **True.** In both English and Japanese.

> [!card]- Multiple selection: which are true of BPE? (A) It is greedy: a merge is never undone. (B) It can sample alternative segmentations of a word. (C) Its only hyperparameter is the number of merges. (D) It starts from a large vocabulary and prunes it.
> **A and C.** B: BPE is deterministic. D: that is Unigram LM; BPE merges up from characters.

> [!card]- Multiple selection: which are true of WordPiece? (A) It builds its vocabulary bottom-up by merging pairs. (B) It merges the pair with the highest raw count. (C) It prefers pairs whose parts rarely occur apart. (D) It prunes a large seed vocabulary.
> **A and C.** B is BPE's criterion. D is Unigram LM.

> [!card]- Multiple selection: which are true of Unigram LM? (A) It scores a segmentation as a product of subword probabilities. (B) It is deterministic, like BPE. (C) It fits its model by EM. (D) It removes the tokens whose removal costs the most likelihood.
> **A and C.** B: it supports n-best and sampled segmentations. D: it removes those costing the least.

> [!card]- Multiple selection: which are true of SentencePiece? (A) It is an algorithm distinct from BPE and Unigram LM. (B) It escapes the space as a visible symbol `▁`. (C) It requires a language-specific pre-tokenizer. (D) Its segmentation is lossless.
> **B and D.** A: it is a toolkit implementing both. C: raw-stream input removes the pre-tokenizer.

> [!card]- Multiple selection: which are true of byte-level BPE? (A) Its base alphabet has 256 symbols. (B) It cannot encode a script unseen in training. (C) Scripts with more bytes per character get shorter subsegments. (D) It operates on Unicode characters.
> **A and C.** B: any script can be encoded byte by byte. D: it operates on raw UTF-8 bytes.

> [!card]- Multiple selection: which are true of fertility? (A) It is subword tokens divided by words. (B) It depends only on the vocabulary size. (C) High fertility fills fixed context windows faster. (D) Agglutinative languages tend to have lower fertility than English.
> **A and C.** B: word length and morphological richness matter too. D: their many rare word forms get split further.

> [!card]- Multiple selection: which are true of Bostrom and Durrett's BPE against Unigram LM comparison? (A) Unigram LM wins English tasks by a large margin. (B) Unigram LM's advantage is much larger on Japanese TyDi QA. (C) Both tokenizers had the same vocabulary size. (D) BPE recovers more English morpheme boundaries.
> **B and C.** A: about a point in English. D: Unigram LM has higher precision and recall against CELEX2.

## Links

- **Course:** [[MNLP - Overview|Course overview]]
- **Previous:** [[MNLP-L03 - Morphology and Word Formation]] (morphemes, isolating and agglutinative languages)
- **Related concept:** [[Tokenization]]
- **Project:** [[MNLP - Mini Project]]
- **Source:** `subword.pdf`, Christof Monz, MNLP: Subword Segmentation, slides 1 to 24
- **Primary references:** Sennrich, Haddow and Birch (2016), *Neural Machine Translation of Rare Words with Subword Units* · Schuster and Nakajima (2012), *Japanese and Korean Voice Search* · Kudo (2018), *Subword Regularization* · Kudo and Richardson (2018), *SentencePiece* · Creutz and Lagus (2002), *Unsupervised Discovery of Morphemes* (the method later released as Morfessor) · Bostrom and Durrett (2020), *Byte Pair Encoding is Suboptimal for Language Model Pretraining*
