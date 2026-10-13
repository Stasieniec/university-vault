---
type: lecture
course: 5204MNLP6Y
week: 2
lecture: 3
status: complete
topics:
  - Words, vocabularies and lexicons
  - Word embeddings and CBOW
  - Vocabulary short-lists and the <unk> token
  - Word segmentation and word boundaries
  - Forward and backward maximum matching
  - BMES character labeling with HMMs and Viterbi
  - Inflection, derivation and compounding
  - Morphemes, affixes and non-concatenative morphology
  - Morphological typology
  - Why morphology breaks word-level NLP
---

# MNLP-L03: Morphology and Word Formation

> [!abstract] In short
> A word-level model stores *move* and *moved* as two unrelated row numbers, and has nothing at all for a word it never saw, such as *hammered*. This lecture asks where words come from and what is inside them.
>
> **Where words come from:** Chinese, Japanese or Thai do not mark word boundaries, so they must be computed: greedily, by maximum matching against a dictionary, or statistically, by tagging each character Begin, Middle, End or Single with an HMM decoded by Viterbi.
>
> **What is inside them:** morphemes, combined by inflection, derivation and compounding, and packaged in four ways: isolating, agglutinative, fusional, polysynthetic. In rich languages one lemma has thousands of forms, so a fixed word vocabulary either explodes or maps much of the language to `<unk>`. The practical fix, subword tokenization, is the next lecture.
>
> The sample exam has no question on this lecture; expect statements on maximum matching, BMES tagging with HMMs, inflection against derivation, and the four morphological types.

## Vocabularies, lexicons and dictionaries

Assume for now that words are given: separated by white space (after splitting off punctuation), or produced by a **word segmentation** step, also called **tokenization**, for languages that do not mark word boundaries (Chinese, Thai). Collect every word in a tokenized corpus and you have a vocabulary.

> [!definition] Vocabulary, lexicon, dictionary
> - A **vocabulary** is the inventory of words in a language (or a corpus, or known to a person): a list of forms.
> - A **lexicon** stores the **meanings and functions** of words.
> - A **dictionary** is a written (published or digital) version of a lexicon.

Should an NLP vocabulary contain typos, rare words, all variants (*book, books, booking, booked*) or just *book*? Be language-specific, or limited in size? There is no clean answer, and these are mostly questions about morphology.

## Word meaning from context: embeddings and CBOW

Word meaning is classically defined through relations: **synonyms** (*purchase* :: *acquire*), **hyponyms** (*car* :: *vehicle*, is-a), **meronyms** (*wheel* :: *car*, part-whole), **antonyms** (*small* :: *large*). These explicit, qualitative relations need hand-crafted resources such as WordNet (a hand-built database of English word relations), which are **expensive**, **incomplete** and **language-specific**. The alternative is to learn relations automatically and quantify them: $\text{sim}(\textit{car}, \textit{vehicle}) > \text{sim}(\textit{car}, \textit{tree})$.

> [!definition] Word embeddings
> **Distributed representations** of words as vectors that are
> - **low dimensional** (e.g. 512, against a vocabulary size $|V|$ in the hundreds of thousands),
> - **dense** (no zeros, unlike a one-hot vector),
> - **continuous** ($c_w \in \mathbb{R}^m$, where $c_w$ is the vector for word $w$ and $m$ its dimension),
> - **learned by performing a task** (a prediction task).

The popular historical approach is **Word2Vec** (Mikolov et al.), which consists of two methods: **Continuous Bag of Words (CBOW)** and **Skip-Gram**. This lecture describes CBOW; Skip-Gram, which does the reverse (predicts each context word from the centre word), is covered in [[MNLP-L05 - Static Embeddings|L05]].

> [!definition] The CBOW task
> Given a position $t$ in a sentence, the $n$ words to its left $\{w_{t-n}, \dots, w_{t-1}\}$ and the $m$ words to its right $\{w_{t+1}, \dots, w_{t+m}\}$, predict the word at position $t$: `the man X the road`, with `X = ?`

This resembles $n$-gram language modelling (with $n$ = model order minus one and $m = 0$), except that CBOW sees **both sides**: it is not trying to be a language model, and prediction is only the means of learning embeddings. Hence the design: a **simple feed-forward** network, so the effort goes into the embeddings; the embedding (projection) layer brought **close to the output**; typically $n = m$, with $n \in \{2, 5, 10\}$.

The context words' vectors are **summed** in the projection layer, and the sum predicts the centre word. A sum ignores order, so `the man X the road` and `road the X man the` give the same projection: the context is a [[Bag of Words|bag of words]], hence the name.

## The fixed vocabulary and the `<unk>` token

In neural models every weight matrix has a **fixed size**, including the input embeddings and the output layer, and the output layer (one row per vocabulary word, a softmax over $|V|$ at every position) contributes much of the memory footprint. Realistic vocabularies are large: a Russian one is about five times the size of an English one, because Russian marks case on every noun.

> [!definition] Short-list
> The quick fix: **ignore rare words and keep only the $n$ most frequent words**. Every rare word is mapped to one special token, `<unk>`. Typical sizes run from about ten thousand to a couple of hundred thousand words.

**The disadvantage:** all rare words receive **equal probability** in a given context, because they are the same token: in `She ___ the nail`, *hammered* and *doomscrolled* are both `<unk>`.

## What context cannot give you

Embedding models learn similar vectors for related words well: from *She moved the desk into her office* and *He moved the table into the office*, they learn that *desk* and *table* (internally just indices, say 11364 and 4527) occur in similar contexts. But *move* (2863) and *moved* (87542), or *transports* and *transported*, are also unrelated indices, so each pair must be learned separately from data. And an unseen or rare word such as *hammered* gets nothing, because it never appeared in any context.

> [!tip] The claim behind the rest of the lecture
> **Word-internal structure (morphology) can help induce meaning for unseen words.** Knowing *hammered* = *hammer* + *-ed*, a model could build a representation from a known stem and suffix. A word-level index uses context only and throws word-internal structure away.

## Two segmentation problems, and finding word boundaries

For languages with spaces around words, tokenization is easy, with exceptions (`Mr. Smith` should not become `Mr . Smith`). Two different problems remain:

- **No word boundaries in the script** (Chinese, Japanese, Thai, Burmese): boundaries must be established before any further processing. This is **word segmentation** or **tokenization**, covered below.
- **Morphologically rich languages blow up fixed vocabularies** with huge numbers of inflected forms. Splitting already delimited words into smaller units is **sub-word segmentation**, the next lecture.

Word segmentation must happen **before** part-of-speech (POS) tagging, parsing or anything else, so it is a task in its own right, with its own benchmarks (the SIGHAN bake-offs, shared tasks on Chinese). It is not trivial, because one string often has more than one valid segmentation:

> [!example] One string, two readings
> 研究生活动
> - 研究生 | 活动: "graduate-student activities"
> - 研究 | 生活 | 动: "research on life activities"

The space of segmentations is huge. A string of $n$ characters has $n-1$ gaps between characters, and each gap is either a boundary or not:

$$\#\text{segmentations} = 2^{\,n-1}$$

Only a few are valid, and enumerating them is out of the question. Approaches are **rule-based** or **data-driven (statistical)**.

## Rule-based segmentation: maximum matching

Assume a pre-compiled vocabulary $V$ of known words. Walk through the string and at each position take the **longest known word**. Greedy, simple, and a surprisingly strong baseline, still used for preprocessing or as a fast fallback in production. It runs in either direction, **Forward Maximum Matching** (FMM, left to right) or **Backward Maximum Matching** (BMM, right to left), and the two often disagree on ambiguous strings.

```pseudo
Algorithm: Forward Maximum Matching (FMM)
──────────────────────────────────────────────────────────────
Input:  sentence S (character string), vocabulary V,
        max word length L
Output: list of segmented words

i ← 0
output ← []
while i < length(S):
    for l ← min(L, length(S) − i) down to 1:   // longest first
        w ← S[i : i+l]                          // candidate substring
        if w ∈ V or l == 1:                     // single char always accepted
            append w to output
            i ← i + l
            break
return output
```

The inner loop counts **down**, so the first hit is the longest match. The `or l == 1` clause accepts a single character even if it is not in $V$, so the scan always terminates; it is also where unknown words end up, chopped into single characters. Cost: $O(n \cdot L)$ dictionary lookups.

```pseudo
Algorithm: Backward Maximum Matching (BMM)
──────────────────────────────────────────────────────────────
Input:  sentence S (character string), vocabulary V,
        max word length L
Output: list of segmented words

i ← length(S)
output ← []
while i > 0:
    for l ← min(L, i) down to 1:
        w ← S[i−l : i]                 // substring ending at i
        if w ∈ V or l == 1:
            prepend w to output        // keeps reading order
            i ← i − l
            break
return output
```

BMM mirrors FMM: start at the end, take the substring **ending** at $i$, **prepend** each word.

> [!example] FMM and BMM on 研究生活, with $V$ = {研究, 研究生, 生活, 生, 活}
> **FMM.** At $i=0$: 研究生活 is not in $V$; 研究生 (3 characters) is, so emit it. At $i=3$: 活 is in $V$. Result: **研究生 | 活**.
>
> **BMM.** At $i=4$: 研究生活 and 究生活 are not in $V$; 生活 (2 characters) is, so prepend it. At $i=2$: 研究 is in $V$. Result: **研究 | 生活**.
>
> Same string, same dictionary, different answers. Only the scan direction changed: the string is ambiguous, and each greedy rule commits to one reading.

**Bidirectional matching** runs both and compares. If the two segmentations **agree**, accept the result with high confidence. If they **disagree**, apply a tie-breaking heuristic: prefer the segmentation with **fewer total words**, or with **fewer single-character words** (both are proxies for less fragmentation). On the example both have two words, a tie; BMM has no single-character word and FMM has one (活), so the second heuristic picks 研究 | 生活.

> [!warning] Limits of maximum matching (bidirectional included)
> - **It assumes a precompiled vocabulary**, which is labour-intensive to compile with good coverage, and **genuinely new words** (names, typos) cannot be matched at all.
> - **No mechanism to exploit data**: more text does not make it better. This motivates statistical methods.
> - **Out-of-vocabulary words break silently**: chopped into single characters or greedily mismatched, with nothing in the output to flag it.
> - **The ambiguity is genuine**, not a matching artifact: some strings have several correct segmentations depending on meaning and context, so no direction-based rule is right in general.

## Statistical segmentation: tagging characters with BMES

The statistical move: instead of asking where to cut, ask what role each character plays in its word.

> [!definition] BMES tags
> - **B** (Begin): first character of a multi-character word
> - **M** (Middle): interior character of a word with **3 or more** characters
> - **E** (End): last character of a multi-character word
> - **S** (Single): a character that is a complete one-character word by itself
>
> Once every character has a label, the segmentation falls out: insert a word boundary **after every E and every S**.

Example: 研/B 究/E 生/S 活/S gives **研究 | 生 | 活**, a third segmentation of the maximum matching example. Segmentation is now ordinary **sequence labelling**, which any sequence labelling method can solve.

### The HMM

A **hidden Markov model** (HMM) assumes that a **hidden** sequence of tags (B/M/E/S) generated the **observed** sequence of characters. Its parameters:

- **Transition probabilities** $P(\text{tag}_i \mid \text{tag}_{i-1})$: how likely one tag follows another (tag bigrams). B is very likely to be followed by M or E, and **cannot** be followed by another B.
- **Emission probabilities** $P(\text{character}_i \mid \text{tag}_i)$: how likely a tag "produces" a given character. For example, 究 is likely under E, since it usually closes 研究. The direction matters: the character given the tag.

The tag definitions rule out half of the 16 possible transitions, and a model trained on segmented text learns these zeros from the counts. After B or M you are inside a word, so only **M or E** can follow. After E or S a word has just ended, so only **B or S** can follow.

The best tag sequence for characters $c_1, \dots, c_n$ is

$$\hat{t}_{1:n} = \arg\max_{t_{1:n}} \prod_{i=1}^{n} P(t_i \mid t_{i-1}) \, P(c_i \mid t_i)$$

where $c_i$ is the $i$-th character, $t_i \in \{B, M, E, S\}$ its tag, $t_0$ a sentence-start symbol, $P(t_i \mid t_{i-1})$ a transition and $P(c_i \mid t_i)$ an emission probability.

**Training** ideally uses **labelled data**, a segmented corpus, and estimates the probabilities from counts (simple maximum likelihood estimates):

$$P(t_i \mid t_{i-1}) = \frac{C(t_{i-1}, t_i)}{C(t_{i-1})} \qquad P(c \mid t) = \frac{C(t, c)}{C(t)}$$

where $C(\cdot)$ counts occurrences in the tagged corpus ($C(t, c)$: character $c$ carrying tag $t$). **Inference**: given an unsegmented string, find the most probable tag sequence.

### Viterbi decoding

Scoring every possible labelling is intractable: with $|T| = 4$ tags there are $4^n$ tag sequences. The **Viterbi algorithm** uses [[Dynamic Programming|dynamic programming]] (storing intermediate results instead of recomputing them). Cell $\text{score}[i][t]$ holds the probability of the **best path that ends at position $i$ with tag $t$**:

$$\text{score}[i][t] = \max_{t'} \; \text{score}[i-1][t'] \cdot P(t \mid t') \cdot P(c_i \mid t)$$

where $t'$ ranges over the tags at position $i-1$. The best path to $(i, t)$ passes through some tag at $i-1$, and only the best path into that tag can be part of it, so each cell compares just four candidates. **Backpointers** record which $t'$ won, so the full path is recovered at the end without recomputing it.

```pseudo
Algorithm: Viterbi decoding for BMES tagging
──────────────────────────────────────────────────────────────
Input:  characters c_1..c_n, tags T = {B, M, E, S},
        transitions P(t | t'), emissions P(c | t), start symbol <s>
Output: best tag sequence t_1..t_n

for each tag t in T:
    score[1][t] ← P(t | <s>) × P(c_1 | t)
for i ← 2 to n:
    for each tag t in T:
        score[i][t] ← max over t' in T of
                      score[i−1][t'] × P(t | t') × P(c_i | t)
        back[i][t]  ← the t' that achieved the max   // backpointer
t_n ← argmax over t of score[n][t]
for i ← n down to 2:                                 // backtrace
    t_{i−1} ← back[i][t_i]
return t_1..t_n
```

**Complexity: $O(n \times |T|^2)$**: at each of $n$ positions, each of $|T|$ tags compares $|T|$ predecessors. With four tags that is 16 per character, linear in sentence length, against $|T|^n$ for brute force.

### What the HMM gains, and what it still lacks

**Advantages over FMM and BMM:**
- **No fixed dictionary** of possible words is required: the model works on characters, so an unseen word is just a character sequence with a plausible tag path.
- **Ambiguity is resolved with probabilistic evidence** instead of tie-breaking heuristics.
- **Trainable directly from any segmented corpus**, even one from a different domain.

**Shortcomings:**
- **Narrow Markov assumption.** In a first-order HMM a tag depends only on the preceding tag and the current character. Higher orders are possible but more involved (transition parameters grow as $|T|^{k+1}$ for order $k$).
- **No longer-range context**, such as the word two positions back or a whole phrase. This motivates **conditional random fields (CRFs)**, which can condition on features of the whole input.
- **Unsupervised training costs quality.** HMMs *can* be trained fully unsupervised with **Baum-Welch** (an instance of EM), but quality drops.

Any neural sequence labeller (LSTM, Transformer) can be used instead.

## What happens inside a word: inflection, derivation, compounding

The **stem**, **root** or **lemma** of a word carries its core meaning. Three processes build words on it.

**Inflection** makes another form of the same word: *see, saw, seen* (tense), *book, books* (number); no new dictionary entry appears. **Derivation** makes a new word: *happy* (adjective) → *happiness* (noun), *happily* (adverb) changes the POS, while *function* → *disfunction* (standard spelling *dysfunction*) and *perfect* → *imperfect* change only the meaning. **Compounding** joins words: German *Tisch* + *Bein* = *Tischbein* (table leg), *sicher* + *gehen* = *sichergehen* (sure + go, "make sure"). German writes compounds as one token where English often uses two.

> [!definition] The three processes
> - **Inflection:** retains the POS, but changes features such as number and tense.
> - **Derivation:** changes the POS and/or the meaning.
> - **Compounding:** combines several words into one word.

These processes are **very productive**: they apply iteratively and in new combinations, which leads to a large increase in vocabulary size. *doomscrolling, doomscrolled, doomscrolls, doomscroller* is a recent compound (*doom* + *scroll*) already carrying inflection (*-ing*, *-ed*, *-s*) and derivation (*-er*). New forms keep appearing, so no fixed vocabulary is ever complete.

### Inflection in English and elsewhere

English inflects little. A verb has five forms (*see, saw, seen, seeing, sees*), and regular verbs use one string for both past slots (*called*). Nouns inflect for **number** (*book, books*) and for **case only on pronouns** (*he, him, his, them, their*).

Many languages inflect far more. **Russian** has more cases and marks them on **all** nouns. **German** marks **mood**: *kommt* (indicative), *käme* (subjunctive), *komm* (imperative). **Slavic languages** (Russian, Czech, Polish) mark **aspect**: Russian *sdelat'* (perfective, completed) against *delat'* (imperfective, ongoing). Features combine independently, so forms per word **multiply** (number times case times possessor...), which is how one lemma reaches thousands of forms.

English derivation, by what the affix does:
- **Nominalization** (makes a noun): *derive* + *-ation*, *kill* + *-er*, *happy* + *-ness*.
- **Adjectivization** (makes an adjective): *account* + *-able*, *colony* + *-al* = *colonial*, *office* + *-al* = *official*.
- **Negation** (changes meaning, keeps POS): *un-* (*unseen*), *mis-* (*misjudge*), *dis-* (*disfunctional*), *im-* (*implausible*), *in-* (*indifferent*).

*colonial* and *official* show that spelling can change where morphemes meet.

## Morphemes and affixes

> [!definition] Morpheme
> The **smallest meaning-carrying parts of words**.
>
> *disproportionally* = *dis* + *proportion* + *al* + *ly* = prefix + stem + suffix + suffix

- A **root** is a **free** morpheme (it can stand alone as a word) that determines the basic meaning of a word: *proportion*.
- An **affix** is a **bound** morpheme (it cannot stand alone) attached to change meaning or grammatical function: *dis-*, *-al*, *-ly*.

Affixes by position:
- **Prefix:** attaches to the front. English words mostly have only one.
- **Suffix:** attaches to the end. There can be more than one (*-al*, *-ly*).
- **Infix:** inserted in the middle of a word: *passerby* → *passersby*, with *-s-* inside. More common in other languages.
- **Circumfix:** attaches to the front and back **simultaneously**: German *ge-seh-en* (past participle of *sehen*, "to see"). *ge-* and *-en* together are one morpheme, so neither half can be stripped on its own.

All of these are **concatenative**: the word is a sequence of morphemes. **Non-concatenative** morphology changes the root itself. The example is **apophony**, a vowel change in the stem: German *Buch* → *Bücher* (book, books) and *Haus* → *Häuser* (house, houses). The plural is the vowel change plus a suffix, and no cut in *Bücher* separates "book" from "plural". Any method that cuts strings into contiguous pieces, from maximum matching to the subword methods of the next lecture, can only fully capture the concatenative case.

## Four ways languages package morphemes

Languages can be grouped by how they realize morphological processes.

> [!definition] Morphological typology
> - **Isolating** (Chinese, Vietnamese): words tend not to change form at all; grammatical relations come from **word order**; **particles** express additional modification.
> - **Agglutinative** (Turkish, Finnish): words are built by stringing morphemes together (potentially with character modifications); **each morpheme contributes one clear piece of meaning**.
> - **Fusional** (Russian, Spanish, Arabic): **a single suffix simultaneously encodes several grammatical features** that often cannot be cleanly separated.
> - **Polysynthetic** (Inuktitut, Mohawk): multiple grammatical roles (verb, subject, object) and various modifiers (negation, tense, mood, instrument, location) are **fused or agglutinated into a single word**.

**Isolating.** Chinese 我 昨天 看 了 一 本 书, "I read a book yesterday" (*I, yesterday, read, PFV, one, CL, book*). The verb 看 (*kàn*) never changes form: pastness comes from 昨天 "yesterday" and the perfective particle 了 (PFV, completed action); 本 is a classifier (CL), required between a number and a noun. Word order is plain SVO (subject, verb, object). Vietnamese *Tôi đã đi học*, "I went to study" (*I, PST, go, study*): neither *đi* nor *học* changes form; *đã* is a past (completed-action) particle.

**Agglutinative.** Turkish *evlerinizden*, "from your houses" = *ev-ler-iniz-den* = house + plural + your + from: one Turkish word for a four-word English phrase. Finnish *taloissamme*, "in our houses" = *talo-i-ssa-mme* = house + plural + inessive + our. The **inessive** is a locative case ("in"); Finnish has others, such as elative ("out of"), illative ("into") and allative ("to"). Each suffix is clearly separable and **reusable across many other nouns**. Vowel or consonant changes may happen at attachment, so boundaries are clean between morphemes but not always between characters.

**Fusional.** Russian *knigami*, "with books" = *knig-ami*, glossed book + instrumental.plural.feminine: one suffix fuses case, number and gender into a single unsegmentable morpheme. (That is the lecture's gloss. Outside the course material: *-ami* is the instrumental plural for all three genders, so strictly it fuses case and number.) Spanish *hablé*, "I spoke" = *habl-é*: the single vowel *-é* fuses person (1st), number (singular), tense (preterite, past) and aspect (preterite, completed). Arabic *yaktubu*, "he writes": the consonant root *k-t-b* (C1 C2 C3, "write") is poured into the pattern *ya*-C1C2-*u*-C3-*u*, which fuses person (3rd), gender (masculine) and aspect (not completed). The pattern's vowels are not written in regular Arabic, so part of the grammar is missing from the written string, and since root and pattern interleave, the word is also non-concatenative.

**Polysynthetic.** Inuktitut *Qangatasuukkuvimmuuriaqalaaqtunga*, "I will have to go to the airport" = *qangata-* (fly) + *-suukkuvik* (thing that flies habitually → airport) + *-mut* (to, allative case) + *-uq-* (go to) + *-riaqaq-* (have to) + *-laaq-* (future) + *-tunga* (1st person singular). One word corresponds to a whole English sentence: a verb, its subject, its object, tense and modal meaning, all fused together. The morphemes also change where they meet (*-k-mut-uq* becomes *-mmuu-*), so cutting the string cannot recover them.

| type | morphemes per word | one feature per affix? | examples |
|---|---|---|---|
| Isolating | about one | nothing to split | Chinese, Vietnamese |
| Agglutinative | several | yes, separable | Turkish, Finnish |
| Fusional | several | no, fused | Russian, Spanish, Arabic |
| Polysynthetic | very many | fused or agglutinated | Inuktitut, Mohawk |

Isolating and polysynthetic are the two ends of a scale of how much meaning one word carries. Agglutinative and fusional sit between them and differ on a separate question:

> [!warning] Agglutinative against fusional
> Both pack several features into affixes. The difference is **separability**: Turkish *-ler* is plural and nothing else; Spanish *-é* is person, number, tense and aspect at once, with no part meaning any one of them. Defining agglutinative as "more morphemes per word" is wrong.

## Why word-level vocabularies break

**Morphology is common to most languages**, and rich morphology encodes what English expresses through word order: Turkish *evlerinizden* and "from your houses" carry the same information.

**Word-level vocabularies blow up or become extremely restrictive.** A single lemma can surface as **thousands of word forms** (surface forms). Keep them all and $|V|$, with the embedding matrix and output softmax, explodes. Keep a **fixed vocabulary sized for English** and other languages get **massive out-of-vocabulary (OOV) rates**, every unknown form becoming `<unk>`. A vocabulary shared across languages inherits the imbalance: rich languages get split into many more pieces per word than English (next lecture).

**These problems are the main motivation for morphological analysers** (tools that split words into morphemes). But morphological analysis is **highly language dependent**, so each language needs its own: the WordNet problem again. Modern approaches **build on data-driven methods**, and the **practical compromise is subword tokenization**: statistically useful pieces learned from data, in a language-independent way, instead of linguistically correct morphemes. Because it can always fall back to single characters, a fixed-size subword vocabulary leaves nothing out of vocabulary. The pieces only sometimes line up with morphemes, and as cuts of a string they cannot capture *Bücher* or Arabic root-and-pattern words. That is [[MNLP-L04 - Subword Segmentation|L04]].

## What to be able to answer

- The four properties of word embeddings; what CBOW predicts from which context, and why it is a "bag".
- What a short-list does, and why all rare words then get equal probability.
- Word segmentation against subword segmentation, and which languages need which.
- FMM and BMM by hand; bidirectional tie-breaking; the limits of maximum matching.
- BMES tags and impossible transitions; emission direction; the Viterbi recursion, backpointers and $O(n|T|^2)$; what HMMs gain and lack.
- Inflection, derivation, compounding; productivity; free and bound morphemes; affix types; apophony.
- The four morphological types with example languages and words; agglutinative against fusional.
- Why fixed word vocabularies fail for rich languages, and why subword tokenization is the compromise.

Model answers are in the exam questions below.

## Exam questions

> [!exam]- Name the four morphological types, give example languages for each, and illustrate each with one analysed word or sentence. What distinguishes agglutinative from fusional?
> **Key points:** isolating: no inflection, word order and particles (Chinese, Vietnamese); agglutinative: one morpheme, one meaning (Turkish *ev-ler-iniz-den*); fusional: one affix, several fused features (Russian *knig-ami*); polysynthetic: a sentence in one word (Inuktitut); agglutinative and fusional differ in separability, not density.
>
> - **Isolating** (Chinese, Vietnamese): words do not change form; grammar comes from **word order** and **particles**. Chinese 我 昨天 看 了 一 本 书 "I read a book yesterday": 看 never changes, past comes from the particle 了 (PFV) and 昨天 "yesterday".
> - **Agglutinative** (Turkish, Finnish): morphemes strung together, **one morpheme, one meaning**, visible boundaries. Turkish *ev-ler-iniz-den* = house + plural + your + from, "from your houses".
> - **Fusional** (Russian, Spanish, Arabic): **one affix encodes several features** that cannot be separated. Russian *knig-ami* "with books": glossed instrumental, plural and feminine in one suffix. Spanish *habl-é*: person, number, tense, aspect in one vowel.
> - **Polysynthetic** (Inuktitut, Mohawk): verb, subject, object and modifiers (negation, tense, mood, instrument, location) packed into **one word**. *Qangatasuukkuvimmuuriaqalaaqtunga* "I will have to go to the airport".
> - **The axis:** isolating and polysynthetic are the two ends of a scale of how much meaning one word carries. Agglutinative and fusional sit in the middle and differ on a separate question: can the packed meaning be cut apart?
>
> The usual mistake: defining agglutinative as "more morphemes". It means **separable** morphemes; fusional words also pack several features into affixes. (On *-ami*: the lecture glosses it with feminine gender, but outside the course material it is the instrumental plural for all three genders.)

> [!exam]- Why do fixed word-level vocabularies break for morphologically rich languages and for multilingual models, and what is the practical fix?
> **Key points:** fixed-size matrices and softmax make $|V|$ drive memory; inflection multiplies forms and productivity keeps the vocabulary open; either $|V|$ explodes or OOV soars with every rare form as `<unk>`; a shared multilingual vocabulary fragments rich languages; fix: subword tokenization with zero OOV.
>
> - Neural weight matrices are **fixed size**; the output layer is a $|V| \times m$ matrix plus a softmax over $|V|$ at every position, so $|V|$ drives memory.
> - A realistic Russian vocabulary is about **five times** the English one, because Russian marks case on every noun, so each lemma has many more surface forms.
> - Inflectional features multiply: forms per lemma = **product** of slot sizes. Productivity (*doomscrolling, doomscrolled, doomscrolls, doomscroller*) means no vocabulary can be closed, so OOV never goes away.
> - **Two horns:** keep all forms and $|V|$ explodes (embedding matrix and softmax with it); or use a vocabulary sized for English and get massive OOV rates, with every rare form mapped to `<unk>` and given **equal probability**.
> - Word indices hide relations (*move* 2863 vs *moved* 87542) and give nothing for unseen words (*hammered*). Morphology could build meaning for unseen words; context alone cannot.
> - **Multilingual:** a vocabulary shared across languages inherits the imbalance: rich languages get split into many more pieces per word than English.
> - Fix: morphological analysers are language dependent and need per-language expertise, so the practical compromise is **subword tokenization**: statistically useful pieces, language-agnostic, learned from data, zero OOV by falling back to characters.

> [!exam]- Run Forward and Backward Maximum Matching on 研究生活 with vocabulary {研究, 研究生, 生活, 生, 活}. Why do the results differ, how does bidirectional matching resolve it, and what are the method's limits?
> **Key points:** FMM gives 研究生 | 活; BMM gives 研究 | 生活; genuine ambiguity leaks through the greedy scan direction; bidirectional: accept if they agree, else prefer fewer words or fewer single-character words; limits: fixed vocabulary, OOV silently split, no learning from data.
>
> 1. **FMM:** at $i=0$, 研究生活 is not in $V$, 研究生 is (3 chars), emit it. At $i=3$, 活 is in $V$. Result **研究生 | 活**.
> 2. **BMM:** at $i=4$, the longest suffix in $V$ is 生活 (2 chars), prepend it. At $i=2$, 研究 is in $V$. Result **研究 | 生活**.
> 3. **Why:** same string, same dictionary, only the scan direction changed. The string is genuinely ambiguous and a greedy rule lets the ambiguity leak through. Neither algorithm is buggy.
> 4. **Bidirectional:** if FMM and BMM agree, accept with high confidence. If not, tie-break: fewer total words, or fewer single-character words. Here both give 2 words (tie), and BMM wins on single-character words (0 against FMM's 1 for 活).
> 5. **Limits:** needs a precompiled per-language vocabulary; new words (names, neologisms, typos) can never be matched; cannot improve from data; OOV words are silently chopped into single characters; genuine ambiguity cannot be fixed by any direction heuristic.

> [!exam]- Reformulate Chinese word segmentation as sequence labelling and explain how an HMM with Viterbi solves it. Give the recursion, the complexity, and what the approach still cannot do.
> **Key points:** BMES tags with a boundary after every E and S; hidden tags emit characters, transitions and emissions estimated by MLE; Viterbi recursion with backpointers; $O(n|T|^2)$ against $|T|^n$; first-order, no long-range context (CRFs), unsupervised training worse.
>
> - **BMES tags:** B (first char of multi-char word), M (interior of 3+ char word), E (last char of multi-char word), S (single-char word). Boundary after every E and S.
> - **HMM:** hidden tag sequence generates the observed characters. Transitions $P(t_i \mid t_{i-1})$, emissions $P(c_i \mid t_i)$. Decode $\hat{t}_{1:n} = \arg\max_{t_{1:n}} \prod_i P(t_i \mid t_{i-1}) P(c_i \mid t_i)$.
> - **Training:** MLE counts from a segmented corpus: $P(t_i \mid t_{i-1}) = C(t_{i-1}, t_i)/C(t_{i-1})$, $P(c \mid t) = C(t,c)/C(t)$.
> - **Viterbi:** $\text{score}[i][t] = \max_{t'} \text{score}[i-1][t'] \cdot P(t \mid t') \cdot P(c_i \mid t)$, with a backpointer to the winning $t'$; backtrace from the best final cell.
> - **Complexity:** $O(n |T|^2)$, i.e. $16n$ for 4 tags, against $4^n$ brute-force tag sequences.
> - **Gains over maximum matching:** no fixed dictionary, ambiguity resolved by probabilistic evidence, trainable from any segmented corpus.
> - **Still cannot:** first-order Markov assumption (higher orders cost $|T|^{k+1}$ parameters); no longer-range context (motivates CRFs); unsupervised Baum-Welch (EM) training degrades quality. Neural sequence labellers (LSTMs, Transformers) can be used instead.

> [!exam]- Distinguish inflection, derivation and compounding with examples, and explain why productivity of these processes matters for NLP.
> **Key points:** inflection keeps POS, changes features (*see, saw*); derivation changes POS and/or meaning (*happy, happiness*; *perfect, imperfect*); compounding joins words (*Tischbein*); productivity means no vocabulary can be closed, so OOV never goes away.
>
> - **Inflection:** keeps the part of speech, changes grammatical features (tense, number). *see, saw, seen*; *book, books*. Another form of the same dictionary entry.
> - **Derivation:** changes POS and/or meaning, giving a new dictionary entry. *happy* → *happiness* (noun), *happy* → *happily* (adverb); *perfect* → *imperfect* (meaning only).
> - **Test:** is the result a new word with its own dictionary entry? A changed part of speech means derivation (*happy / happiness*), but derivation can also keep it (*perfect / imperfect*), so a kept part of speech alone does not prove inflection.
> - **Compounding:** several words into one. German *Tisch* + *Bein* = *Tischbein* "table leg"; *sicher* + *gehen* = *sichergehen* "make sure". German writes it as one token where English uses two.
> - **Productivity:** the processes apply iteratively and to new combinations. *doomscroll* is a new compound already carrying inflection (*-ing, -ed, -s*) and derivation (*-er*). No fixed vocabulary can be closed over a productive process, which is the formal reason OOV never disappears.

> [!exam]- What is non-concatenative morphology, and why can no string-cutting segmenter (from maximum matching to subword tokenization) fully handle it? Use German, Arabic and Inuktitut.
> **Key points:** change inside the root (apophony, *Buch*/*Bücher*); circumfix *ge-...-en* is one morpheme in two pieces; Arabic interleaves root and pattern and leaves the vowels unwritten; Inuktitut morphemes change where they meet; segmenters cut contiguous pieces, so they fully handle only the concatenative case.
>
> - **Concatenative:** word = concatenation of morphemes; string splitting can recover them.
> - **Non-concatenative:** the change happens **inside the root**. Apophony: German *Buch* → *Bücher*, *Haus* → *Häuser* (vowel change plus suffix). No cut separates "book" from "plural" in *Bücher*.
> - **Circumfix** (related problem): German *ge-seh-en*: *ge-* and *-en* are one morpheme in two pieces, so neither stripping the prefix nor the suffix alone works.
> - **Arabic:** consonantal root *k-t-b* "write" poured into a pattern *ya-* C1 C2 *-u-* C3 *-u* = *yaktubu* "he writes". Root and pattern are **interleaved**, and the pattern's vowels are not written in regular Arabic, so part of the information is absent from the string.
> - **Inuktitut:** morphemes change where they meet (*-k-mut-uq* surfaces as *-mmuu-*), so cutting the string cannot recover them.
> - Every segmentation method cuts a string into **contiguous pieces**, so subword segmentation can only fully capture the concatenative case.

> [!exam]- Word segmentation and subword segmentation are often confused. Define each, give a language where each is needed, and say how they are solved.
> **Key points:** word segmentation finds boundaries in unmarked scripts (Chinese, Thai); solved by maximum matching or BMES labelling (HMM, CRF, neural); subword segmentation splits delimited words in rich languages (Finnish); solved by statistical subword tokenization learned from data; same cutting problem, different target units.
>
> - **Word segmentation (tokenization):** establishing word boundaries in scripts that do not mark them (Chinese, Japanese, Thai, Burmese). Must happen before POS tagging, parsing or anything else. Solved by maximum matching (FMM, BMM, bidirectional) or statistical BMES labelling (HMM plus Viterbi, CRFs, neural taggers). Benchmarked in the SIGHAN bake-offs.
> - **Subword segmentation:** splitting words that are already well delimited into units smaller than the word, because morphologically rich languages (Finnish) blow up fixed vocabularies. Solved by statistical, language-independent subword tokenization (next lecture).
> - Both decide where to cut a character string; they differ in what the pieces are meant to be: whole words against units smaller than words.

## Flashcards

> [!card]- What is the difference between a vocabulary and a lexicon?
> A **vocabulary** is an inventory of word forms; a **lexicon** also stores the meanings and functions of the words.

> [!card]- What is the difference between word segmentation and subword segmentation?
> Word segmentation finds **word boundaries** in scripts that do not mark them (Chinese); subword segmentation splits already delimited words into **smaller units** (Finnish).

> [!card]- Why are hand-built lexical resources like WordNet a poor basis for word meaning at scale?
> They are **expensive** (every relation entered by hand), **incomplete**, and **language-specific**.

> [!card]- What four properties define word embeddings?
> **Low dimensional**, **dense**, **continuous**, and **learned by performing a prediction task**.

> [!card]- What is the difference between CBOW and Skip-Gram?
> CBOW predicts the **centre word from its context**; Skip-Gram predicts **each context word from the centre word**.

> [!card]- How does the CBOW task differ from n-gram language modelling?
> A language model sees only the **left** context; CBOW sees **both sides**, because its goal is embeddings and prediction is only the means.

> [!card]- Why is the CBOW network deliberately kept simple?
> So that capacity goes into the **embeddings** themselves instead of into prediction quality.

> [!card]- Why is CBOW called "bag of words"?
> Its projection layer **sums** the context vectors, and summation is commutative, so the **order** of the context words is lost.

> [!card]- Why does vocabulary size $|V|$ drive the memory footprint of a neural model?
> Weight matrices are **fixed size**, and the output layer is a $|V| \times m$ matrix plus a **softmax over $|V|$** at every position.

> [!card]- What is a vocabulary short-list?
> Keeping only the **$n$ most frequent** words and mapping every rare word to a single `<unk>` token.

> [!card]- What is the main disadvantage of a short-list with `<unk>`?
> All rare words get **equal probability** in a given context, because they are literally the same token.

> [!card]- Why is a realistic Russian word vocabulary about five times the size of the English one?
> Russian marks **case on every noun**, so each lemma surfaces in many more distinct forms.

> [!card]- Why can a word-level model not see that *move* and *moved* are related?
> They are **unrelated indices**, so the relation must be learned from data separately for every inflected pair.

> [!card]- How could morphology give a representation for an unseen word like *hammered*?
> Compose it from known parts: an embedding for **hammer** plus what **-ed** does to a verb. Context cannot help, since the word was never seen.

> [!card]- Why is Chinese word segmentation hard even with a good dictionary?
> The same character string often has **several valid segmentations**, and the characters alone do not decide between them.

> [!card]- How many possible segmentations does a string of $n$ characters have?
> $2^{n-1}$: each of the $n-1$ internal gaps is a boundary or not. Too many to enumerate.

> [!card]- What does maximum matching do at each position of a character string?
> It greedily takes the **longest word** found in a precompiled vocabulary.

> [!card]- What does the `or l == 1` clause in maximum matching do?
> Emits a **single character** even if it is not in the vocabulary, so the algorithm always terminates.

> [!card]- What happens to an out-of-vocabulary word under maximum matching?
> It is **silently** chopped into single characters or greedily mismatched; nothing in the output signals it.

> [!card]- How does Backward Maximum Matching differ from Forward Maximum Matching?
> It starts at the **end**, takes the substring **ending** at position $i$, and **prepends** each word so output stays in reading order.

> [!card]- With vocabulary {研究, 研究生, 生活, 生, 活}, what does Forward Maximum Matching return for 研究生活?
> **研究生 | 活**

> [!card]- With vocabulary {研究, 研究生, 生活, 生, 活}, what does Backward Maximum Matching return for 研究生活?
> **研究 | 生活**

> [!card]- Why can Forward and Backward Maximum Matching give different segmentations of the same string?
> The string is **genuinely ambiguous**, and the greedy rule lets that leak through; only the scan direction differs. Neither algorithm is buggy.

> [!card]- How does bidirectional maximum matching decide when FMM and BMM disagree?
> A tie-breaking heuristic: prefer **fewer total words** or **fewer single-character words**. If they agree, accept with high confidence.

> [!card]- Why can no direction-based heuristic segment correctly in general?
> Some strings have several **correct** segmentations depending on meaning and context, so you need a method that weighs evidence.

> [!card]- How is word segmentation reformulated for statistical methods?
> As **sequence labeling**: tag each character with its role in a word (BMES).

> [!card]- What do the BMES tags stand for?
> **Begin**, **Middle**, **End**, **Single**.

> [!card]- Which BMES tag only occurs in words of three or more characters?
> **M** (Middle), for an interior character.

> [!card]- After which BMES tags is a word boundary inserted?
> After every **E** and every **S**.

> [!card]- What segmentation does the tagging 研/B 究/E 生/S 活/S give?
> **研究 | 生 | 活**

> [!card]- In an HMM word segmenter, which sequence is hidden?
> The **BMES tag sequence**, which generates the observed characters.

> [!card]- What is the direction of the emission probability in a BMES HMM?
> $P(\text{character} \mid \text{tag})$: the character given the tag.

> [!card]- Why can the BMES tag B never be followed by another B?
> A word opened by B must continue with M or **close with E** before a new word can begin.

> [!card]- Which tags can follow E or S in BMES tagging?
> Only **B or S**: a word has just ended, so the next one must start.

> [!card]- How many of the 16 BMES tag transitions are structurally impossible?
> **Half**; a model trained on segmented text learns these zeros from the counts.

> [!card]- How are BMES HMM parameters estimated from a segmented corpus?
> **Maximum likelihood** from counts: e.g. a transition is the tag-bigram count divided by the count of the previous tag.

> [!card]- What is the Viterbi recursion for BMES tagging?
> $\text{score}[i][t] = \max_{t'} \text{score}[i-1][t'] \cdot P(t \mid t') \cdot P(c_i \mid t)$

> [!card]- What are backpointers for in Viterbi?
> They store each cell's **winning predecessor**, so the best path is recovered by backtracing instead of recomputing paths.

> [!card]- How does the cost of Viterbi compare with scoring every tag sequence?
> Viterbi is $O(n|T|^2)$, **linear** in sentence length; brute force is **exponential**, $|T|^n$.

> [!card]- Why does an HMM handle unseen words better than maximum matching?
> It needs **no fixed dictionary**: it works over characters, so an unseen word is just a plausible tag path.

> [!card]- How does an HMM resolve segmentation ambiguity, compared with bidirectional matching?
> With **probabilistic evidence** learned from data, instead of direction-based tie-breaking heuristics.

> [!card]- What does the first-order Markov assumption restrict in an HMM segmenter?
> Each tag depends only on the **previous tag** and the current character, so there is no longer-range context.

> [!card]- Which model is motivated by the HMM's lack of longer-range context in word segmentation?
> **Conditional random fields (CRFs)**, which condition on arbitrary features of the whole input.

> [!card]- What is the difference between inflection and derivation?
> Inflection **keeps the part of speech** and changes grammatical features; derivation changes the **part of speech and/or the meaning**.

> [!card]- *perfect* → *imperfect* keeps the part of speech. Is it inflection or derivation?
> **Derivation**: it changes the meaning and yields a new dictionary entry.

> [!card]- What is compounding?
> Combining **several words into one word**, e.g. German *Tisch* + *Bein* = *Tischbein* (table leg).

> [!card]- Why can no fixed vocabulary be closed over morphology?
> Morphological processes are **productive**: they apply iteratively and to new combinations (*doomscrolled*), so new forms keep appearing.

> [!card]- Why do inflectional features multiply the number of word forms instead of adding to it?
> Features are **independent slots**, so forms per lemma are the **product** of the slot sizes.

> [!card]- What is a morpheme?
> The **smallest meaning-carrying** part of a word.

> [!card]- What is the difference between free and bound morphemes?
> Free morphemes can **stand alone** as words (*book*); bound morphemes cannot (*dis-*, *-ed*).

> [!card]- Why can a circumfix like German *ge-seh-en* not be analysed by stripping a prefix or a suffix?
> *ge-* and *-en* are **one morpheme in two pieces**, attached front and back simultaneously, so neither half can be stripped alone.

> [!card]- What is non-concatenative morphology?
> Morphology that changes the **root itself**, instead of stringing morphemes together.

> [!card]- What is the name for a vowel change inside the root, as in German *Buch* → *Bücher*?
> **Apophony**, a kind of non-concatenative morphology.

> [!card]- Why can no string-cutting segmenter fully handle non-concatenative morphology?
> Segmenters cut **contiguous pieces**, but the change is **inside the root**: no cut in *Bücher* separates "book" from "plural".

> [!card]- In an isolating language, what carries the grammatical relations?
> **Word order**, with **particles** for additional modification.

> [!card]- Which morphological type has words that tend not to change form at all?
> **Isolating** (Chinese, Vietnamese).

> [!card]- Which morphological type has each morpheme contribute one clear meaning, with visible boundaries?
> **Agglutinative** (Turkish, Finnish).

> [!card]- Which morphological type has a single suffix encode several grammatical features that cannot be separated?
> **Fusional** (Russian, Spanish, Arabic).

> [!card]- Which morphological type packs verb, subject, object and modifiers into a single word?
> **Polysynthetic** (Inuktitut, Mohawk).

> [!card]- What distinguishes agglutinative from fusional morphology?
> **Separability**: an agglutinative affix carries one feature; a fusional affix carries several fused features.

> [!card]- Why is "more morphemes per word" the wrong definition of agglutinative?
> Fusional affixes also pack several features; the criterion is **separability** (one feature per affix or several fused).

> [!card]- What scale do isolating and polysynthetic languages sit at the two ends of?
> **How much meaning** is packed into one word.

> [!card]- What are the morphemes of Turkish *evlerinizden* ("from your houses")?
> *ev-ler-iniz-den*: **house + plural + your + from**.

> [!card]- What makes agglutinative morphology learnable for a segmenter?
> Each suffix is separable and **reusable across many other nouns** (Finnish *-ssa*, "in").

> [!card]- Are agglutinative morpheme boundaries always clean at the character level?
> **No**: vowels or consonants may change at attachment. Boundaries are clean between morphemes, not always between characters.

> [!card]- What does Monz's gloss of Russian *knig-ami* ("with books") put into the one suffix *-ami*?
> **Case** (instrumental), **number** (plural) and **gender** (feminine), fused and unsegmentable.

> [!card]- Strictly, does Russian *-ami* mark gender?
> **No**: it is the instrumental plural for all three genders (a correction from outside the course material).

> [!card]- How does Arabic root-and-pattern morphology form *yaktubu* ("he writes")?
> The consonantal root **k-t-b** ("write") is poured into a **pattern** template of prefixes and vowels, *ya-* C1 C2 *-u-* C3 *-u*.

> [!card]- Why is written Arabic hard for a tokenizer beyond its interleaved morphemes?
> The pattern's **vowels are not written**, so part of the grammar is absent from the string.

> [!card]- Why can Inuktitut words not be split back into morphemes by string matching?
> The morphemes **change where they meet** (*-k-mut-uq* surfaces as *-mmuu-*).

> [!card]- What are the two ways a word-level vocabulary fails for morphologically rich languages?
> Keep all forms and **$|V|$ explodes**; keep an English-sized vocabulary and get massive **OOV** rates.

> [!card]- How does a vocabulary shared across languages treat morphologically rich languages compared with English?
> Rich languages get split into **many more pieces per word** than English.

> [!card]- Why are morphological analysers not the general fix for vocabulary blow-up?
> They are **highly language dependent**: one per language, each needing linguistic expertise.

> [!card]- What does subword tokenization guarantee?
> **Nothing out of vocabulary** with a fixed-size vocabulary, because it can always fall back to single characters.

> [!card]- Do subword tokens line up with morphemes?
> **Only sometimes**; as contiguous cuts they cannot capture *Bücher* or Arabic root-and-pattern words.

> [!card]- True or false: CBOW preserves the order of the context words in its projection layer.
> **False.** It sums the context vectors, and summation is commutative, so order is lost.

> [!card]- True or false: Forward and Backward Maximum Matching always produce the same segmentation given the same vocabulary.
> **False.** On ambiguous strings they often disagree (研究生 | 活 against 研究 | 生活).

> [!card]- True or false: Maximum matching flags out-of-vocabulary words in its output.
> **False.** OOV words are silently chopped into single characters or mismatched.

> [!card]- True or false: Viterbi decoding of BMES tags runs in time linear in the sentence length.
> **True.** It is $O(n|T|^2)$, and $|T| = 4$ is fixed.

> [!card]- True or false: In a BMES HMM, the emission probability is the probability of the character given the tag.
> **True.** Emission is $P(c_i \mid t_i)$, not the tag given the character.

> [!card]- True or false: An HMM word segmenter can be trained without a segmented corpus, at a cost in quality.
> **True.** Unsupervised Baum-Welch (EM) training is possible, but quality drops.

> [!card]- True or false: Derivation always changes a word's part of speech.
> **False.** It changes POS **and/or** meaning: *perfect* → *imperfect* keeps the adjective.

> [!card]- True or false: A fixed word vocabulary sized for English produces massive OOV rates for morphologically rich languages.
> **True.** Each lemma has many surface forms, so most forms fall outside the vocabulary.

> [!card]- True or false: Agglutinative languages differ from fusional languages mainly in having more morphemes per word.
> **False.** The difference is separability: fusional affixes also pack several features, but fused.

> [!card]- True or false: Subword segmentation recovers the morphemes of non-concatenative forms such as German *Bücher*.
> **False.** It only cuts contiguous pieces, so it fully handles only the concatenative case.

> [!card]- True or false: An HMM word segmenter needs a fixed dictionary of possible words.
> **False.** It works over characters; maximum matching is the method that needs a dictionary.

> [!card]- True or false: Inflection changes a word's part of speech.
> **False.** Inflection retains the POS and changes features such as number and tense.

> [!card]- True or false: Apophony is a kind of concatenative morphology.
> **False.** It changes a vowel inside the root (*Buch* → *Bücher*), so it is non-concatenative.

> [!card]- True or false: Arabic is classed as a fusional language.
> **True.** One root-and-pattern template fuses person, gender and aspect.

> [!card]- True or false: With a vocabulary short-list, rare words keep distinct probabilities in a given context.
> **False.** They all map to `<unk>`, so all rare words receive equal probability.

> [!card]- Multiple selection: which are true of Forward Maximum Matching? (A) It takes the longest vocabulary word at each position (B) It improves as it sees more text (C) It always terminates because a single character is emitted as fallback (D) It needs no precompiled vocabulary
> **A and C.** B: it has no mechanism to exploit data. D: it depends entirely on a precompiled vocabulary.

> [!card]- Multiple selection: which are true of BMES tagging with an HMM? (A) A word boundary follows every E and every S (B) B can be followed by another B (C) Viterbi keeps a backpointer to each cell's best predecessor (D) The model conditions on arbitrary features of the whole input
> **A and C.** B: a word opened by B must continue with M or close with E. D: that describes CRFs; the HMM is first-order.

> [!card]- Multiple selection: which are true of morphological types? (A) In Turkish, each agglutinative suffix carries one meaning (B) Russian *-ami* fuses instrumental case and plural (C) Isolating languages mark tense mainly by inflecting the verb (D) In a polysynthetic language one word can express a whole sentence
> **A, B and D.** C: isolating verbs do not change form; tense comes from particles and adverbs.

> [!card]- Multiple selection: which are true of HMM word segmentation? (A) A tag depends only on the preceding tag and the current character (B) It cannot be trained without a segmented corpus (C) CRFs address its lack of longer-range context (D) It resolves ambiguity with tie-breaking heuristics
> **A and C.** B: Baum-Welch trains it unsupervised, at a cost in quality. D: that is bidirectional matching; the HMM uses probabilistic evidence.

> [!card]- Multiple selection: which are correct? (A) *books* is an inflection of *book* (B) *imperfect* is an inflection of *perfect*, since the POS stays (C) *Tischbein* is a compound (D) *happiness* is derived from *happy*
> **A, C and D.** B: *imperfect* is derivation; derivation changes the POS and/or the meaning.

## Links

- **Course:** [[MNLP - Overview|Course overview]] · [[MNLP - Mini Project]]
- **Previous:** [[MNLP-L02 - Multilinguality and Writing Systems]]
- **Next:** [[MNLP-L04 - Subword Segmentation]], morphological analysis and subword segmentation, which takes the OOV problem from section 15 and solves it statistically
- **Concepts:** [[Word Embeddings]] · [[Tokenization]] · [[Bag of Words]] · [[Dynamic Programming]]
- **Source:** `MNLP_morphology.pdf` (Christof Monz, Morphology: Word Formation, 34 slides)
