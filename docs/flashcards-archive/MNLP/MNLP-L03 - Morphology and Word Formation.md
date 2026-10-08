# Archived flashcards: MNLP-L03 - Morphology and Word Formation

The long-form cards this note carried until 2026-10-08, when they were replaced by short single-fact cards. Kept for reference; not published.

Click a question to reveal its answer, or press **Study** to drill the whole set. Cards marked as exam questions are meant to be answered out loud or on paper first, then checked against the points listed.

> [!exam]- Name the four morphological types, give example languages for each, and illustrate each with one analysed word or sentence. What distinguishes agglutinative from fusional?
> - **Isolating** (Chinese, Vietnamese): words do not change form; grammar comes from **word order** and **particles**. Chinese 我 昨天 看 了 一 本 书 "I read a book yesterday": 看 never changes, past comes from the particle 了 (PFV) and 昨天 "yesterday".
> - **Agglutinative** (Turkish, Finnish): morphemes strung together, **one morpheme, one meaning**, visible boundaries. Turkish *ev-ler-iniz-den* = house + plural + your + from, "from your houses".
> - **Fusional** (Russian, Spanish, Arabic): **one affix encodes several features** that cannot be separated. Russian *knig-ami* "with books": *-ami* = instrumental and plural at once. Spanish *habl-é*: person, number, tense, aspect in one vowel.
> - **Polysynthetic** (Inuktitut, Mohawk): verb, subject, object and modifiers (negation, tense, mood, instrument, location) packed into **one word**. *Qangatasuukkuvimmuuriaqalaaqtunga* "I will have to go to the airport".
> - **The axis:** isolating and polysynthetic are the two ends of a scale of how much meaning one word carries. Agglutinative and fusional sit in the middle and differ on a separate question: can the packed meaning be cut apart?
>
> Losing marks: defining agglutinative as "more morphemes". It means **separable** morphemes. Fusional languages can be just as dense; defining the classes by density misclassifies Spanish. Also: claiming *-ami* marks feminine gender. It is the instrumental plural for all three genders.

> [!exam]- Why do fixed word-level vocabularies break for morphologically rich languages and for multilingual models, and what is the practical fix?
> - Neural weight matrices are **fixed size**; the output layer is a $|V| \times m$ matrix plus a softmax over $|V|$ at every position, so $|V|$ drives memory.
> - Realistic sizes: English about **200K** words, Russian about **1M**, because Russian marks case on every noun so each lemma has many more surface forms.
> - Inflectional features multiply: forms per lemma = **product** of slot sizes. Productivity (*doomscrolling, doomscrolled, doomscrolls, doomscroller*) means no vocabulary can be closed, so OOV never goes away.
> - **Two horns:** keep all forms and $|V|$ explodes (embedding matrix and softmax with it); or use a vocabulary sized for English and get massive OOV rates, with every rare form mapped to `<unk>` and given **equal probability**.
> - Word indices hide relations (*move* 2863 vs *moved* 87542) and give nothing for unseen words (*hammered*). Morphology could build meaning for unseen words; context alone cannot.
> - **Multilingual:** a shared vocabulary leaves English words whole while Finnish gets fragmented into meaningless pieces, so sentences cost different numbers of tokens: a bias built into the tokenizer before training.
> - Fix: morphological analysers are language dependent and need per-language expertise, so the practical compromise is **subword tokenization**: statistically useful pieces, language-agnostic, learned from data, zero OOV by falling back to characters.

> [!exam]- Run Forward and Backward Maximum Matching on 研究生活 with vocabulary {研究, 研究生, 生活, 生, 活}. Why do the results differ, how does bidirectional matching resolve it, and what are the method's limits?
> 1. **FMM:** at $i=0$, 研究生活 is not in $V$, 研究生 is (3 chars), emit it. At $i=3$, 活 is in $V$. Result **研究生 | 活**.
> 2. **BMM:** at $i=4$, the longest suffix in $V$ is 生活 (2 chars), prepend it. At $i=2$, 研究 is in $V$. Result **研究 | 生活**.
> 3. **Why:** same string, same dictionary, only the scan direction changed. The string is genuinely ambiguous and a greedy rule lets the ambiguity leak through. Neither algorithm is buggy.
> 4. **Bidirectional:** if FMM and BMM agree, accept with high confidence. If not, tie-break: fewer total words, or fewer single-character words. Here both give 2 words (tie), and BMM wins on single-character words (0 against FMM's 1 for 活).
> 5. **Limits:** needs a precompiled per-language vocabulary; new words (names, neologisms, typos) can never be matched; cannot improve from data; OOV words are silently chopped into single characters; genuine ambiguity cannot be fixed by any direction heuristic.

> [!exam]- Reformulate Chinese word segmentation as sequence labelling and explain how an HMM with Viterbi solves it. Give the recursion, the complexity, and what the approach still cannot do.
> - **BMES tags:** B (first char of multi-char word), M (interior of 3+ char word), E (last char of multi-char word), S (single-char word). Boundary after every E and S.
> - **HMM:** hidden tag sequence generates the observed characters. Transitions $P(t_i \mid t_{i-1})$, emissions $P(c_i \mid t_i)$. Decode $\hat{t}_{1:n} = \arg\max_{t_{1:n}} \prod_i P(t_i \mid t_{i-1}) P(c_i \mid t_i)$.
> - **Training:** MLE counts from a segmented corpus: $P(t_i \mid t_{i-1}) = C(t_{i-1}, t_i)/C(t_{i-1})$, $P(c \mid t) = C(t,c)/C(t)$.
> - **Viterbi:** $\text{score}[i][t] = \max_{t'} \text{score}[i-1][t'] \cdot P(t \mid t') \cdot P(c_i \mid t)$, with a backpointer to the winning $t'$; backtrace from the best final cell.
> - **Complexity:** $O(n |T|^2)$, i.e. $16n$ for 4 tags, against $4^n$ brute-force tag sequences.
> - **Gains over maximum matching:** no fixed dictionary, ambiguity resolved by probabilistic evidence, trainable from any segmented corpus.
> - **Still cannot:** first-order Markov assumption (higher orders cost $|T|^{k+1}$ parameters); no longer-range context (motivates CRFs); unsupervised Baum-Welch (EM) training degrades quality. Neural sequence labellers (LSTMs, Transformers) are used in practice.

> [!exam]- Distinguish inflection, derivation and compounding with examples, and explain why productivity of these processes matters for NLP.
> - **Inflection:** keeps the part of speech, changes grammatical features (tense, number). *see, saw, seen*; *book, books*. Another form of the same dictionary entry.
> - **Derivation:** changes POS and/or meaning, giving a new dictionary entry. *happy* → *happiness* (noun), *happy* → *happily* (adverb); *perfect* → *imperfect* (meaning only).
> - **One-line test:** does the part of speech survive? *see / saw* inflection, *happy / happiness* derivation.
> - **Compounding:** several words into one. German *Tisch* + *Bein* = *Tischbein* "table leg"; *sicher* + *gehen* = *sichergehen* "make sure". German writes it as one token where English uses two.
> - **Productivity:** the processes apply iteratively and to new combinations. *doomscroll* is a new compound already carrying inflection (*-ing, -ed, -s*) and derivation (*-er*). No fixed vocabulary can be closed over a productive process, which is the formal reason OOV never disappears.

> [!exam]- What is non-concatenative morphology, and why can no string-cutting segmenter (from maximum matching to byte-pair encoding) fully handle it? Use German, Arabic and Inuktitut.
> - **Concatenative:** word = concatenation of morphemes; string splitting can recover them.
> - **Non-concatenative:** the change happens **inside the root**. Apophony: German *Buch* → *Bücher*, *Haus* → *Häuser* (vowel change plus suffix); English *foot / feet*, *sing / sang / sung*. No cut separates "book" from "plural" in *Bücher*.
> - **Circumfix** (related problem): German *ge-seh-en*: *ge-* and *-en* are one morpheme in two pieces, so neither stripping the prefix nor the suffix alone works.
> - **Arabic:** consonantal root *k-t-b* "write" poured into a pattern *ya-* C1 C2 *-u-* C3 *-u* = *yaktubu* "he writes". Morphemes are **interleaved**. Short vowels are not written, so يكتب cannot distinguish *yaktubu* from *yuktabu* "it is written": part of the information is absent from the string.
> - **Inuktitut:** phonological rules at boundaries rewrite morphemes (*-k-mut-uq* surfaces as *-mmuu-*), so the word cannot be split back into its morphemes by string matching.
> - Every segmentation method cuts a string into **contiguous pieces**, so subword segmentation is a solution to the concatenative case only.

> [!exam]- Word segmentation and subword segmentation are often confused. Define each, give a language where each is needed, and say how they are solved.
> - **Word segmentation (tokenization):** establishing word boundaries in scripts that do not mark them (Chinese, Japanese, Thai, Burmese). Must happen before POS tagging, parsing or anything else. Solved by maximum matching (FMM, BMM, bidirectional) or statistical BMES labelling (HMM plus Viterbi, CRFs, neural taggers). Benchmarked in the SIGHAN bake-offs.
> - **Subword segmentation:** splitting words that are already well delimited into units smaller than the word, because morphologically rich languages (Finnish) blow up fixed vocabularies. Solved statistically, e.g. byte-pair encoding.
> - Both decide where to cut a character string; they differ in what the pieces are meant to be: whole words against units smaller than words.

> [!card]- Define vocabulary, lexicon and dictionary, and say which one an NLP system usually has.
> - **Vocabulary:** the inventory of words in a language, a corpus, or known to a person. A list.
> - **Lexicon:** a store of the **meanings and functions** of words, beyond the word forms.
> - **Dictionary:** a written version (published or digital) of a lexicon.
> - An NLP system almost always has a vocabulary and rarely a lexicon.

> [!card]- What open design questions arise when building a vocabulary, and why are they all really questions about morphology?
> Questions: all words seen in a corpus? language-specific? include typos? include very rare words? all variants (*book, books, booking, booked*) or just *book*? limited in size?
>
> They are all one trade-off: giving related forms separate slots costs vocabulary size and hides that they are related, while refusing slots to rare forms and typos means you need some other way to represent them.

> [!card]- Name the four classical lexical relations with an example each. Why are hand-built resources like WordNet a poor basis for meaning, and what is the alternative?
> - **Synonyms** (same meaning): *purchase* :: *acquire*. **Hyponyms** (is-a): *car* :: *vehicle*. **Meronyms** (part-whole): *wheel* :: *car*. **Antonyms** (opposites): *small* :: *large*.
> - Hand-built resources are **expensive** (every relation entered by a human), **incomplete** (never cover a real corpus) and **language-specific** (cost multiplies per language).
> - Alternative: learn relations automatically and **quantitatively**, e.g. $\text{sim}(\textit{car}, \textit{vehicle}) > \text{sim}(\textit{car}, \textit{tree})$, which is what word embeddings provide.

> [!card]- Define word embeddings by their four properties.
> Distributed representations of words as vectors that are:
> - **low dimensional** (e.g. 512, against $|V|$ of hundreds of thousands)
> - **dense** (no zeros, unlike one-hot or count vectors)
> - **continuous**: $c_w \in \mathbb{R}^m$
> - **learned by performing a prediction task**, rather than counted or hand-assigned.

> [!card]- State the CBOW task. How does it differ from n-gram language modelling, and how does Skip-Gram relate to it?
> - **CBOW** (part of Word2Vec, Mikolov et al.): given position $t$, the $n$ words to the left $\{w_{t-n},\dots,w_{t-1}\}$ and $m$ words to the right $\{w_{t+1},\dots,w_{t+m}\}$, predict $w_t$. Example: `the man X the road`.
> - It resembles n-gram LM with $n$ = LM order $-1$ and $m = 0$, but an LM sees only the left context. CBOW sees both sides, which is fine because its goal is good embeddings and prediction is only the means.
> - **Skip-Gram** is the mirror image: predict each context word from the centre word.

> [!card]- What are the architectural choices in CBOW, and why is it called "bag of words"?
> - Feed-forward network; focus on learning the **embeddings**, not prediction quality; **simple** network so capacity goes into the representation; embedding/projection layer moved **closer to the output**; typically $n = m$ with $n \in \{2, 5, 10\}$.
> - Context word vectors are **summed** in the projection layer, and the sum predicts the centre word.
> - Summation is commutative, so word order is destroyed: `the man X the road` and `road the X man the` give the same projection. The context is a bag. That information loss buys a very cheap model that trains on enormous corpora.

> [!card]- What is a vocabulary short-list, what are typical sizes, and what is its main disadvantage?
> - Keep only the $n$ most frequent words and map every rare word to a single token `<unk>`. Typical sizes: 10K, 50K, 100K, sometimes 200K.
> - Motivation: weight matrices are fixed size, and the output layer ($|V| \times m$ plus softmax over $|V|$ at every position) dominates memory.
> - Disadvantage: all rare words get **equal probability** in a given context, since they are the same token. The model cannot prefer *hammered* over *doomscrolled* in `She ___ the nail`. The whole tail collapses into one symbol.

> [!card]- Why is the English vocabulary about 200K words and the Russian about 1M?
> Russian speakers do not know five times more concepts. Russian marks **case on every noun**, so each lemma surfaces in many more distinct forms, and each form needs its own slot in a word-level vocabulary.

> [!card]- Why is distributional (contextual) information not enough to represent words, and what does morphology add?
> - Contexts pull related words together (*desk* and *table*), but word indices carry no structure: *move* (2863) and *moved* (87542), or *transports* and *transported*, must each have their relation learned from scratch from data.
> - **Unseen or rare words** (*hammered*) have no contexts at all, so context cannot help.
> - Claim: **word-internal structure can induce meaning for unseen words**. Knowing *hammered* = *hammer* + *-ed*, a model with an embedding for *hammer* and knowledge of *-ed* could construct a representation. Context and word-internal structure are complementary; word-level indexing uses only the first.

> [!card]- Why is word segmentation hard and why does it matter for the rest of the pipeline? Give the standard ambiguous example.
> - It must happen **before** POS tagging, parsing and everything else, so errors propagate; it is a task in its own right (SIGHAN bake-offs).
> - The same string often has several valid segmentations: 研究生活动 can be 研究生 | 活动 "graduate-student activities", or cut as 研究 | 生活 | 动 "research on life activities". Whether the second is a natural reading is not settled (it strands 动 as a single character), but it is a lexically possible cut.
> - Even in whitespace languages tokenization has exceptions: `Mr. Smith` must not become `Mr . Smith`, since the period belongs to the abbreviation.

> [!card]- How many segmentations does a string of $n$ characters have? Compute it for 4 and for 20 characters.
> Each of the $n-1$ internal gaps is independently a boundary or not:
> $$\#\text{segmentations} = 2^{\,n-1}$$
> - 4 characters (研究生活): $2^3 = 8$.
> - 20 characters: $2^{19} = 524{,}288$.
>
> Enumeration is impossible at sentence length, so practical methods are greedy (maximum matching) or dynamic programs (Viterbi).

> [!card]- Give Forward Maximum Matching as numbered steps, explain the `or l == 1` clause, and state its complexity.
> Input: string $S$, vocabulary $V$, max word length $L$.
> 1. Set $i \leftarrow 0$, output empty.
> 2. While $i < |S|$: for $l$ from $\min(L, |S| - i)$ **down** to 1, take $w = S[i : i+l]$.
> 3. If $w \in V$ or $l = 1$: append $w$, set $i \leftarrow i + l$, break.
> 4. Return output.
>
> Counting down makes the first hit the longest match. `or l == 1` is the escape hatch: a single character is emitted even if not in $V$, so the algorithm always terminates. It is also where OOV words get silently chopped into characters. Complexity: $O(n \cdot L)$ dictionary lookups.

> [!card]- How does Backward Maximum Matching differ from Forward Maximum Matching?
> Structurally identical, with three changes:
> 1. Start at the **end**: $i \leftarrow |S|$, loop while $i > 0$.
> 2. Take the substring **ending** at $i$: $w = S[i-l : i]$, with $l$ from $\min(L, i)$ down to 1.
> 3. **Prepend** $w$ to the output (so it stays in reading order) and set $i \leftarrow i - l$.

> [!card]- Define the BMES tag set and the rule for turning tags into a segmentation.
> - **B** (Begin): first character of a multi-character word.
> - **M** (Middle): interior character of a word with 3 or more characters.
> - **E** (End): last character of a multi-character word.
> - **S** (Single): a character that is a complete one-character word.
>
> Insert a word boundary after every **E** and every **S**. The problem becomes ordinary sequence labelling.

> [!card]- Write the BMES tags for the FMM output 研究生 | 活 and the BMM output 研究 | 生活. What segmentation does 研/B 究/E 生/S 活/S give?
> - 研究生 | 活 → 研/B 究/M 生/E 活/S.
> - 研究 | 生活 → 研/B 究/E 生/B 活/E.
> - 研/B 究/E 生/S 活/S → 研究 | 生 | 活, a third answer for the same four characters.
>
> The point of labelling is that the right one can be **learned from data** instead of legislated by a scan direction.

> [!card]- What are the two kinds of HMM parameters for word boundary detection? Give an example of each and note the direction of the emission probability.
> - **Transition** $P(t_i \mid t_{i-1})$ (tag bigrams): B is very likely followed by M or E, and can never be followed by another B.
> - **Emission** $P(c_i \mid t_i)$: probability of the character given the tag. $P(\text{究} \mid E)$ may be high because 究 usually closes 研究.
> - Direction matters: emission is character given tag, the reverse of tag given character.

> [!card]- Toy calculation: estimate HMM parameters by MLE from a segmented corpus of two sentences, 研究 | 生活 and 研究生 | 活. Give $P(B \mid \langle s \rangle)$, $P(E \mid B)$, $P(M \mid B)$, $P(\text{研} \mid B)$ and $P(\text{活} \mid E)$.
> Tags: 研究 | 生活 = B E B E; 研究生 | 活 = B M E S.
> - $P(B \mid \langle s \rangle) = 2/2 = 1$ (both sentences start with B).
> - B occurs 3 times, each followed by a tag: B→E twice, B→M once. $P(E \mid B) = C(B,E)/C(B) = 2/3$, $P(M \mid B) = 1/3$.
> - B emits 研 twice and 生 once: $P(\text{研} \mid B) = C(B,\text{研})/C(B) = 2/3$.
> - E occurs 3 times (究, 活, 生), emits 活 once: $P(\text{活} \mid E) = 1/3$.
>
> Formulas: $P(t_i \mid t_{i-1}) = C(t_{i-1}, t_i)/C(t_{i-1})$ and $P(c \mid t) = C(t,c)/C(t)$.

> [!card]- Which BMES transitions are structurally possible, and why does that help a first-order HMM?
> - B → M or E only. M → M or E only. E → B or S only. S → B or S only.
> - Impossible (probability 0): B→B, B→S, M→B, M→S, E→M, E→E, S→M, S→E.
> - Reason: a word opens with B, continues with zero or more M, closes with E; so B and M must be followed by something inside the same word, E and S by something that starts a new word.
> - Half the transition matrix is effectively zero, and the model learns this from data without being told, which is why even a first-order model does well.

> [!card]- Give the Viterbi algorithm for BMES tagging as numbered steps and say what backpointers are for.
> 1. Initialise $\text{score}[0][\langle s \rangle] = 1$, all other $\text{score}[0][t] = 0$.
> 2. For each position $i = 1 \dots n$ and each tag $t \in \{B,M,E,S\}$: $\text{score}[i][t] = \max_{t'} \text{score}[i-1][t'] \cdot P(t \mid t') \cdot P(c_i \mid t)$.
> 3. Store a backpointer to the $t'$ that achieved the max.
> 4. Backtrace from the highest-scoring final tag to recover the full sequence.
>
> $\text{score}[i][t]$ is the probability of the best path ending at position $i$ with tag $t$. Backpointers avoid recomputing whole paths: only the winning predecessor is stored, and the pointers are walked backwards.

> [!card]- State the complexity of Viterbi for BMES tagging and compare it with brute force for a 4-character string.
> $O(n \times |T|^2)$: at each of $n$ positions, each of $|T|$ tags is compared against each of $|T|$ predecessors. With $|T| = 4$ that is $16n$, linear in sentence length.
>
> For 4 characters: $16 \times 4 = 64$ predecessor comparisons, against $4^4 = 256$ complete tag sequences by brute force. The gap grows exponentially with $n$.

> [!card]- What are the advantages of an HMM over FMM and BMM for word boundary detection?
> - **No fixed dictionary** needed: it works over characters, so an unseen word is just a character sequence with a plausible tag path.
> - **Ambiguity resolved by probabilistic evidence** instead of direction-based tie-breaking heuristics.
> - **Trainable directly from any segmented corpus**, even from another domain.

> [!card]- What are the shortcomings of HMMs for word boundary detection, and what replaces them?
> - **First-order Markov assumption:** a tag depends only on the previous tag and the current character. Higher orders are possible but parameters grow as $|T|^{k+1}$.
> - **No longer-range context** (the word two positions back, the surrounding phrase). Motivates **CRFs**, which condition on arbitrary features of the whole input.
> - **Unsupervised training** with Baum-Welch (an instance of EM) is possible but degrades quality.
> - In practice, neural sequence labellers (LSTMs, Transformers) are used.

> [!card]- What is the morphological status of a word whose part of speech is unchanged but whose meaning is negated, e.g. *function* → *dysfunction* or *perfect* → *imperfect*?
> **Derivation.** Derivation changes the POS **and/or** the meaning, so a negating prefix that keeps the POS is still derivational: it yields a new dictionary entry. (Course material spells the example *disfunction*; the standard spelling is *dysfunction*, from Greek *dys-*. The morphological point is unaffected.)

> [!card]- Give the five forms of an English verb with *see* and *call*. How many distinct strings does each paradigm have, and what does that say about English?
> Present, simple past, past participle, present participle, 3rd person singular:
> - *see, saw, seen, seeing, sees*: 5 slots, 5 distinct forms.
> - *call, called, called, calling, calls*: 5 slots, **4** distinct forms (past = past participle). Likewise *send* (*sent, sent*).
>
> English inflection is impoverished, which is why English-shaped assumptions transfer badly to other languages.

> [!card]- What does English noun inflection mark, and what is its crucial limitation?
> - **Number:** *book, books*; *house, houses*.
> - **Case on pronouns only:** *he*, *him* (accusative), *his* (possessive), *them* (accusative plural), *their* (possessive plural).
> - English nouns do not mark case at all, so an English-trained intuition has no slot for it.

> [!card]- Give three inflectional features other languages mark that English does not (or marks only marginally), with examples.
> - **Case on all nouns:** Russian (more cases than English, marked on every noun), e.g. *knigami*.
> - **Mood** in German: *kommt* (indicative), *käme* (subjunctive), *komm* (imperative).
> - **Aspect** in Russian, Czech, Polish: Russian *sdelat'* (perfective, completed) against *delat'* (imperfective, ongoing). English has no inflectional marking for it.

> [!card]- Why do inflectional features multiply rather than add the number of word forms? Work through the numbers.
> Features are independent slots, so forms per lemma = **product** of slot sizes.
> - English noun: 2 numbers, no case: $2$ forms.
> - Add a six-way case system: $2 \times 6 = 12$.
> - Add possessive marking with six persons: $12 \times 6 = 72$ forms from one lemma, before any derivation.
>
> This is the arithmetic behind English 200K against Russian 1M.

> [!card]- Give English derivational patterns for nominalization, adjectivization and negation, with examples.
> - **Nominalization:** verb + *-ation* (*derivation*); verb + *-er* (*killer*, *baker*); adjective + *-ness* (*happiness*).
> - **Adjectivization:** verb + *-able* (*accountable*, *reasonable*); noun + *-al* (*parental*, *colonial*, *official*).
> - **Negation** (meaning changes, POS kept): *un-* (*unseen*, *unheard*), *mis-* (*misjudge*, *misappropriation*), *dis-*/*dys-* (*dysfunctional*), *im-* (*implausible*), *in-* (*indifferent*).

> [!card]- Why does a naive "strip the suffix and look up the stem" analyser fail on English derivation? Give two phenomena.
> - **Spelling changes at the boundary:** *colony* + *-al* = *colonial* (not *colonyal*); *office* + *-al* = *official*. The morpheme is regular but the surface string is not.
> - **Conditioned variants of one morpheme:** *im-* before labials (*implausible*), *in-* elsewhere (*indifferent*). A model treating them as unrelated strings misses that they do the same job.

> [!card]- Define a morpheme and decompose *disproportionally*, saying what each piece contributes.
> The **smallest meaning-carrying part of a word**.
>
> *dis* + *proportion* + *al* + *ly* = prefix + stem + suffix + suffix: negation, core meaning, POS change to adjective, POS change to adverb.

> [!card]- Distinguish root from affix and free from bound morphemes. Why does the free/bound distinction matter for segmentation?
> - **Root** (free): determines the basic meaning and can stand alone.
> - **Affix** (bound): attaches to a root to change meaning or grammatical function, cannot stand alone.
> - Free morphemes are words on their own (*proportion*, *book*, *see*); bound ones are not (*dis-*, *-al*, *-ly*, *-ed*).
> - Bound morphemes are exactly the pieces that never appear as standalone tokens in a corpus, so a word-level system never sees them on their own.

> [!card]- Name the four kinds of affix by position, with an example of each. What is the caveat about the English infix example?
> - **Prefix:** front; in English mostly one per word (*dis-* in *disproportionally*).
> - **Suffix:** end; can stack (*-al*, *-ly*).
> - **Infix:** inserted inside the word; more common in other languages. English example *passerby* → *passersby*.
> - **Circumfix:** front and back simultaneously: German *ge-seh-en*, past participle of *sehen*. *ge-* and *-en* are one morpheme in two pieces, so it cannot be analysed by stripping a prefix or suffix independently.
>
> Caveat: *passersby* is really plural marking on the head *passer* inside a compound. English has almost no true infixes; languages like Tagalog have productive infixation of a genuine morpheme inside the root.

> [!card]- Isolating languages: describe the type and analyse the Chinese and Vietnamese examples. What do they mean for a tokenizer?
> - Words do not change form; grammatical relations come from **word order**; **particles** add modification. Chinese, Vietnamese.
> - Chinese 我 昨天 看 了 一 本 书 "I read a book yesterday": 看 *kàn* never changes; past comes from particle 了 (perfective, PFV) and 昨天 "yesterday"; SVO order carries the relations; 本 *běn* is an obligatory **classifier** (CL) between numeral and noun.
> - Vietnamese *Tôi đã đi học* "I went to study": *đi* and *học* do not change; *đã* is a past (PST) particle.
> - Every morpheme is already a separate token: the easy case for word-level vocabularies, the hard case for anything expecting tense to be recoverable from the verb.

> [!card]- Analyse Turkish *evlerinizden* and Finnish *taloissamme*. What makes the agglutinative type learnable, and where are the boundaries not clean?
> - *ev-ler-iniz-den* = house + plural + your + from, "from your houses": one word for a four-word English phrase, each suffix one job.
> - *talo-i-ssa-mme* = house + plural + inessive + our, "in our houses". Inessive = locative "in"; other locative cases include elative "out of", illative "into", allative "to".
> - Learnable because suffixes are separable and **reusable across nouns**: a segmenter that finds *-ssa* once can apply it everywhere.
> - Not clean at the character level: *kala* "fish" + *-i-* + *-ssa* = *kaloissa*, with stem vowel *a* becoming *o*. Boundaries are clean at the level of morphemes, not always of characters.

> [!card]- Analyse Russian *knigami* and Spanish *hablé*. Why can a string-based segmenter not recover their features, and what is wrong with glossing *-ami* as feminine?
> - *knig-ami* "with books": *-ami* fuses **case** (instrumental) and **number** (plural). No substring of *-ami* means "plural", so no string segmenter can recover the features even in principle.
> - *habl-é* "I spoke": *-é* fuses person (1st), number (singular), tense (preterite, past) and aspect (completed). Four features, one vowel.
> - Gender: Russian neutralises gender in the plural; *-ami* is used for all three (*stolami* masculine, *oknami* neuter, *knigami* feminine). An ending that really fuses case, number and gender is the instrumental singular: feminine *knig-oy* against masculine *stol-om*.

> [!card]- Explain Arabic root-and-pattern morphology with *yaktubu*, and why Arabic is the worst case for a naive tokenizer.
> - Root = three consonants *k-t-b* "write"; pattern = template *ya-* C1 C2 *-u-* C3 *-u*; result *ya-k-t-u-b-u* = *yaktubu* "he writes".
> - The pattern fuses person (3rd), gender (male) and aspect (not completed).
> - Morphemes are **interleaved**, so no set of cuts recovers them (non-concatenative).
> - Short vowels are not written: in يكتب the prefix ي is written but the vowels are not, so the string cannot distinguish *yaktubu* "he writes" from *yuktabu* "it is written", and the mood ending *-u* is invisible.
> - A subword tokenizer learns consonant clusters and cannot represent the pattern as a unit.

> [!card]- Analyse the Inuktitut word *Qangatasuukkuvimmuuriaqalaaqtunga* and say what polysynthesis implies for vocabulary size.
> - *qangata-* fly; *-suukkuvik* (*-suu-* habitually + *-kkuvik* place) = airport; *-mut* to (allative); *-uq-* go to; *-riaqaq-* have to; *-laaq-* future; *-tunga* 1st person singular. "I will have to go to the airport".
> - One word = a whole English sentence; it contains its own internal derivation (fly + habitual place = airport).
> - Phonological rules rewrite boundaries (*-k-mut-uq* → *-mmuu-*), so string matching cannot split it.
> - Word types number closer to **sentences** than to morphemes, so no vocabulary size (not 200K, not 1M) covers the language.

> [!card]- Is a morphologically rich language more complex than English? Explain with Turkish.
> No. Rich morphology encodes what morphologically poor languages encode through **word order** and separate words. Turkish *evlerinizden* and English "from your houses" carry the same information, one in suffixes, the other in separate words and their arrangement. Only one of them suits a whitespace tokenizer.

> [!card]- Why are morphological analysers not the general solution to the vocabulary problem, and what does subword tokenization guarantee and not guarantee?
> - Analysers are **highly language dependent**: one per language, built with per-language linguistic expertise (the WordNet cost problem again). Modern approaches are data-driven.
> - **Subword tokenization:** find statistically useful pieces, language-agnostically, from data, without seeking linguistically correct morphemes.
> - Guarantees: a fixed vocabulary with **zero OOV**, by falling back to characters.
> - Does not guarantee: alignment with morphemes. It does not recover *knig* + *ami* or *k-t-b* + pattern; alignment is "somewhat, in concatenative languages".
