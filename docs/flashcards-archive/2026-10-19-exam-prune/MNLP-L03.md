Cards removed on 2026-10-19 from the Flashcards section of Courses/MNLP/Lectures/MNLP-L03 - Morphology and Word Formation.md, when the set was pruned to what the multiple-choice exam on 22 Oct 2026 can ask: on the slides, decidable as a statement, not a number, not a duplicate, not obscure.

## Duplicate

> [!card]- Why does CBOW have no non-linear hidden layer?
> So the embedding layer sits right next to the output and the **embeddings** do most of the work.
Reason: duplicate of "True or false: In CBOW the summed context vectors go from the projection layer straight to the output layer."

> [!card]- Why is CBOW called "bag of words"?
> Its projection layer **sums** the context vectors, and summation is commutative, so the **order** of the context words is lost.
Reason: duplicate of "True or false: CBOW's projection layer loses the order of the context words."

> [!card]- What is the main disadvantage of a short-list with `<unk>`?
> All rare words get **equal probability** in a given context, because they are literally the same token.
Reason: duplicate of "True or false: With a vocabulary short-list, rare words keep distinct probabilities in a given context."

> [!card]- How many possible segmentations does a string of $n$ characters have?
> $2^{n-1}$: each of the $n-1$ internal gaps is a boundary or not. Too many to enumerate.
Reason: duplicate of the same card in L04, where the slides give the formula; the L03 slide only says the space is huge

> [!card]- What does maximum matching do at each position of a character string?
> It greedily takes the **longest word** found in a precompiled vocabulary.
Reason: duplicate of "Multiple selection: which are true of Forward Maximum Matching?" option A

> [!card]- What does maximum matching do when no vocabulary word starts at a position?
> Emits a **single character** even if it is not in the vocabulary, so the algorithm always terminates.
Reason: duplicate of "Multiple selection: which are true of Forward Maximum Matching?" option C

> [!card]- What happens to an out-of-vocabulary word under maximum matching?
> It is **silently** chopped into single characters or greedily mismatched; nothing in the output signals it.
Reason: duplicate of "True or false: Maximum matching flags out-of-vocabulary words in its output."

> [!card]- Why can Forward and Backward Maximum Matching give different segmentations of the same string?
> The string is **genuinely ambiguous**, and the greedy rule lets that leak through; only the scan direction differs. Neither algorithm is buggy.
Reason: duplicate of "Why can no direction-based heuristic segment correctly in general?" and the FMM/BMM true/false card

> [!card]- After which BMES tags is a word boundary inserted?
> After every **E** and every **S**.
Reason: duplicate of "Multiple selection: which are true of BMES tagging with an HMM?" option A

> [!card]- What segmentation does the tagging 研/B 究/E 生/S 活/S give?
> **研究 | 生 | 活**
Reason: tests the same rule as the boundary-after-E-and-S card (MS option A)

> [!card]- What is the direction of the emission probability in a BMES HMM?
> $P(\text{character} \mid \text{tag})$: the character given the tag.
Reason: duplicate of "True or false: In a BMES HMM, the emission probability is the probability of the character given the tag."

> [!card]- Why can the BMES tag B never be followed by another B?
> A word opened by B must continue with M or **close with E** before a new word can begin.
Reason: duplicate of "Multiple selection: which are true of BMES tagging with an HMM?" option B

> [!card]- What are backpointers for in Viterbi?
> They store each cell's **winning predecessor**, so the best path is recovered by backtracing instead of recomputing paths.
Reason: duplicate of "Multiple selection: which are true of BMES tagging with an HMM?" option C

> [!card]- How does the cost of Viterbi compare with scoring every tag sequence?
> Viterbi is $O(n|T|^2)$, **linear** in sentence length; brute force is **exponential**, $|T|^n$.
Reason: duplicate of "True or false: Viterbi decoding of BMES tags runs in time linear in the sentence length."

> [!card]- Why does an HMM handle unseen words better than maximum matching?
> It needs **no fixed dictionary**: it works over characters, so an unseen word is just a plausible tag path.
Reason: duplicate of "True or false: An HMM word segmenter needs a fixed dictionary of possible words."

> [!card]- How does an HMM resolve segmentation ambiguity, compared with bidirectional matching?
> With **probabilistic evidence** learned from data, instead of direction-based tie-breaking heuristics.
Reason: duplicate of "Multiple selection: which are true of HMM word segmentation?" option D

> [!card]- What does the first-order Markov assumption restrict in an HMM segmenter?
> Each tag is generated from the **previous tag** alone (equivalently: a tag depends only on the previous tag and the current character), so no longer-range context is used.
Reason: duplicate of "Multiple selection: which are true of HMM word segmentation?" option A

> [!card]- Which model is motivated by the HMM's lack of longer-range context in word segmentation?
> **Conditional random fields (CRFs)**, which can condition on features of the whole input.
Reason: duplicate of "Multiple selection: which are true of HMM word segmentation?" option C

> [!card]- What is the difference between inflection and derivation?
> Inflection **keeps the part of speech** and changes grammatical features; derivation changes the **part of speech and/or the meaning**.
Reason: duplicate of "True or false: Inflection keeps a word's part of speech." and the inflection/derivation multiple-selection card

> [!card]- *perfect* → *imperfect* keeps the part of speech. Is it inflection or derivation?
> **Derivation**: it changes the meaning and yields a new dictionary entry.
Reason: duplicate of "Multiple selection: which are correct? (A) *books* ..." option B

> [!card]- What is compounding?
> Combining **several words into one word**, e.g. German *Tisch* + *Bein* = *Tischbein* (table leg).
Reason: duplicate of "Multiple selection: which are correct? (A) *books* ..." option C

> [!card]- What is non-concatenative morphology?
> Morphology that changes the **root itself**, instead of stringing morphemes together.
Reason: duplicate of "True or false: Apophony is a kind of concatenative morphology."

> [!card]- What is the name for a vowel change inside the root, as in German *Buch* → *Bücher*?
> **Apophony**, a kind of non-concatenative morphology.
Reason: duplicate of "True or false: Apophony is a kind of concatenative morphology."

> [!card]- Why can no string-cutting segmenter fully handle non-concatenative morphology?
> Segmenters cut **contiguous pieces**, but the change is **inside the root**: no cut in *Bücher* separates "book" from "plural".
Reason: duplicate of "True or false: Subword segmentation recovers the morphemes of non-concatenative forms such as German *Bücher*."

> [!card]- Which morphological type has words that tend not to change form at all?
> **Isolating** (Chinese, Vietnamese).
Reason: duplicate of "Multiple selection: which are true of morphological types?" option C

> [!card]- Which morphological type has each morpheme contribute one clear meaning, with visible boundaries?
> **Agglutinative** (Turkish, Finnish).
Reason: duplicate of "Multiple selection: which are true of morphological types?" option A

> [!card]- What distinguishes agglutinative from fusional morphology?
> **Separability**: an agglutinative affix carries one feature; a fusional affix carries several fused features.
Reason: duplicate of "True or false: Agglutinative languages differ from fusional languages mainly in having more morphemes per word."

> [!card]- Why are agglutinative suffixes easy to reuse across words?
> Each suffix is **separable** and carries one meaning, so it attaches to many other nouns.
Reason: near-duplicate of the agglutinative separability cards (MS option A, agglutinative/fusional true/false)

> [!card]- In the course's gloss, which features does the Russian suffix *-ami* in *knigami* ("with books") fuse?
> **Case** (instrumental), **number** (plural) and **gender** (feminine), fused and unsegmentable.
Reason: duplicate of "Multiple selection: which are true of morphological types?" option B

> [!card]- What are the two ways a word-level vocabulary fails for morphologically rich languages?
> Keep all forms and **$|V|$ explodes**; keep an English-sized vocabulary and get massive **OOV** rates.
Reason: duplicate of "True or false: A fixed word vocabulary sized for English produces massive OOV rates for morphologically rich languages."

> [!card]- What does falling back to single characters give subword tokenization?
> **Very low OOV rates** with a fixed-size vocabulary; unseen characters can still be OOV unless it works on bytes.
Reason: duplicate of "True or false: Subword tokenization that falls back to single characters guarantees a zero OOV rate."

> [!card]- True or false: An HMM word segmenter can be trained without a segmented corpus, at a cost in quality.
> **True.** Unsupervised Baum-Welch (EM) training is possible, but quality drops.
Reason: duplicate of "Multiple selection: which are true of HMM word segmentation?" option B

> [!card]- True or false: Derivation always changes a word's part of speech.
> **False.** It changes POS **and/or** meaning: *perfect* → *imperfect* keeps the adjective.
Reason: duplicate of "Multiple selection: which are correct? (A) *books* ..." option B

## Not on the slides

> [!card]- What is the difference between CBOW and Skip-Gram?
> CBOW predicts the **centre word from its context**; Skip-Gram predicts **each context word from the centre word**.
Reason: not on these slides: L03 only names Skip-Gram, L05 explains it

> [!card]- Which tags can follow E or S in BMES tagging?
> Only **B or S**: a word has just ended, so the next one must start.
Reason: not on slides: the slide only says B is followed by M or E; the E/S rule is the note's own derivation

> [!card]- How many of the 16 BMES tag transitions are structurally impossible?
> **Half**; a model trained on segmented text learns these zeros from the counts.
Reason: not on slides: the count of impossible transitions is the note's own derivation

> [!card]- What is the Viterbi recursion for BMES tagging?
> $\text{score}[i][t] = \max_{t'} \text{score}[i-1][t'] \cdot P(t \mid t') \cdot P(c_i \mid t)$
Reason: formula not on slides (the slide defines the cell in words); recall of the formula is not MC material

> [!card]- Why do inflectional features multiply the number of word forms instead of adding to it?
> Features are **independent slots**, so forms per lemma are the **product** of the slot sizes.
Reason: not on slides: the multiplying-slots argument is the note's own

> [!card]- With a subword vocabulary shared across languages, how are morphologically rich languages treated compared with English?
> They get split into **many more pieces per word** than English.
Reason: not on the L03 slides; the fertility cards in L04 cover it

## Obscure or side detail

> [!card]- Why is a realistic Russian word vocabulary much larger than an English one?
> Russian **inflects far more** (for example case on every noun), so each lemma has many more surface forms.
Reason: side example: the slide gives Russian only as a vocabulary size (a number); the reason is the note's inference

> [!card]- What are the morphemes of Turkish *evlerinizden* ("from your houses")?
> *ev-ler-iniz-den*: **house + plural + your + from**.
Reason: single example word gloss; the type it illustrates is covered by the morphological-types cards

> [!card]- How does Arabic root-and-pattern morphology form *yaktubu* ("he writes")?
> The consonantal root **k-t-b** ("write") is poured into a **pattern** template of prefixes and vowels, *ya-* C1 C2 *-u-* C3 *-u*.
Reason: single example word; Arabic as fusional is kept in "True or false: Arabic is classed as a fusional language."

> [!card]- Why is written Arabic hard for a tokenizer beyond its interleaved morphemes?
> The pattern's **vowels are not written**, so part of the grammar is absent from the string.
Reason: side remark on one language's script

> [!card]- Why can Inuktitut words not be split back into morphemes by string matching?
> The morphemes **change where they meet** (*-k-mut-uq* surfaces as *-mmuu-*).
Reason: single-language side example (-k-mut-uq becomes -mmuu-)
