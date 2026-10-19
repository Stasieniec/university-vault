Cards removed on 2026-10-19 from the Flashcards section of Courses/MNLP/Lectures/MNLP-L05 - Static Embeddings.md, pruning the set to what the multiple-choice exam (22 Oct 2026) can ask.

## Not on the slides (paper, outside material or the note's own reasoning)

> [!card]- Why is word similarity measured with cosine rather than the raw dot product?
> Cosine **ignores vector length**, so a frequent word's large counts do not dominate; only proportions matter.
Reason: not on slides: the reason for cosine over the dot product is the note's own

> [!card]- Do Skip-gram's $2n$ output layers produce different distributions?
> **No.** They share $W'$ and $h$; only the target each is scored against differs.
Reason: not stated on slides: the note's own inference from the shared $W'$

> [!card]- Which vectors does one step of Skip-gram with negative sampling update?
> The input word's $v_w$ and the **$k+1$ output vectors** of the true context and the negatives.
Reason: not on slides: the slides give only the objective, not which vectors a step updates

> [!card]- What is the difference between PCA and t-SNE for visualising embeddings?
> PCA: **linear**, top-variance directions. t-SNE: **non-linear**, keeps local neighbours, gives up global distances.
Reason: not on slides: the slides only name PCA and t-SNE; the linear against non-linear contrast is outside material

> [!card]- Why does fastText wrap each word in `<` and `>`?
> So **prefixes and suffixes are distinct n-grams**: `<ru` only starts a word, `ns>` only ends one.
Reason: not on slides: the slide shows `<w>`, but the reason for the boundary markers is not given

> [!card]- Why does $r_T(x)$ not change which translation CSLS picks for a given source word?
> It is **the same for every candidate** $y$.
Reason: not on slides: the note's own reasoning about $r_T(x)$

> [!card]- Why does MUSE's synthetic dictionary keep only mutual nearest neighbours?
> A pair is trusted only if each word picks the other, which **filters out hub matches**.
Reason: not on slides: the slide says only "keep high-confidence mutual matches"; the hub-filtering reason is the note's own

> [!card]- Why does marking type alone not explain where unsupervised alignment fails?
> **French is mixed-marking** yet aligns well with English; Estonian and Finnish are also agglutinative with 10+ cases.
Reason: not on slides: the note's own commentary on the language table

> [!card]- Why does eigenvector similarity not need a bilingual dictionary?
> Laplacian eigenvalues **do not change when nodes are relabelled**.
Reason: not on slides: the slide says only that $\Delta$ needs no alignment; the relabelling reason is the note's own

> [!card]- How does eigenvector similarity $\Delta$ relate to adversarial alignment success?
> It **tracks it**: the lowest-$\Delta$ pair aligns best, the highest-$\Delta$ pairs fail.
Reason: not stated on slides: the note's reading of the table (the slide says only that eigenvector similarity is poor for most languages)

## Duplicate (same fact tested by a kept card, preferring true/false and multiple-selection cards)

> [!card]- How is the CBOW hidden layer $h$ computed from the context words?
> As the **average** of the $2n$ context embeddings, $h = \frac{1}{2n} W w_C$, with no non-linearity.
Reason: duplicate of "True or false: In Skip-gram, the hidden layer is the average of the context word embeddings."

> [!card]- What task replaces full-vocabulary prediction in negative sampling?
> **Binary classification**: is this (word, context) pair from the data or from a noise distribution?
Reason: duplicate of the negative sampling multiple-selection card, option A

> [!card]- How does negative sampling change the per-example cost of Word2Vec training?
> From **$|V|$ dot products to $k+1$**.
Reason: duplicate of the negative sampling multiple-selection card, option C

> [!card]- How many negative samples does Skip-gram add per observed pair?
> **5 to 20.**
Reason: duplicate of the negative sampling multiple-selection card, option B

> [!card]- Why are Word2Vec negatives drawn from $P(w)^{0.75}$ rather than the raw unigram distribution?
> To **bias sampling towards rarer words**; otherwise nearly every negative would be `the`, `of` or `and`.
Reason: duplicate of "True or false: Raising the unigram distribution to the power 0.75 biases negative sampling towards frequent words."

> [!card]- What weakness of prediction-based methods like Word2Vec does GloVe address?
> They treat **each context window as an independent event**, repeatedly rediscovering the same association.
Reason: duplicate of the GloVe multiple-selection card, option C

> [!card]- In GloVe's weighted least-squares objective, what is $w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j$ fitted to?
> The **log co-occurrence count** $\log X_{ij}$.
Reason: duplicate of the GloVe multiple-selection card, option A

> [!card]- What is GloVe's final embedding for word $i$?
> **$w_i + \tilde{w}_i$**: the word and context vectors summed.
Reason: duplicate of the GloVe multiple-selection card, option D

> [!card]- How does fastText compute a word's vector?
> As the **sum of its character n-gram vectors**, plus the whole word.
Reason: duplicate of the fastText multiple-selection card, option A

> [!card]- How do fastText's gains on semantic analogies compare with its gains on syntactic ones?
> Syntactic analogies gain; **semantic ones barely change**, and for German get worse.
Reason: duplicate of the fastText multiple-selection card, option C

> [!card]- What does the shared space hypothesis claim?
> Separately trained spaces are **approximately isomorphic**, so **a general linear map** can align them.
Reason: duplicate of "True or false: The shared space hypothesis says the two embedding spaces differ only by a rotation."

> [!card]- Which CSLS term penalises hubs?
> **$r_S(y)$**: the candidate's average cosine to its $K$ nearest mapped source vectors.
Reason: duplicate of the hubness and CSLS multiple-selection card, option B

> [!card]- Why does MUSE feed the discriminator only the 50k most frequent words?
> **Rare words have poorly estimated embeddings**; frequent ones give a cleaner signal.
Reason: duplicate of "True or false: In MUSE, the discriminator only sees vectors of the 50k most frequent words of each language."

> [!card]- What does re-orthogonalisation do in MUSE's adversarial training?
> After each step it **pulls $W$ back towards an orthogonal matrix** ($\beta = 0.01$).
Reason: duplicate of the MUSE multiple-selection card, option B

> [!card]- How does MUSE build its synthetic dictionary for refinement?
> From **mutual CSLS nearest neighbours** among frequent words, keeping only high-confidence pairs.
Reason: duplicate of the MUSE multiple-selection card, option D

> [!card]- Why is MUSE's refinement step called semi-supervised?
> It runs **supervised orthogonal Procrustes (SVD) on a dictionary the system built itself**.
Reason: duplicate of "Is MUSE a supervised, semi-supervised or unsupervised method?" and the MUSE multiple-selection card, option D

> [!card]- Which language pairs did adversarial alignment fail on completely in Søgaard et al. (2018)?
> **English with Estonian, Finnish and Greek**: the mixed- and double-marking languages.
Reason: duplicate of "True or false: Søgaard et al. found that adversarial alignment fails for mixed- and double-marking languages (Estonian, Finnish, Greek)."

> [!card]- Multiple selection: which are true of unsupervised (adversarial) alignment, according to Søgaard et al. (2018)? (A) English-Finnish fails even with far more Finnish data (B) It survives a domain mismatch if the languages are close (C) It works when one language uses Skip-gram and the other CBOW (D) Estonian-Finnish aligns adversarially
> **A and D.** B: cross-domain English-Spanish collapses to about zero. C: mixing Skip-gram and CBOW fails.
Reason: duplicate of the true/false cards on English-Finnish data size, Estonian-Finnish and Skip-gram against CBOW, and of the domain mismatch card

## Number or formula (on the cheat sheet)

> [!card]- What are the typical parameters of GloVe's weighting function?
> **$x_\text{max} = 100$, $\alpha = 0.75$**.
Reason: number, on cheat sheet

> [!card]- What character n-gram lengths does fastText use by default?
> **3 to 6**.
Reason: number, on cheat sheet

> [!card]- What is the closed-form solution of the unconstrained least-squares mapping $\min_W \sum_i \lVert W x_i - y_i \rVert^2$?
> $W^* = Y X^\top (X X^\top)^{-1}$.
Reason: formula, on cheat sheet

> [!card]- What is the CSLS score between a mapped source vector $x$ and a target vector $y$?
> $\text{CSLS}(x, y) = 2\cos(x, y) - r_T(x) - r_S(y)$.
Reason: formula, on cheat sheet; what CSLS does is tested by the hubness and CSLS multiple-selection card

## Obscure or side detail

> [!card]- How does idf weighting, $\log(N/n_w)$, help count-based context vectors?
> A context word found in nearly every context gets **idf close to 0**, so it is effectively removed.
Reason: side remark: the slide only says idf "could" be used

> [!card]- Why is CBOW useless as a language model?
> It also uses words **to the right** of the target; a language model uses left context only.
Reason: side remark: the slide only notes CBOW is "seemingly similar" to n-gram language modelling

> [!card]- How do the n-grams of a word share the gradient in fastText?
> **Every n-gram gets the same gradient**, since the word vector is their sum.
Reason: implementation detail from a pseudocode comment; the n-gram sum is tested by the fastText multiple-selection card

> [!card]- What is building a bilingual dictionary by nearest-neighbour search in a crosslingual embedding space called?
> **Bilingual lexicon induction** (BLI).
Reason: term appears only as a figure axis label; the slides say "dictionary induction"

> [!card]- Why does MUSE use label smoothing ($s = 0.2$) on the discriminator?
> To stop it becoming **pathologically overconfident**.
Reason: discriminator architecture detail; $s = 0.2$ is on the cheat sheet

> [!card]- What can MUSE's unsupervised validation criterion not guarantee?
> That the induced pairs are **correct translations**; it only shows mapped words land near real ones.
Reason: side remark (a question mark on the slide)

> [!card]- In a clause, which word is the head and which are the dependents?
> Head: the **verb**. Dependents: **subject and object**.
Reason: example relationship; the head and dependent distinction is kept in "What is the difference between head-marking and dependent-marking?"

> [!card]- For English-Hungarian unsupervised alignment, which parts of speech do worst?
> **Nouns and verbs**, the most inflected.
Reason: table detail for one language pair
