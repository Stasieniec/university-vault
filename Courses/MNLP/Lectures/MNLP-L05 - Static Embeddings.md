---
type: lecture
course: 5204MNLP6Y
week: 3
lecture: 5
date: 2026-09-14
status: complete
topics:
  - Distributional semantics and count-based context vectors
  - Cosine similarity
  - Word2Vec CBOW
  - Word2Vec Skip-gram
  - Negative sampling
  - GloVe
  - Intrinsic and extrinsic evaluation of embeddings
  - Word similarity and analogy benchmarks
  - Visualising embeddings with t-SNE and PCA
  - fastText character n-gram embeddings
  - Cross-lingual embeddings and the shared space hypothesis
  - Seed dictionaries and word alignment
  - Least-squares linear mapping
  - Orthogonal Procrustes mapping
  - Hubness and CSLS
  - Adversarial unsupervised alignment (MUSE)
  - Self-learning refinement
  - Dependent marking and the limits of isomorphism
  - Eigenvector similarity
---

# MNLP-L05: Static Embeddings

> [!abstract] In short
> How do you give every word a vector, so that nearby vectors mean similar words, using only raw text? By the company the word keeps: count its context words, train a small network to predict them (Word2Vec, made affordable by negative sampling), or fit vectors to co-occurrence counts gathered over the whole corpus (GloVe). fastText builds each word's vector from its character n-grams, so related word forms share parts and an unseen word still gets a vector.
>
> The second half joins two languages' spaces. If they have roughly the same shape, one rotation should put `king` next to `rey`. The map can be learned from a seed dictionary (a translation matrix, then an orthogonal one solved exactly by SVD), translations retrieved with CSLS so that "hub" words stop winning everything, or the map learned with no dictionary at all by adversarial training (MUSE). That last method works for similar languages and breaks down for very different morphology, for corpora from different domains and for embeddings from different algorithms.
>
> "Static" means one vector per word type, whatever the sentence; [[MNLP-L06 - Contextual Embeddings]] removes that limit. The sample exam has no question on this lecture; expect statements about what each method does, which data it uses, what is fixed or constrained, and when it fails.

## The idea: a word is known by its contexts

We want word representations that let us **compute similarity along several dimensions**: meaning (synonyms such as `car` and `automobile`, topical relatedness such as `car` and `road`) and morphology (`run`, `runs`, `running`).

> [!definition] Distributional hypothesis
> "*You shall know a word by the company it keeps.*" (Firth, 1957)
>
> Produce vector representations of words **based on the contexts in which they occur**: words in similar contexts get similar vectors, and are taken to have similar meanings.

There are **two types of approach**: **count-based** (**distributional semantics**), which counts the context words directly, and **prediction-based** (**neural embeddings**), which trains a network to predict words from contexts, or the reverse, and keeps its weights as the vectors. GloVe combines the two.

### Counting the company a word keeps

> [!definition] Count-based context vectors
> - Define a vocabulary $V_c$ of **context words** and a vocabulary $V_t$ of **target words** (they can be the same).
> - Define a **window of size $n$** (a sliding window; the context can also be the sentence or document).
> - For each occurrence of $w \in V_t$, count how often each $w' \in V_c$ occurs **within $n$ words to the left or right** of $w$.
> - Store the counts in a vector $c_w$ of length $|V_c|$: $c_w[i]$ is the frequency of context word $i$ near $w$.

With $n = 3$ and $V_c = \{\textit{leash}, \textit{walk}, \textit{owner}, \ldots\}$, "...he took the dog for a walk..." adds 1 to $c_\text{dog}[\textit{walk}]$. `took` is in the window but not in $V_c$, so it is ignored.

Similarity is the **cosine** of the angle between two vectors.

> [!formula] Cosine similarity #key-formula
> $$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert \, \lVert \mathbf{v} \rVert} = \frac{\sum_i u_i v_i}{\sqrt{\sum_i u_i^2}\,\sqrt{\sum_i v_i^2}}$$
>
> where $\mathbf{u} \cdot \mathbf{v}$ is the dot product and $\lVert \mathbf{u} \rVert$ the Euclidean length. It is 1 for vectors pointing the same way, 0 for orthogonal ones.

Cosine ignores length on purpose: a frequent word has large counts everywhere, and only the *proportions* of its contexts should matter. With two context words (`runs`, `legs`), `dog` = (1, 4), `cat` = (1, 5) and `car` = (4, 1) give $\cos(\text{dog}, \text{cat}) = 21/(\sqrt{17}\sqrt{26}) = 0.999$ and $\cos(\text{dog}, \text{car}) = 8/17 = 0.47$.

**Properties.** Context vectors are **high-dimensional** (one dimension per context word), **discrete** (counts) and **sparse** (mostly zeros).

**Advantages.** An **unsupervised** way to induce word similarity, with **interpretable dimensions**: $c_w[i]$ is the frequency of an actual word.

**Disadvantages.** The vectors **remain relatively sparse**, even with stemming. And there is **no explicit criterion to tell signal from noise**: some words fall in the window by chance, and nothing says which context words discriminate a word's meaning (`the` co-occurs with everything). One fix is to weight context words by **inverse document frequency**, $\text{idf}(w) = \log (N / n_w)$, with $N$ the number of contexts and $n_w$ the number containing $w$: a word found in nearly every context gets weight near 0.

## Learning dense vectors by prediction

A neural language model such as the probabilistic neural language model (PNLM: predict the next word from the previous ones with a feed-forward network) starts with a **projection layer** that maps each word $w$ to a learned dense vector $C(w)$. A *good* embedding has $C(w) \approx C(w')$ **if and only if** $w$ and $w'$ **mean the same thing and behave the same syntactically**. **In most models the projections are by-products** of optimising a task such as next-word prediction. **Representation learning** is when learning good embeddings *is* the main objective. Word2Vec is built that way: the prediction task is a pretext, and afterwards everything but the embedding matrices is discarded.

> [!definition] Word embeddings
> The same intuition as distributional semantics, but the vectors are **low-dimensional** (e.g. 100, against $|V_c|$), **dense** (no zeros), **continuous** ($c_w \in \mathbb{R}^m$) and **learned by performing a task** (prediction). Their dimensions are not interpretable.

**Word2Vec** (Mikolov et al., 2013) has two variants: **CBOW** predicts a word from its context, **Skip-gram** predicts the context from the word.

### CBOW: predict the word from its context

**Continuous Bag of Words (CBOW)**: given position $t$, the $n$ words to its left and the $m$ to its right, **predict the word at position $t$** ("the man X the road"). An n-gram language model predicts from the left only ($m = 0$); CBOW also looks right, so it is useless as a language model and exists only to learn representations. Compared with the PNLM it is a **simpler feed-forward network**: it **focuses on the embeddings themselves**, has **no non-linear hidden layer**, and **puts the embedding layer closer to the output**. **Typically $n = m$, with $n \in \{2, 5, 10\}$.**

> [!formula] CBOW #key-formula
> **No non-linearities. One hidden layer, then the output:**
> $$h = \frac{1}{2n} W \, w_C, \quad w_C = \sum_{\substack{i = t-n \\ i \neq t}}^{t+n} w_i, \qquad \hat{y} = \operatorname{softmax}(W' h)$$
>
> where:
> - $w_i$ is the **one-hot vector** of the word at position $i$
> - $W$ is the $|h| \times |V|$ input (projection) matrix, $|h|$ the embedding size
> - $W'$ is the $|V| \times |h|$ output matrix, and $\hat{y}$ the predicted distribution over the vocabulary
> - $W$ and $W'$ are **not (necessarily) shared**: $W' \neq W^\top$
>
> **Loss:** cross entropy. **Training:** SGD.

Multiplying $W$ by a one-hot vector selects a **column** of $W$, so $h$ is the **average of the context words' embeddings**. It is a "**bag of words**" because summing loses the word order. The softmax normalises over all $|V|$ words, so every example touches all of $W'$: that is the cost negative sampling removes.

**Where the embeddings live.** Column $i$ of $W$ is word $i$ as an input (context) word; row $i$ of $W'$ is word $i$ as an output word. **Typically $W$ is used**, or both combined: $W_s = W^\top + W'$.

### Skip-gram: predict the context from the word

**Skip-gram**: **given the word at position $t$, predict the words between $t-n$ and $t-1$ and between $t+1$ and $t+n$.** One input word, $2n$ predictions.

> [!formula] Skip-gram #key-formula
> **One hidden layer:** $h = W \, w_I$, with $w_I$ the one-hot input word at position $t$.
>
> **$2n$ output layers**, the context words **independent given the input word**:
> $$p(w_{t-n} \ldots w_{t-1} w_{t+1} \ldots w_{t+n} \mid w_I) \;\propto\; \prod_{\substack{i=t-n \\ i \neq t}}^{t+n} p(w_i \mid w_I), \qquad \hat{y}_i = \operatorname{softmax}(W' h)$$
>
> $W$, $W'$ as in CBOW, not necessarily shared. **Loss:** cross entropy. **Training:** SGD.

$h$ is just the input word's column of $W$, with no averaging. All $2n$ outputs share $W'$ and $h$, so they give the **same** distribution; each is scored against a different context word. Skip-gram is therefore training on $2n$ separate (input word, context word) pairs per position.

| | CBOW | Skip-gram |
|---|---|---|
| Input | $2n$ context words | the centre word |
| Predicts | the centre word | each context word |
| Hidden layer | average of context embeddings | the centre word's embedding |

## Negative sampling: classify pairs instead of predicting words

Both models **benefit from large amounts of data**, and that is where **computing the full output layer** hurts: a score for every vocabulary word, for every example.

> [!definition] Negative sampling
> **Distinguish between words that do and words that do not occur in the context of the input word**:
> - a **classification task** (is this (word, context) pair real?)
> - **1 positive example** from the ground truth
> - **$k$ negative examples** from a **random noise distribution**

The cost per example drops from $|V|$ dot products to $k + 1$ (see [[Negative Sampling]], an instance of [[Contrastive Learning]]).

**The objective.** With $D$ the observed (word, context) pairs and $D'$ pairs drawn from the noise distribution, choose parameters $\theta$ (the matrices $W$, $W'$) that make observed pairs look real and noise pairs look fake:

$$\arg\max_\theta \prod_{(w,c) \in D} p(D = 1 \mid c, w; \theta) \prod_{(w,c) \in D'} p(D = 0 \mid c, w; \theta), \qquad p(D = 1 \mid c, w; \theta) = \sigma(v_c \cdot v_w)$$

where $\sigma(x) = 1/(1+e^{-x})$, $v_w = Ww$ (the input word's column of $W$) and $v_c = W'^\top c$ (the context word's row of $W'$). Since $p(D = 0) = 1 - \sigma(v_c \cdot v_w)$ and $1 - \sigma(x) = \sigma(-x)$, taking logs gives:

> [!formula] Negative sampling objective #key-formula
> $$\arg\max_\theta \sum_{(w,c) \in D} \log \sigma(v_c \cdot v_w) + \sum_{(w,c) \in D'} \log \sigma(-v_c \cdot v_w)$$

The dot product is a compatibility score, pushed up for real pairs and down for random ones. Words that share many contexts are pulled towards the same output vectors, and so towards each other. Concretely, the gradient of the per-example loss with respect to $v_w$ is $(\sigma(v_c \cdot v_w) - 1)\, v_c + \sum_j \sigma(v_{n_j} \cdot v_w)\, v_{n_j}$: a descent step moves $v_w$ towards the true context and away from each negative $n_j$, most strongly from negatives the model currently mistakes for real contexts.

> [!definition] Word2Vec practical settings
> - **Skip-gram:** for each observed pair $(w, c)$, **add 5 to 20 negative samples**.
> - Draw negatives from the **unigram distribution $P(w)$**, **scaled to $P(w)^{0.75}$ to bias sampling towards rarer words**.
> - **Context size typically around 2 to 5.**
> - **The more data, the smaller the context and the negative sample set** can be.

The 0.75 power flattens the distribution, moving probability from frequent words to rare ones. Without it nearly every negative would be `the`, `of` or `and`.

## GloVe: fit vectors to global counts

**Two main strategies**, each with the other's weakness:

- **Count-based**, directly or as **global matrix factorisation** (latent semantic analysis: factorise the co-occurrence matrix with SVD). It **uses global information** (co-occurrence counts) directly, but **tends to perform worse**.
- **Prediction-based**, with a **local context window** (CBOW, Skip-gram). It **tends to outperform** count-based methods, but **treats each context window as an independent event**, rediscovering the association between two words every time they co-occur.

> [!definition] GloVe
> **GloVe** (Global Vectors; Pennington et al., 2014) **combines global co-occurrence statistics with local prediction-based updating**: count once over the whole corpus, then fit vectors to those counts with gradient updates.

$X_{ij}$ is the **number of times word $j$ occurs in the context of word $i$**; $X_i = \sum_k X_{ik}$, and $P_{ij} = X_{ij}/X_i$ is the probability of $j$ in the context of $i$.

> [!formula] GloVe objective #key-formula
> $$J = \sum_{(i,j)} f(X_{ij}) \cdot \left( w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$
>
> where $w_i$ is the vector for word $i$, $\tilde{w}_j$ the vector for context word $j$, $b_i$ and $\tilde{b}_j$ scalar biases, $\log X_{ij}$ the target, and $f$ a weighting function.

So GloVe is **weighted least-squares regression**: vectors whose dot product (plus biases) predicts the log of how often two words co-occur. Words with similar co-occurrence rows get similar vectors.

> [!formula] GloVe weighting function
> $$f(x) = \begin{cases} (x / x_\text{max})^{\alpha} & \text{if } x < x_\text{max} \\ 1 & \text{otherwise} \end{cases} \qquad \text{typically } x_\text{max} = 100,\ \alpha = 0.75$$

It **de-emphasises very rare (noisy) and very frequent (uninformative) co-occurrences**: it **rises smoothly from zero for rare pairs** and **caps at 1 for frequent pairs**, so they cannot take over. Since $f(0) = 0$, pairs that never co-occur (where $\log X_{ij} = -\infty$) drop out, and training runs only over the non-zero cells of $X$.

```pseudo
Algorithm: GloVe
──────────────────────────────────────────────────────────
Build co-occurrence matrix X by a single pass over the corpus:
for each center word i and each context word j within the window:
    X_ij += weight     // often 1/distance, nearer words more important

Initialize w_i, w̃_j, b_i, b̃_j randomly for all words

Repeat (for several epochs):
for each pair (i,j) (sampled/shuffled):
    weight  ← f(X_ij)
    error   ← w_iᵀw̃_j + b_i + b̃_j − log X_ij
    loss_ij ← weight · error²
    update w_i, w̃_j, b_i, b̃_j by AdaGrad on this loss

Final embedding for word i ← w_i + w̃_i   (sum the two vector sets)
```

**One pass over the corpus** builds $X$; training then only sees $X$, which is what "global" means. **AdaGrad** is SGD with a per-parameter learning rate that shrinks for parameters that have had large gradients. The **final vector $w_i + \tilde{w}_i$** is the same trick as $W^\top + W'$ in Word2Vec.

| | Word2Vec | GloVe |
|---|---|---|
| Trains on | individual local context windows | the global co-occurrence matrix |
| Objective | predict context word(s) | weighted least squares on log counts |
| Frequency imbalance | subsampling frequent words (and negative sampling) | explicit weighting $f(X_{ij})$ |
| Interpretability | indirect: learns to predict context | directly fits log co-occurrence counts |

**Subsampling** randomly discards occurrences of very frequent words before training, more often the more frequent the word.

## Judging whether embeddings are any good

> [!definition] Evaluation axes
> - **Intrinsic:** how good are the embeddings **by themselves**? **Extrinsic:** how useful are they **for downstream tasks** (summarisation, MT, IR)?
> - **Qualitative:** **examine selected examples**. **Quantitative:** an **overall, task-dependent score**.

The axes are independent: a 2D plot is intrinsic and qualitative, a similarity benchmark intrinsic and quantitative.

**Word similarity.** **Rank word pairs** such as (`car`, `bicycle`) **by similarity** (cosine) and compare with human judgements by **Spearman correlation**, a correlation between the two rankings, so only the order matters. **WS-353 annotates relatedness; SimLex-999 annotates similarity.** Such benchmarks **mix all kinds of similarity**. In WS-353 `bread`/`butter` scores fairly high although bread is not a kind of butter: relatedness. The distinction matters, because `coffee` and `cup` are related but not similar, so the two benchmarks can rank the same embeddings differently.

**Analogies.** *Paris is to France as Berlin is to X*, **scored by accuracy**. The "**arithmetic magic**": $X = v_\text{king} - v_\text{man} + v_\text{woman}$, answer the word nearest to $X$ by cosine (`queen`), excluding the input words. There are **syntactic analogies** too: *acquired* is to *acquire* as *tried* is to X.

**Visualisation.** Project to 2D with **PCA** (linear, onto the two directions of largest variance) or **t-SNE** (non-linear, keeps each point's neighbours close, gives up global distances). Plots show senses separating (around `power`: electrical, mechanical and abstract clusters) and relations as directions: lines from `brother` to `sister`, `uncle` to `aunt`, `king` to `queen` all point roughly the same way, a consistent **gender direction**. That geometry is behind analogy arithmetic and behind crosslingual alignment.

> [!warning] Do not take 2D plots at face value
> They **can use non-linear projections**, so plot distances are not real distances, and **t-SNE hyperparameters have a substantial impact** on the picture (Wattenberg 2016). Plots **complement, but do not substitute**, quantitative and extrinsic evaluation.

## fastText: words built from character n-grams

**Word2Vec and GloVe give one opaque vector per word type**: `run`, `runs`, `running`, `runner` get four potentially unrelated vectors. **Morphologically rich languages suffer most**: Finnish, Turkish and Russian have huge families of word forms, each a separate, rarer word with a worse vector. And there is **no fallback for unseen (out-of-vocabulary, OOV) words**. This is the problem of [[MNLP-L04 - Subword Segmentation]] from the embedding side.

Two fixes: **sub-segment words** (e.g. with BPE) **and learn an embedding per segment**, or use **a sliding character window: fastText** (Bojanowski et al., 2017), which represents a word by all its character n-grams.

```pseudo
Algorithm: fastText
──────────────────────────────────────────────────────────
Input:  corpus, window size m, n-gram range [n_min, n_max] (default 3 to 6),
        embedding dimension d, negatives K

for each word w in vocabulary:
    G_w ← { all char n-grams of "<w>" for n in [n_min, n_max] } ∪ { "<w>" }
    initialize n-gram vectors z_g (dimension d) and context vectors u_c randomly

for each (center word w, context word c) pair in corpus, with window m:
    sample K negatives n_1..n_K from noise distribution P_n
    v_w ← Σ_{g ∈ G_w} z_g                              // compose center vector
    loss ← −log σ(v_wᵀ u_c) − Σ_k log σ(−v_wᵀ u_{n_k})

    for each g ∈ G_w:
        z_g ← z_g − η · ∂loss/∂v_w          // same gradient to each n-gram
    update u_c and all u_{n_k}

return { z_g }
```

Compared with Skip-gram with negative sampling:

- **The centre word's vector is the sum of its character n-gram vectors**, $v_w = \sum_{g \in G_w} z_g$, n-grams of **3 to 6 characters** by default, **plus the whole word** `<w>`.
- `<` and `>` are **boundary markers**, so prefixes and suffixes are distinct n-grams (`<ru` only starts a word, `ns>` only ends one).
- **The loss is the negative-sampling loss.** Only the centre word is decomposed; **context vectors $u_c$ stay per-word**.
- **Every n-gram of the word gets the same gradient**, since $v_w$ is a sum.
- **It returns the n-gram table.** Any word's vector, seen in training or not, is the sum of its n-grams.

> [!example] The n-grams of `runs`
> `<runs>` gives `<ru`, `run`, `uns`, `ns>`, `<run`, `runs`, `uns>`, `<runs`, `runs>`, `<runs>`. `<running>` shares `<ru`, `<run` and `run`, so the two vectors are pulled together. The unseen `runnings` still has known n-grams such as `<run` and `ings>`, so it gets a sensible vector where Word2Vec has none.

**The evidence** (Bojanowski et al.), comparing Word2Vec Skip-gram and CBOW with fastText (*sisg*, subword information skip-gram) and with *sisg-*, which gives unseen words a null vector:

- **Word similarity:** fastText is best or tied best on almost every dataset, with the **largest gains in morphologically rich languages** (German, Russian) and a gain on **English rare words**. Its one loss is English WS-353, frequent words whose whole-word vectors are already good. Building unseen words from n-grams (sisg over sisg-) adds further.
- **Analogies:** the gains are on **syntactic analogies** (largest for Czech), which are about morphology. **Semantic analogies do not improve and sometimes get worse**: a country and its capital share nothing in spelling, and tying words with shared n-grams together can add noise.

## Crosslingual embeddings: one space for two languages

Embeddings give $E(\text{house}) \approx E(\text{building})$ in English and $E(\text{huis}) \approx E(\text{gebouw})$ in Dutch. Can we get $E(\text{house}) \approx E(\text{gebouw})$? Such a space gives **bilingual lexicon induction** (BLI: building a dictionary by nearest-neighbour search) and transfer between languages ([[MNLP-L07 - Crosslingual NLP]]).

**Three basic set-ups:**

1. **Train each language separately, then align the spaces.**
2. **Train all languages together** (benefiting from shared words such as names and numbers), **then align the regions**.
3. **Train on data aligned** at the word or sentence level.

**Either way some alignment is needed**: a **mapping** under which **translation-equivalent words end up close together**. The hope is to use **large monolingual corpora** and only a **small bilingual signal**. The rest of the lecture is about set-up 1.

### The shared space hypothesis

Plot English number words and animals in 2D next to their Spanish translations, and the arrangement is the same (`one`/`uno` apart from the other numbers, `horse`/`caballo` near `cow`/`vaca`), only stretched and shifted.

> [!definition] Shared space hypothesis
> - **Different languages, trained separately, still encode similar relational structure**: `king` and `queen` relate in English as `rey` and `reina` do in Spanish.
> - There is an **approximate isomorphism** between the spaces: the **overall shape of the point cloud is similar**, but **coordinates and orientations can differ**.
> - **If true (it is a hypothesis)**, a **single geometric transformation can align the spaces**: a general linear map applied to one space brings translations close together.

In English, `king` and `queen` are separated by roughly the same offset as `man` and `woman`; in Spanish the same holds for `rey`, `reina`, `hombre`, `mujer`, but the cloud may be **rotated and scaled differently**. A structure-preserving map $m$ satisfies $m(a \circ b) = m(a) \circ m(b)$, so

$$m(v_\text{king} - v_\text{man} + v_\text{woman}) \approx m(v_\text{king}) - m(v_\text{man}) + m(v_\text{woman})$$

and the English gender offset becomes the Spanish one. A linear map $m(v) = Wv$ satisfies this exactly; the "$\approx$" is there because the spaces are only approximately isomorphic.

> [!definition] Types of alignment methods
> - **Supervised:** a **seed bilingual dictionary** as anchors, **typically thousands** of translations.
> - **Semi-supervised:** a **very small seed dictionary (hundreds)**, then **bootstrapping new translations based on statistical confidence**.
> - **Unsupervised:** **no bilingual signal**; **adversarial training**.

**Where seed dictionaries come from:**

1. **Human-compiled, machine-readable dictionaries**: ideal in principle, often not in practice. They list lemmas (`talk`) while embedding vocabularies are full of inflected forms (`hablamos`, "we talk"), mix senses, cover low-resource languages poorly, and miss multi-word expressions.
2. **Learned from data**: this **requires human translations at the sentence level**, then **word alignment of each sentence pair**. Alignments are not one-to-one (Dutch *brengt een bezoek aan*, "brings a visit to", aligns to English *visits*), so the resulting dictionary is noisy.
3. **Words occurring in several languages** (identically spelled names, numbers, borrowings). Issues: false friends (Dutch `bad` means bath), a shared script is needed, and the words skew towards names and numbers. It still works surprisingly well, as the critique at the end shows.

> [!definition] Seed dictionary
> A **list of (source word, target word) translation pairs**: often the **top few thousand most frequent words**, translated with an existing dictionary; **automatic translation** can be used **with caution**; **limited to words**, since there are **no embeddings for multi-word expressions**. **Each pair is one anchor** across the spaces (the source vector of `cat`, the target vector of `gato`). The **goal is a mapping that generalises beyond the anchors** to words never in the dictionary.

## Learning a translation matrix from a seed dictionary

> [!formula] General alignment problem #key-formula
> $$W^* = \arg\min_W \sum_i \lVert W x_i - y_i \rVert^2$$
>
> where $(x_i, y_i)$ are **paired vectors**, a source embedding $x_i \in \mathbb{R}^{d_1}$ and the target embedding $y_i \in \mathbb{R}^{d_2}$ of its translation, and $W$ ($d_2 \times d_1$) is applied to every source vector.

> [!warning] Two uses of the name "Procrustes"
> The lecture calls this general problem "the **Procrustes** algorithm". In the literature and in Conneau et al.'s results tables, "Procrustes" means the **orthogonally constrained** version below, solved by SVD. The unconstrained least-squares map is Mikolov et al.'s translation matrix (2013); the orthogonal one is Xing et al.'s (2015).

> [!definition] Linear transformation hypothesis
> - **Strong assumption:** a **single linear transformation $W$ is sufficient** to map one language's space onto another's.
> - **Why linear and not non-linear:** king/queen and man/woman analogies are **already linear regularities within one space**, assumed to **hold across spaces** too.
> - It **reduces alignment to a classic regression problem**.

**Solving it.** Stack the dictionary pairs as columns of $X$ ($d_1 \times n$) and $Y$ ($d_2 \times n$). Either run gradient descent on the squared loss, or use the closed form, found by setting the gradient $2(WX - Y)X^\top$ to zero:

> [!formula] Closed-form solution
> $$W^* = Y X^\top (X X^\top)^{-1}$$
> It needs $XX^\top$ invertible, so at least $d_1$ linearly independent pairs.

**Mikolov et al. (2013) used SGD**, which **scales better with large dimensions and dictionaries**; the **closed form is exact, SGD approximate but more memory-friendly**. To translate $x$, compute $Wx$ and return the nearest target word by cosine.

**How well it worked** (Mikolov et al., English with Spanish and Czech):

- The **translation matrix beats both baselines by a wide margin**: edit distance (pick the most similarly spelled word) and word co-occurrence (count-based vectors).
- Adding edit distance helps much more for Spanish than Czech, since English and Spanish share more cognates. Czech is harder for every method.
- **Accuracy rises with monolingual data size** (roughly linearly in its log, flattening at a few billion words) and **drops for rarer words**, whose embeddings are worse.
- The errors are related words: `imperio` (empire) gives `dictatorship`, `imperialism`, `tyranny`; `millas` ranks `kilometers` above `miles`; `hablamos` gives `talking`, `talked`, `talk`. Nearness captures **relatedness**, while a dictionary demands **equivalence**.

## Keeping the map rigid: orthogonal mappings

> [!warning] Limitations of unconstrained mappings
> - **An arbitrary $W$ can stretch, shear and rotate**; nothing preserves the source space's **internal geometry** (distances, angles).
> - So it can **overfit the seed dictionary**: pairs in or near the dictionary fit well, and the mapping **does not generalise to the words outside it** (the majority).
> - **Training and inference objectives differ**: training minimises **Euclidean distance**, retrieval uses **cosine**, and they are **not equivalent unless vectors are unit-normalised**.

> [!formula] Orthogonal mapping (Xing et al., 2015) #key-formula
> $$W^* = \arg\min_W \sum_i \lVert W x_i - y_i \rVert^2 \quad \text{subject to} \quad W^\top W = I$$
> with **all embeddings normalised to unit length**.
>
> An orthogonal $W$ **only allows rotations and reflections, no scaling or shearing**, so it **preserves angles and distances**. $W^*$ **can be solved exactly using SVD**.

It fixes both problems:

1. **Geometry is preserved**: $\lVert Wx \rVert^2 = x^\top W^\top W x = \lVert x \rVert^2$, so the source space moves rigidly and cannot be warped to fit the seed pairs. A rigid rotation is what the shared space hypothesis says should suffice.
2. **Training and retrieval agree**: for unit vectors and orthogonal $W$, $\lVert W x - y \rVert^2 = 1 + 1 - 2\,(Wx) \cdot y = 2 - 2 \cos(Wx, y)$, so minimising distance **is** maximising cosine.

After normalisation every vector lies on the unit sphere, and an orthogonal map is a rotation (or reflection) of that sphere.

> [!formula] Orthogonal Procrustes solution
> $$U \Sigma V^\top = \operatorname{SVD}(Y X^\top), \qquad W^* = U V^\top$$
> One SVD of a $d \times d$ matrix: exact, no iterations, no learning rate.

**The evidence** (English to Spanish): the **unconstrained map gets worse as the embedding dimension grows**, because $W$ has $d^2$ free parameters for the same dictionary. The **orthogonal map is better at every dimension and improves slightly** with it.

## Hubs, and how CSLS discounts them

Retrieval is **nearest-neighbour search at inference time** on the mapped embeddings, and high-dimensional spaces spoil it.

> [!definition] Hubness
> In high-dimensional spaces **some points become hubs**: a **small number of vectors are the nearest neighbour of many other points, regardless of actual similarity**. It is a **known phenomenon of high-dimensional geometry**, not specific to embeddings, and it **worsens as dimensionality increases**.

A target hub gets nominated as the "translation" of many unrelated source words: nn(`cat`) = nn(`car`) = nn(`house`) = `thing`. Nearest neighbour is not symmetric: `thing` can be `cat`'s nearest neighbour while `cat` is far down `thing`'s list. A hub sits near the middle of the cloud, moderately close to everything. Conneau et al. (2018) modify cosine to correct for this.

> [!formula] Cross-domain Similarity Local Scaling (CSLS) #key-formula
> $$\text{CSLS}(x, y) = 2 \cos(x, y) - r_T(x) - r_S(y)$$
>
> where $x = Wx_s$ is a mapped source vector and $y$ a target vector,
> - $r_T(x)$ is the **average cosine between $x$ and its $K$ nearest neighbours in the target space**
> - $r_S(y)$ is the **average cosine between $y$ and its $K$ nearest neighbours in the (mapped) source space**

- **Subtracting $r_S(y)$ penalises a candidate that is close to lots of things**: a hub has high $r_S$.
- $r_T(x)$ is the same for every candidate when translating a fixed $x$, so it cannot change the winner; it matters when scores are compared across source words, as when building a dictionary.
- **CSLS significantly increases translation accuracy without parameter tuning**; only $K$ is set ($K = 10$ in Conneau et al.).

> [!example] CSLS defusing a hub (toy numbers)
> Translating `cat`, with $r_T(x) = 0.5$. The hub `thing`: $\cos = 0.65$, $r_S = 0.70$, CSLS $= 1.30 - 0.5 - 0.70 = 0.10$. `gato`: $\cos = 0.60$, $r_S = 0.40$, CSLS $= 1.20 - 0.5 - 0.40 = 0.30$. Nearest neighbour picks `thing`; CSLS picks `gato`.

**The evidence** (supervised orthogonal mapping on fastText embeddings): **CSLS beats plain nearest neighbour on every language pair**, about level with inverted softmax (ISF), another hubness correction. All methods get worse with distance from English: Spanish and French best, then German, Russian, Chinese, and Esperanto lowest.

## No dictionary at all: adversarial training (MUSE)

Supervised methods need **thousands of pairs**; for **many low-resource pairs even a small dictionary may not exist**, and dictionaries **may not cover the inflected forms** most common in real text. **Can two independently trained spaces be aligned from their geometry alone**, as two point clouds hypothesised to be approximately isomorphic? The **chicken-and-egg problem**: a **mapping is needed to find translation pairs**, and **pairs to learn the mapping**. Conneau et al. (2018) break it with **adversarial training, which bootstraps a rough alignment from distributional structure alone**. Their toolkit is **MUSE**.

### The adversarial game

Borrowed from **Generative Adversarial Networks (GANs)**, where a generator learns to produce fakes and a discriminator to spot them:

- **Generator:** the mapping $W$, mapping source embeddings into the target space, **aiming to make mapped source vectors indistinguishable from real target vectors**.
- **Discriminator:** a classifier, **trained simultaneously**, that **tells mapped source vectors from real target vectors**.
- **Real data:** target embeddings $y_1, \ldots, y_n$. **Fake data:** mapped source embeddings $Wx_1, \ldots, Wx_m$.
- They are **trained in opposition**: the discriminator gets better at telling them apart, $W$ at fooling it.

If the discriminator cannot tell the clouds apart, they have the same shape and position, and under the shared space hypothesis the rotation that achieves this puts translations on top of each other.

> [!important] The generator is deliberately not a deep network
> Unlike a standard GAN, the generator is a **single $d \times d$ matrix, a linear map with no non-linearity**. A **deep generator could learn an arbitrarily complex warping** that makes the mapped cloud statistically indistinguishable from the target cloud **while scrambling which source word lands where**. Only a constrained, near-rigid map makes the word-to-word correspondence come with the distribution match.

**Discriminator:** a **multilayer perceptron, 2 hidden layers of 2048 units**, **ReLU**, **dropout on the input layer**, and a **two-class output with label smoothing $s = 0.2$**: targets $\{s, 1-s\}$ instead of $\{0, 1\}$, which **keeps it from becoming pathologically overconfident** (that would starve the generator of gradient).

**Sampling:** both source and target vectors come from **the 50k most frequent words** of each language, because **rare words have poorly estimated embeddings** and frequent ones give a **cleaner discriminatory signal**.

```pseudo
Algorithm: Adversarial training (MUSE)
──────────────────────────────────────────────────────────
Input: source embeddings X, target embeddings Y (monolingual, unaligned)
       orthogonalization coefficient β = 0.01

Initialize W ← I  (identity/random orthogonal matrix)
Initialize discriminator parameters θ_D randomly

repeat:
    // Discriminator step (W frozen):
    sample batch of source vectors {x_i}
    sample batch of target vectors {y_j}
    compute mapped vectors {W x_i}
    ℒ_D ← −(1/n) Σ log P_θD(source=1 | W x_i)
          −(1/m) Σ log P_θD(source=0 | y_j)
    θ_D ← θ_D − η · ∂ℒ_D/∂θ_D

    // Generator step (θ_D frozen):
    sample new batches {x_i}, {y_j}
    ℒ_W ← −(1/n) Σ log P_θD(source=0 | W x_i)
          −(1/m) Σ log P_θD(source=1 | y_j)
    // backpropagate ℒ_W through the frozen discriminator into W
    ∇_W ← (1/n) Σ δ_i x_iᵀ        // where δ_i = ∂ℒ_W/∂(W x_i)
    W ← W − η · ∇_W

    // Re-orthogonalization:
    W ← (1 + β) W − β (W Wᵀ) W
```

- $P_{\theta_D}(\text{source} = 1 \mid v)$ is the discriminator's probability that $v$ is a mapped source vector.
- **The steps alternate, each with the other side frozen.** The discriminator trains on the true labels, the generator on **flipped labels**, so $W$ is rewarded when mapped vectors are judged "target".
- **Re-orthogonalisation** pulls $W$ back towards an orthogonal matrix after each step (an orthogonal $W$ has $WW^\top = I$ and is left unchanged). Without it, $W$ would drift into a general linear map with the warping problems of the unconstrained mapping.

**Choosing a checkpoint without a dictionary.** The **adversarial loss is not a reliable indicator** of alignment quality: the **discriminator loss can look fine while the mapping is poor**, and there is **no validation dictionary**.

> [!definition] Unsupervised validation criterion
> 1. Take the **10k most frequent source words**.
> 2. With the current $W$, find each one's **CSLS nearest neighbour** in the target space.
> 3. Compute the **average cosine similarity** over these pairs, and keep the checkpoint where it is highest.

The empirical finding behind it: a poor mapping puts mapped words **far from any real target words**, so a **high average cosine** means the mapping lands on actual target words. Whether they are the *correct* ones it cannot tell; it is a proxy.

### Refinement: from a rough map to a precise one

The adversarial $W$ is **rough**: the **global orientation is about right**, but **not precise enough** for good retrieval. The **full MUSE pipeline** continues:

1. **Adversarial training** for a rough $W$.
2. **Build a synthetic dictionary**: use $W$ with **CSLS** to find **mutual nearest-neighbour pairs among frequent words**, keeping **only high-confidence mutual matches**.
3. **Re-solve for $W$ exactly via SVD** on that dictionary (orthogonal Procrustes).
4. **Repeat steps 2 and 3** (self-learning).

**Mutual**: $(s, t)$ enters only if $t$ is $s$'s best match **and** $s$ is $t$'s, which filters out hubs. The refinement is called **semi-supervised**: steps 2 to 4 are the supervised orthogonal method run on a dictionary the system built itself. Start the same loop from a small human seed dictionary and you have the semi-supervised alignment method.

**How well it works:** **each component helps** (adversarial alone is weakest; CSLS and refinement each add; both together are best). **Fully unsupervised MUSE matches or slightly beats supervised Procrustes with CSLS** on close, well-resourced pairs (English with Spanish, French, German), and **falls well short on distant pairs** (English with Russian, Chinese). The critique starts there.

## When unsupervised mapping fails

**Søgaard, Ruder and Vulić (2018)** questioned Conneau et al.'s assumptions and results from several angles.

**The choice of languages.** Conneau et al.'s languages (Spanish, French, German, Russian, Chinese) are mostly **dependent-marking**, have few or no cases, and none is **agglutinative**. Søgaard et al. added **Estonian and Finnish** (mixed marking, agglutinative, 10+ cases), **Greek** (double marking, fusional), and **Hungarian, Polish, Turkish** (dependent marking). The morphological types are from [[MNLP-L03 - Morphology and Word Formation]]: fusional packs several meanings into one affix, agglutinative chains many affixes, one per meaning.

> [!definition] Head marking and dependent marking
> When two words are in a grammatical relationship, **which one carries the morphological marker**?
> - **Dependent-marking:** the **dependent**. **Head-marking:** the **head**. **Double-marking:** **both**. **Zero-marking:** **neither**.
>
> In a possessive phrase the head is the **possessed** noun and the dependent the **possessor**; in a clause the head is the **verb**, the dependents **subject and object**.

German *das Haus des Mannes* (the house of the man) marks the genitive on the dependent *des Mannes*; *der Mann sieht den Hund* (the man sees the dog) puts nominative and accusative on the dependents. The verb also agrees with its subject, so there is some head-marking, but dependent marking predominates. English uses word order instead.

**A free baseline: identical-word supervision.** Use the **words spelled identically in both languages** as the seed dictionary and run supervised alignment. It needs no human dictionary either.

**The results:**

- **Adversarial alignment fails completely** (essentially zero precision) for **English-Estonian, English-Finnish and English-Greek**, the mixed- and double-marking languages.
- **Identical-word supervision beats adversarial training on every pair involving English**, from a small margin (English-Spanish) to working against not working (English-Estonian).
- **Estonian-Finnish**, two structurally similar languages, **aligns adversarially** though neither aligns with English, the one pair where adversarial beats identical.

Marking type alone does not decide it: French is classed as mixed-marking and aligns well with English. The failing languages combine unusual marking with case-rich morphology, and Hungarian (dependent-marking, agglutinative) lands in between.

**Why it hurts.** In mixed- or double-marking languages **grammatical information is distributed across more word forms**: a **lemma is realised in many more distinct surface forms, each with its own embedding and each of lower frequency**. That makes the **mapping between a language that marks relations by word order and one that marks them by morphology more complex (non-isomorphic)**. English `house` faces Finnish `talo`, `talon`, `taloa`, `talossa`, `talosta`, ... (illustration from outside the lecture), and no rotation maps one point onto a dozen.

### Measuring how isomorphic two spaces are

**Isomorphism is a strict true/false criterion**; to measure **degrees** of it, Søgaard et al. compare nearest-neighbour graphs through their eigenvalues.

> [!definition] Eigenvector similarity
> 1. Compute each word's **nearest neighbours** and represent them as an **adjacency matrix $A$**.
> 2. Form the **Laplacian $L = D - A$**, with **$D$ the diagonal matrix of neighbour counts**.
> 3. Take the **eigenvalues of $L$** and keep the **largest $n\%$**.
> 4. For two languages, **sum the squared differences of their Laplacian eigenvalues**:
> $$\Delta = \sum_{i} \left( \lambda^{(1)}_i - \lambda^{(2)}_i \right)^2$$
> with $\lambda^{(1)}_i, \lambda^{(2)}_i$ the $i$-th largest kept eigenvalues of each graph.

(The lecture words step 2 as subtracting $D$ from $A$. The standard Laplacian is $D - A$, with eigenvalues $\geq 0$; the other sign flips them and "keep the largest" would pick the wrong end.)

- **Laplacian eigenvalues indicate how connected the nodes are**: bigger, denser.
- **The bigger $\Delta$, the more structurally different** the graphs, the further from isomorphism.
- It **does not require knowing the alignment**: eigenvalues do not change when nodes are relabelled. If dense parts are similarly dense and sparse parts similarly sparse, $\Delta$ is small.

**A high $\Delta$ means** that words **cluster in language A unlike in language B**, that **no rotation can fix it** (a rotation keeps the neighbour graph, so the eigenvalues), and that **the isomorphism assumption does not hold**. For Finnish the plausible reason is morphology: **one lemma spreads across dozens of inflected forms** that **cluster tightly** through overlapping meaning, while **English gives a flatter, more evenly connected graph**. In the results **$\Delta$ tracks adversarial success**: English-Spanish has the lowest $\Delta$, and the three failing pairs the highest.

> [!warning] The inference only goes one way
> **A low $\Delta$ does not mean the spaces are near-isomorphic.** Near-isomorphic graphs have a low $\Delta$, not the other way round (in graph theory, different graphs can share the same eigenvalues). $\Delta$ can rule isomorphism out, never in.

### Not data size, but domain and algorithm

**Not just data size.** Finnish Wikipedia is far smaller than Spanish, but retraining on the **Finnish WaC corpus, 1.7 billion words, left English-Finnish at zero**, and Estonian-Finnish works. **Structural similarity matters: pairing structurally dissimilar languages is the problem.** Results **differ by part of speech**: for English-Hungarian, **nouns and verbs**, the most inflected, are **the worst**, about half as accurate as adjectives and adverbs; English-Finnish is zero for every part of speech; for English-Spanish, verbs are the weakest.

**Domain.** Embeddings trained on equal-sized corpora from **EuroParl (political debates), Wikipedia (general knowledge) or EMEA (medical texts)**, English aligned with Spanish:

- **Same domain on both sides:** both methods work.
- **Different domains:** identical-word supervision **degrades but survives**; **adversarial induction collapses to essentially zero in every cross-domain combination**, even for English and Spanish.

That matters because a low-resource language, the reason to go unsupervised, rarely offers a corpus from the same domain as the English one.

**Embedding algorithm.** English fixed at fastText Skip-gram (window 2, n-grams 3 to 6), Spanish varied:

- **Variations within one model barely matter**: another window size or n-gram range for Spanish Skip-gram costs under two points.
- **Two different base models do not work at all**: Spanish CBOW against English Skip-gram gives essentially zero, even with otherwise identical settings.

So the "approximate isomorphism" is partly an artefact of using the same algorithm on both sides.

> [!tip] What the critique adds up to
> Unsupervised alignment works when the languages are typologically similar, the corpora share a domain, and both sides use the same embedding algorithm: the conditions it was evaluated under, and the opposite of the low-resource setting that motivates it. A free dictionary of identically spelled words beats it on every English pair tested.

## The alignment methods side by side

| Method | Bilingual signal | Constraint on $W$ | Solved by | Retrieval |
|---|---|---|---|---|
| Translation matrix (Mikolov 2013) | seed dictionary | none | SGD or closed form | cosine NN |
| Orthogonal (Xing 2015) | seed dictionary | $W^\top W = I$, unit vectors | SVD | cosine NN |
| Procrustes + CSLS (Conneau 2018) | seed dictionary | orthogonal | SVD | CSLS |
| MUSE (Conneau 2018) | none | kept near-orthogonal | adversarial, then SVD refinement | CSLS |
| Identical words (Søgaard 2018) | identical spellings | orthogonal | as supervised | CSLS |

## What to be able to answer

- Count-based vectors: the recipe, cosine, their properties, signal against noise, idf.
- CBOW against Skip-gram: what each predicts, the averaged hidden layer, no non-linearity, $W$ and $W'$ not shared.
- Negative sampling: 1 positive and 5 to 20 negatives from $P(w)^{0.75}$, the objective, more data allowing a smaller context and fewer negatives.
- GloVe: weighted least squares on log counts, one pass to build $X$, the shape of $f$, $w_i + \tilde{w}_i$, the contrasts with Word2Vec.
- Evaluation: intrinsic and extrinsic, relatedness (WS-353) against similarity (SimLex-999), analogies, the limits of plots.
- fastText: character n-grams plus the word, only the centre word decomposed, unseen words, syntactic gains only.
- Crosslingual: the shared space hypothesis, why linear, supervised against semi-supervised against unsupervised.
- Unconstrained against orthogonal mapping: overfitting, Euclidean against cosine, unit vectors, SVD.
- Hubness, and which CSLS term penalises hubs.
- MUSE: why the generator is linear, discriminator settings, re-orthogonalisation, checkpoint choice, mutual-neighbour refinement.
- Søgaard et al.: where unsupervised alignment fails and why, the identical-word baseline, what $\Delta$ can and cannot show.

Model answers are in the exam questions below.

## Exam questions

> [!exam]- Does unsupervised bilingual lexicon induction work? Answer with the mechanism, how it compares with supervised alignment, and the conditions under which it fails.
> **Key points:** shared space hypothesis and the MUSE pipeline (adversarial, CSLS checkpoint selection, Procrustes refinement); matches supervised on close pairs, falls short on distant ones; collapses for case-rich mixed- and double-marking languages, across domains and across algorithms, and data size does not explain it; the free identical-word baseline beats it.
>
> 1. **Mechanism.** The shared space hypothesis says separately trained spaces are approximately isomorphic, so one linear (orthogonal) map should align them. MUSE (Conneau et al., 2018) bootstraps a rough $W$ adversarially (a linear generator against an MLP discriminator, with re-orthogonalisation), picks the checkpoint by the unsupervised CSLS criterion, then refines with Procrustes ($W = UV^\top$) on a synthetic dictionary of mutual CSLS nearest neighbours, repeated.
> 2. **Against supervised alignment.** Fully unsupervised MUSE matches or slightly beats supervised Procrustes with CSLS on close, well-resourced pairs (English with Spanish, French, German), and falls well short on distant ones (English with Russian, Chinese).
> 3. **Where it collapses (Søgaard et al., 2018).** Mixed- or double-marking, case-rich languages: English-Estonian, English-Finnish and English-Greek give essentially zero. English-Finnish stays at zero even with 1.7 billion words of Finnish, so data size is not the explanation; Estonian-Finnish, structurally similar, does align. Mismatched domains give essentially zero even for English-Spanish, and so does Spanish CBOW against English Skip-gram.
> 4. **A free baseline beats it.** A seed dictionary of identically spelled words beats adversarial training on every pair involving English, from a small margin (English-Spanish) to working against not working (English-Estonian).
> 5. **Verdict.** It works when the languages are typologically similar, the corpora share a domain and both sides use the same algorithm: the conditions it was evaluated under, and the opposite of the low-resource setting that motivates it.

> [!exam]- Derive the negative sampling objective from a binary classification set-up, then give its gradient with respect to the input vector $v_w$ and interpret it.
> **Key points:** real-or-noise classification with 1 positive and $k$ negatives; $p(D=1) = \sigma(v_c \cdot v_w)$ and $1-\sigma(x) = \sigma(-x)$; log objective over $D$ and $D'$; gradient pulls $v_w$ to $v_c$ and pushes it from negatives; cost $k+1$ instead of $|V|$.
>
> 1. **Why.** A full softmax needs a score for all $|V|$ words per example. Instead ask "is this (word, context) pair real or noise?": 1 positive pair from the data, $k$ negatives from a noise distribution.
> 2. **Likelihood.** $\arg\max_\theta \prod_{(w,c) \in D} p(D=1 \mid c,w;\theta) \prod_{(w,c) \in D'} p(D=0 \mid c,w;\theta)$, with $p(D=1 \mid c,w) = \sigma(v_c \cdot v_w)$, $v_w = Ww$, $v_c = W'^\top c$.
> 3. **Sigmoid identity.** $1 - \sigma(x) = \sigma(-x)$, so after logs: $\sum_{D} \log \sigma(v_c \cdot v_w) + \sum_{D'} \log \sigma(-v_c \cdot v_w)$.
> 4. **Gradient** of the per-example loss $\mathcal{L} = -\log \sigma(v_c \cdot v_w) - \sum_j \log \sigma(-v_{n_j} \cdot v_w)$:
> $$\frac{\partial \mathcal{L}}{\partial v_w} = (\sigma(v_c \cdot v_w) - 1)\, v_c + \sum_{j} \sigma(v_{n_j} \cdot v_w)\, v_{n_j}$$
> 5. **Interpretation.** A descent step moves $v_w$ towards $v_c$ and away from each $v_{n_j}$. Each coefficient shrinks to zero once its pair is classified correctly, so the biggest corrections come from negatives the model mistakes for real contexts.

> [!exam]- Compare Word2Vec (CBOW and Skip-gram) with GloVe: what each optimises, what information each uses, how each handles frequency imbalance, and what the final embedding is.
> **Key points:** CBOW and Skip-gram predict within local windows; GloVe is weighted least squares on global log counts; subsampling against $f(X_{ij})$; final embedding $W$ or $W^\top + W'$ against $w_i + \tilde{w}_i$.
>
> - **CBOW:** predict the centre word from the average of $2n$ context embeddings, $h = \frac{1}{2n} W w_C$, $\hat{y} = \operatorname{softmax}(W'h)$, no non-linearity, cross entropy, SGD.
> - **Skip-gram:** predict each of the $2n$ context words from the centre word, $h = W w_I$, context words independent given $w_I$; in practice trained with negative sampling.
> - **GloVe (Pennington et al., 2014):** weighted least squares on log co-occurrence counts, $J = \sum f(X_{ij})(w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$.
> - **Information used:** Word2Vec sees individual local windows, each an independent event, so it keeps rediscovering the same association. GloVe counts once over the whole corpus and fits those global counts.
> - **Frequency imbalance:** Word2Vec subsamples frequent words (and draws negatives from $P(w)^{0.75}$); GloVe uses the weighting $f(X_{ij})$ ($x_\text{max} = 100$, $\alpha = 0.75$).
> - **Final embedding:** Word2Vec typically $W$, or $W^\top + W'$; GloVe $w_i + \tilde{w}_i$.

> [!exam]- Why does constraining a crosslingual mapping to be orthogonal help? Give the objective, the two problems it fixes with a derivation for each, its solution, and the evidence.
> **Key points:** $W^\top W = I$ with unit vectors; a rigid map cannot overfit the seed pairs; $\lVert Wx - y \rVert^2 = 2 - 2\cos$ aligns training with retrieval; $W = UV^\top$ by SVD; unconstrained degrades with dimension, orthogonal better everywhere.
>
> 1. **Objective (Xing et al., 2015).** $\arg\min_W \sum_i \lVert W x_i - y_i \rVert^2$ subject to $W^\top W = I$, all embeddings unit length. Orthogonal $W$ only rotates and reflects.
> 2. **Overfitting.** An unconstrained $W$ can stretch and shear to fit the seed pairs and fail on the words outside the dictionary. Orthogonal $W$ preserves geometry, $\lVert Wx \rVert^2 = x^\top W^\top W x = \lVert x \rVert^2$, so the space moves rigidly.
> 3. **Objective mismatch.** Training uses Euclidean distance, retrieval cosine. For unit vectors and orthogonal $W$: $\lVert Wx - y \rVert^2 = 2 - 2\cos(Wx, y)$, so they coincide.
> 4. **Solution.** $U\Sigma V^\top = \operatorname{SVD}(YX^\top)$, $W^* = UV^\top$: exact, one SVD.
> 5. **Evidence (English to Spanish).** The unconstrained map gets worse as the dimension grows ($d^2$ parameters, same dictionary); the orthogonal map is better at every dimension and improves slightly.
>
> Losing marks: claiming Euclidean training and cosine retrieval always agree, or calling the unconstrained map "Procrustes" in the results-table sense.

> [!exam]- Explain the hubness problem in crosslingual word retrieval and how CSLS corrects it, including which term of the formula does the work.
> **Key points:** hubs are nearest neighbours of many points, worse in high dimensions; $\text{CSLS} = 2\cos - r_T - r_S$; $r_S(y)$ penalises hubs; $r_T(x)$ constant per source word; better retrieval with only $K$ to set.
>
> - **Hubness:** in high-dimensional spaces a few vectors become the nearest neighbour of many points regardless of real similarity; a general property of high-dimensional geometry that worsens with dimension. One target hub becomes the "translation" of many unrelated source words. Nearest neighbour is asymmetric.
> - **CSLS (Conneau et al., 2018):** $2\cos(x,y) - r_T(x) - r_S(y)$, with $r_T(x)$ the mean cosine of $x$ to its $K$ nearest target vectors and $r_S(y)$ the mean cosine of $y$ to its $K$ nearest mapped source vectors.
> - **Which term works:** $r_S(y)$ penalises a candidate close to many things. $r_T(x)$ is the same for every candidate of a fixed $x$, so it matters only when comparing scores across source words.
> - **Effect:** better than plain nearest neighbour on every language pair, with no tuning beyond $K$.

> [!exam]- Describe the full MUSE pipeline of Conneau et al. (2018) for aligning two embedding spaces without any bilingual signal, and justify each design choice.
> **Key points:** linear generator $W$ against MLP discriminator; linear so word correspondence follows the distribution match; label smoothing, 50k frequent words, re-orthogonalisation; CSLS-based checkpoint selection; Procrustes refinement on mutual nearest neighbours.
>
> 1. **Adversarial step.** Generator = the mapping $W$, a single $d \times d$ linear matrix initialised to the identity (or a random orthogonal matrix); discriminator = an MLP (2 hidden layers of 2048, ReLU, input dropout) telling $Wx_i$ from real $y_j$. Steps alternate with the other side frozen; the generator uses flipped labels.
> 2. **Linear generator.** A deep generator could match the target distribution while scrambling which word lands where.
> 3. **Label smoothing $s = 0.2$** keeps the discriminator from becoming overconfident, which would starve the generator of gradient.
> 4. **Only the 50k most frequent words**: rare words have poorly estimated embeddings.
> 5. **Re-orthogonalisation** $W \leftarrow (1+\beta)W - \beta(WW^\top)W$, $\beta = 0.01$, keeps $W$ near orthogonal.
> 6. **Checkpoint selection.** The adversarial loss is unreliable; instead average the cosine between the 10k most frequent source words and their CSLS nearest targets, and keep the best checkpoint.
> 7. **Refinement.** Synthetic dictionary of high-confidence mutual CSLS nearest neighbours among frequent words, $W = UV^\top$ by SVD, repeat. Each component adds accuracy; both CSLS and refinement together are best.

> [!exam]- What is fastText, how does it differ from Skip-gram with negative sampling, and where does it help and not help?
> **Key points:** sum of character n-grams of `<w>` (3 to 6) plus the word; same loss, only the centre word decomposed; vectors for unseen words; gains on rich morphology, rare words and syntactic analogies; none on semantic analogies.
>
> - **Problem:** one opaque vector per word type, so `run` and `runs` share nothing, morphologically rich languages split lemmas into many rare forms, and unseen words get no vector.
> - **Model:** $v_w = \sum_{g \in G_w} z_g$, with $G_w$ the n-grams of `<w>` for $n$ from 3 to 6 plus `<w>` itself; `<` and `>` make prefixes and suffixes distinct.
> - **Same as SGNS:** the negative-sampling loss; context vectors $u_c$ stay per word.
> - **Different:** every n-gram of $w$ gets the same gradient; the output is the n-gram table, so any word gets a vector.
> - **Evidence (Bojanowski et al., 2017):** best or tied on almost all word-similarity sets, most in morphologically rich languages and on English rare words; the only loss is English WS-353. Building unseen words from n-grams helps further. Analogy gains are syntactic; semantic analogies do not improve and sometimes drop.

> [!exam]- How can the degree of isomorphism between two embedding spaces be measured without a dictionary, what did it show for English and Finnish, and what can it not show?
> **Key points:** Laplacian eigenvalues of nearest-neighbour graphs; $\Delta$ = sum of squared eigenvalue differences; no dictionary needed; high for the failing pairs and tracks adversarial success; low $\Delta$ does not prove isomorphism.
>
> 1. **Why:** isomorphism is true/false; Søgaard et al. want a degree.
> 2. **Eigenvector similarity:** nearest-neighbour graph per language (adjacency $A$), degree matrix $D$, Laplacian $L = D - A$, keep the largest $n\%$ of eigenvalues, $\Delta = \sum_i (\lambda_i^{(1)} - \lambda_i^{(2)})^2$.
> 3. **Reading it:** larger eigenvalues mean denser connectivity; larger $\Delta$ means more structurally different graphs. Eigenvalues ignore node labels, so no alignment is needed.
> 4. **Results:** English-Spanish has the lowest $\Delta$; English-Estonian, English-Finnish and English-Greek, where adversarial alignment fails, the highest.
> 5. **Finnish:** one lemma's many inflected forms cluster tightly, while English gives a flatter graph; no rotation can fix that, since a rotation keeps the neighbour graph.
> 6. **Limit:** near-isomorphic graphs have low $\Delta$, but low $\Delta$ does not imply near-isomorphism. $\Delta$ rules isomorphism out, never in.

> [!exam]- How should word embeddings be evaluated? Cover the evaluation axes, the word similarity and analogy tasks, and the limits of 2D visualisations.
> **Key points:** intrinsic against extrinsic, qualitative against quantitative; word similarity by Spearman, WS-353 relatedness against SimLex-999 similarity; analogies via $v_b - v_a + v_c$ scored by accuracy; 2D projections distort, t-SNE hyperparameters matter.
>
> - **Axes:** intrinsic (the embeddings by themselves) against extrinsic (downstream tasks such as MT, IR, summarisation); qualitative (selected examples, plots) against quantitative (an overall score). Independent of each other.
> - **Word similarity:** rank pairs by cosine, compare with human ratings by Spearman correlation. WS-353 annotates relatedness, SimLex-999 similarity; `coffee`/`cup` is related but not similar.
> - **Analogies:** *a : b :: c : X*, answer the word nearest to $v_b - v_a + v_c$, scored by accuracy; semantic and syntactic sets.
> - **Visualisation:** PCA (linear) or t-SNE (non-linear, local). Shows sense clusters and a consistent gender direction, but distorts distances, and t-SNE hyperparameters change the picture. Plots complement quantitative and extrinsic evaluation.

> [!exam]- Derive the closed-form solution of the unconstrained linear mapping $\min_W \sum_i \lVert W x_i - y_i \rVert^2$, state when it exists, and explain why this mapping gets worse as the embedding dimension grows.
> **Key points:** loss $\lVert WX - Y \rVert_F^2$; zero gradient gives $W = YX^\top(XX^\top)^{-1}$; needs $XX^\top$ invertible; SGD as the scalable alternative; $d^2$ parameters overfit.
>
> 1. **Matrix form.** Dictionary pairs as columns: $X$ is $d_1 \times n$, $Y$ is $d_2 \times n$. Loss $\lVert WX - Y \rVert_F^2$.
> 2. **Gradient** $2(WX - Y)X^\top$; set to zero: $WXX^\top = YX^\top$, so $W^* = YX^\top(XX^\top)^{-1}$.
> 3. **Existence.** $XX^\top$ must be invertible: at least $d_1$ linearly independent source vectors.
> 4. **Closed form against SGD.** Closed form is exact; SGD (Mikolov et al., 2013) is approximate, scales better and is more memory-friendly.
> 5. **Dimension.** $W$ has $d^2$ free parameters for the same dictionary, so accuracy falls as $d$ grows: it stretches and shears to fit the anchors and generalises poorly.

## Flashcards

> [!card]- What is the distributional hypothesis behind both count-based and prediction-based word vectors?
> A word is characterised by **the contexts it occurs in** (Firth: "the company it keeps").

> [!card]- When building count-based context vectors, is a word inside the window but outside the context vocabulary $V_c$ counted?
> **No.** It must be inside the window **and in $V_c$**.

> [!card]- Why is word similarity measured with cosine rather than the raw dot product?
> Cosine **ignores vector length**, so a frequent word's large counts do not dominate; only proportions matter.

> [!card]- What are the three properties of count-based context vectors?
> **High-dimensional**, **discrete** (counts), **sparse** (mostly zeros).

> [!card]- What is the main weakness of count-based context vectors besides sparsity?
> **No explicit criterion to separate signal from noise**: chance co-occurrences, and words like `the` that discriminate nothing.

> [!card]- How does idf weighting, $\log(N/n_w)$, help count-based context vectors?
> A context word found in nearly every context gets **idf close to 0**, so it is effectively removed.

> [!card]- What four properties distinguish prediction-based word embeddings from count-based context vectors?
> **Low-dimensional, dense, continuous**, and **learned by performing a prediction task**.

> [!card]- What is representation learning?
> Training where **learning good embeddings is the main objective**, the task being a pretext. In most models embeddings are by-products.

> [!card]- What is the difference between the CBOW and Skip-gram tasks?
> CBOW predicts **the centre word from its context**. Skip-gram predicts **each context word from the centre word**.

> [!card]- Why is CBOW useless as a language model?
> It also uses words **to the right** of the target; a language model uses left context only.

> [!card]- How is the CBOW hidden layer $h$ computed from the context words?
> As the **average** of the $2n$ context embeddings, $h = \frac{1}{2n} W w_C$, with no non-linearity.

> [!card]- What window sizes does CBOW typically use?
> **$n = m$**, with $n \in \{2, 5, 10\}$.

> [!card]- Are Word2Vec's input matrix $W$ and output matrix $W'$ tied together?
> **No, not necessarily**: $W' \neq W^\top$.

> [!card]- Which matrix is typically used as the Word2Vec embedding?
> **$W$** (its columns), or the combination $W^\top + W'$.

> [!card]- Why is CBOW called a "bag of words" model?
> The context vectors are **summed**, so **word order is lost**.

> [!card]- What independence assumption does Skip-gram make?
> The context words are **independent given the input (centre) word**.

> [!card]- Do Skip-gram's $2n$ output layers produce different distributions?
> **No.** They share $W'$ and $h$; only the target each is scored against differs.

> [!card]- What task replaces full-vocabulary prediction in negative sampling?
> **Binary classification**: is this (word, context) pair from the data or from a noise distribution?

> [!card]- How does negative sampling change the per-example cost of Word2Vec training?
> From **$|V|$ dot products to $k+1$**.

> [!card]- In negative sampling, what is the probability that a (word, context) pair is real?
> $\sigma(v_c \cdot v_w)$: the **sigmoid of the dot product** of the context and word vectors.

> [!card]- How many negative samples does Skip-gram add per observed pair?
> **5 to 20.**

> [!card]- Why are Word2Vec negatives drawn from $P(w)^{0.75}$ rather than the raw unigram distribution?
> To **bias sampling towards rarer words**; otherwise nearly every negative would be `the`, `of` or `and`.

> [!card]- How should Word2Vec's context size and number of negatives change with more data?
> They can be **smaller**.

> [!card]- Which negatives does a negative-sampling update push hardest against?
> Those the model **currently mistakes for real contexts** (high $\sigma(v_n \cdot v_w)$).

> [!card]- What weakness of prediction-based methods like Word2Vec does GloVe address?
> They treat **each context window as an independent event**, repeatedly rediscovering the same association.

> [!card]- Which tends to perform better: count-based or prediction-based word representations?
> **Prediction-based.** Count-based methods use global information directly but tend to perform worse.

> [!card]- Which method counts co-occurrences once over the whole corpus and then fits vectors to the log counts?
> **GloVe.**

> [!card]- In GloVe's weighted least-squares objective, what is $w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j$ fitted to?
> The **log co-occurrence count** $\log X_{ij}$.

> [!card]- What is the purpose of GloVe's weighting function $f(X_{ij})$?
> To **de-emphasise very rare (noisy) and very frequent (uninformative)** co-occurrences.

> [!card]- What does GloVe's weighting function do for frequent pairs?
> It **caps at 1**, so they cannot take over.

> [!card]- What are the typical parameters of GloVe's weighting function?
> **$x_\text{max} = 100$, $\alpha = 0.75$**.

> [!card]- Why do word pairs that never co-occur drop out of GloVe's objective?
> **$f(0) = 0$**, so their terms (where $\log X_{ij}$ is undefined) vanish.

> [!card]- How does GloVe use the corpus during training?
> **One pass** builds $X$; training then sees only $X$, never the corpus.

> [!card]- What is GloVe's final embedding for word $i$?
> **$w_i + \tilde{w}_i$**: the word and context vectors summed.

> [!card]- How do Word2Vec and GloVe differ in handling frequency imbalance?
> Word2Vec **subsamples frequent words**. GloVe uses the **explicit weighting function** $f(X_{ij})$.

> [!card]- What is the difference between intrinsic and extrinsic evaluation of word embeddings?
> Intrinsic: the embeddings **by themselves**. Extrinsic: their usefulness **in downstream tasks**.

> [!card]- How is word similarity evaluated against human judgements?
> Rank word pairs by **cosine** and compare with human ratings by **Spearman (rank) correlation**.

> [!card]- What is the difference between the WS-353 and SimLex-999 benchmarks?
> WS-353 annotates **relatedness** (`coffee`/`cup` close). SimLex-999 annotates **similarity**.

> [!card]- How is the answer to "a is to b as c is to X" computed with word embeddings?
> The word nearest by cosine to **$v_b - v_a + v_c$**, excluding the input words. Scored by accuracy.

> [!card]- What is the difference between PCA and t-SNE for visualising embeddings?
> PCA: **linear**, top-variance directions. t-SNE: **non-linear**, keeps local neighbours, gives up global distances.

> [!card]- Why should 2D t-SNE plots of embeddings not be taken at face value?
> Non-linear projections **distort distances**, and t-SNE **hyperparameters substantially change** the picture.

> [!card]- Why do morphologically rich languages suffer most from one vector per word type?
> Each lemma has **many inflected forms**, each a separate, rarer word with a worse-estimated vector.

> [!card]- How does fastText compute a word's vector?
> As the **sum of its character n-gram vectors**, plus the whole word.

> [!card]- What character n-gram lengths does fastText use by default?
> **3 to 6**.

> [!card]- Why does fastText wrap each word in `<` and `>`?
> So **prefixes and suffixes are distinct n-grams**: `<ru` only starts a word, `ns>` only ends one.

> [!card]- In fastText, are the context vectors also built from n-grams?
> **No.** Only the **centre word** is decomposed; context vectors stay per word.

> [!card]- How do the n-grams of a word share the gradient in fastText?
> **Every n-gram gets the same gradient**, since the word vector is their sum.

> [!card]- Why can fastText give a vector to a word never seen in training?
> It **sums the vectors of the word's character n-grams**, most of which were seen.

> [!card]- Where does fastText gain most on word similarity over Word2Vec?
> In **morphologically rich languages**; it also helps **rare words**.

> [!card]- Why do semantic analogies not benefit from fastText's character n-grams?
> Relations like **capital and country have nothing to do with spelling**; shared n-grams can add noise.

> [!card]- What is building a bilingual dictionary by nearest-neighbour search in a crosslingual embedding space called?
> **Bilingual lexicon induction** (BLI).

> [!card]- What are the three basic set-ups for training crosslingual embeddings?
> **Separately, then align spaces**; **together, then align regions**; on **word- or sentence-aligned data**.

> [!card]- What does the shared space hypothesis claim?
> Separately trained spaces are **approximately isomorphic**: similar shape, different coordinates and orientation.

> [!card]- Why is the crosslingual mapping assumed to be linear?
> Analogies are **already linear regularities** within one space, assumed to hold across spaces.

> [!card]- What bilingual signal does each type of crosslingual alignment use?
> Supervised: **thousands** of seed translations. Semi-supervised: **hundreds**, then bootstrapping. Unsupervised: **none**.

> [!card]- Why can seed dictionaries contain only single words?
> There are **no embeddings for multi-word expressions**, so phrases cannot be anchors.

> [!card]- What is the closed-form solution of the unconstrained least-squares mapping $\min_W \sum_i \lVert W x_i - y_i \rVert^2$?
> $W^* = Y X^\top (X X^\top)^{-1}$.

> [!card]- What is the difference between solving the least-squares mapping in closed form and with SGD?
> Closed form is **exact**. SGD is **approximate** but scales better and saves memory; Mikolov et al. used SGD.

> [!card]- In crosslingual results tables, what does "Procrustes" refer to?
> The **orthogonally constrained** mapping solved by SVD.

> [!card]- How does linear-mapping translation accuracy change for rarer words?
> It **drops**: rarer words have worse embeddings.

> [!card]- How does linear-mapping translation accuracy change with more monolingual training data?
> It **rises**, roughly with the log of the data size, then flattens.

> [!card]- Why does an unconstrained mapping $W$ overfit the seed dictionary?
> It can **stretch, shear and rotate** the space to fit the seed pairs, and generalises poorly.

> [!card]- What mismatch between training and inference does the unconstrained mapping have?
> Training minimises **Euclidean distance**, retrieval uses **cosine**. They agree only for unit-normalised vectors.

> [!card]- What constraint and normalisation define Xing et al.'s orthogonal mapping?
> **$W^\top W = I$**, with all embeddings normalised to **unit length**.

> [!card]- What transformations can an orthogonal mapping $W$ perform?
> Only **rotations and reflections**: no scaling or shearing.

> [!card]- How is the orthogonal Procrustes mapping solved?
> **Exactly by SVD**: $U \Sigma V^\top = \operatorname{SVD}(Y X^\top)$, $W = U V^\top$.

> [!card]- Why does the unconstrained mapping get worse as the embedding dimension grows?
> $W$ has **$d^2$ free parameters** for the same seed dictionary, so more room to overfit.

> [!card]- How does the orthogonal mapping compare with the unconstrained one across embedding dimensions?
> **Better at every dimension**, and it **improves slightly** as dimension grows.

> [!card]- What is a hub in a high-dimensional embedding space?
> A vector that is **the nearest neighbour of many points regardless of actual similarity**.

> [!card]- How does hubness change with dimensionality?
> It **worsens** as dimensionality increases.

> [!card]- What is the CSLS score between a mapped source vector $x$ and a target vector $y$?
> $\text{CSLS}(x, y) = 2\cos(x, y) - r_T(x) - r_S(y)$.

> [!card]- Which CSLS term penalises hubs?
> **$r_S(y)$**: the candidate's average cosine to its $K$ nearest mapped source vectors.

> [!card]- Why does $r_T(x)$ not change which translation CSLS picks for a given source word?
> It is **the same for every candidate** $y$.

> [!card]- How does CSLS retrieval compare with plain nearest neighbour on the same supervised mapping?
> **CSLS is better on every language pair**, with no tuning beyond $K$.

> [!card]- What is the chicken-and-egg problem of unsupervised crosslingual alignment?
> You need a **mapping to find translation pairs**, and **translation pairs to learn the mapping**.

> [!card]- In MUSE's adversarial training, what does the discriminator try to distinguish?
> **Mapped source vectors** $Wx$ from **real target vectors** $y$. The generator is $W$.

> [!card]- Why is MUSE's generator a single linear matrix rather than a deep network?
> A deep generator could **match the target distribution while scrambling which word maps where**.

> [!card]- How is MUSE's mapping $W$ initialised?
> As the **identity** (or a random orthogonal matrix).

> [!card]- Why does MUSE use label smoothing ($s = 0.2$) on the discriminator?
> To stop it becoming **overconfident**, which would starve the generator of gradient.

> [!card]- Why does MUSE feed the discriminator only the 50k most frequent words?
> **Rare words have poorly estimated embeddings**; frequent ones give a cleaner signal.

> [!card]- What does re-orthogonalisation do in MUSE's adversarial training?
> After each step it **pulls $W$ back towards an orthogonal matrix** ($\beta = 0.01$).

> [!card]- What unsupervised criterion does MUSE use to select a checkpoint?
> The **average cosine** between the 10k most frequent source words and their **CSLS nearest** targets.

> [!card]- What can MUSE's unsupervised validation criterion not guarantee?
> That the induced pairs are **correct translations**; it only shows mapped words land near real ones.

> [!card]- How does MUSE build its synthetic dictionary for refinement?
> From **mutual CSLS nearest neighbours** among frequent words, keeping only high-confidence pairs.

> [!card]- Why does MUSE's synthetic dictionary keep only mutual nearest neighbours?
> A pair is trusted only if each word picks the other, which **filters out hub matches**.

> [!card]- Why does the MUSE refinement loop count as semi-supervised?
> It runs **supervised Procrustes on a dictionary the system built itself**.

> [!card]- How does fully unsupervised MUSE compare with supervised Procrustes-CSLS for close pairs (en-es, en-fr, en-de)?
> It **matches or slightly beats** supervised alignment.

> [!card]- How does fully unsupervised MUSE compare with supervised alignment for distant pairs (en-ru, en-zh)?
> It **falls well short**.

> [!card]- What is the difference between head-marking and dependent-marking?
> Head-marking puts the grammatical marker **on the head**. Dependent-marking puts it **on the dependent**.

> [!card]- In a clause, which word is the head and which are the dependents?
> Head: the **verb**. Dependents: **subject and object**.

> [!card]- Which language pairs did adversarial alignment fail on completely in Søgaard et al. (2018)?
> **English with Estonian, Finnish and Greek**: the mixed- or double-marking languages.

> [!card]- What is identical-word supervision for crosslingual alignment?
> A seed dictionary of **words spelled identically in both languages**. No human dictionary needed.

> [!card]- What does the Estonian-Finnish result show about unsupervised alignment?
> **Structurally similar** languages align adversarially even when neither aligns with English.

> [!card]- Why does unsupervised alignment fail for mixed- and double-marking languages, according to Søgaard et al.?
> Each lemma has **many distinct surface forms**, each its own lower-frequency embedding, so the spaces are non-isomorphic.

> [!card]- Is the English-Finnish alignment failure caused by too little Finnish training data?
> **No.** A far larger Finnish corpus still gave zero precision.

> [!card]- For English-Hungarian unsupervised alignment, which parts of speech do worst?
> **Nouns and verbs**, the most inflected.

> [!card]- How does a domain mismatch between the two corpora affect identical-word supervision and adversarial alignment?
> Identical-word supervision **degrades but survives**. Adversarial alignment **collapses to about zero**.

> [!card]- How much do hyperparameter changes within one embedding algorithm (window, n-gram range) hurt unsupervised alignment?
> **Barely.**

> [!card]- What happens to unsupervised alignment when one language uses Skip-gram embeddings and the other CBOW?
> It **fails completely**, even with otherwise identical settings.

> [!card]- How is eigenvector similarity $\Delta$ computed for two languages?
> **Sum of squared differences** between the largest Laplacian eigenvalues of their nearest-neighbour graphs.

> [!card]- What does a larger eigenvector similarity $\Delta$ mean?
> The graphs are **more structurally different**: further from isomorphism.

> [!card]- Why does eigenvector similarity not need a bilingual dictionary?
> Laplacian eigenvalues **do not change when nodes are relabelled**.

> [!card]- Why can no rotation fix a high eigenvector similarity $\Delta$?
> A rotation **preserves the nearest-neighbour graph**, so the eigenvalues stay the same.

> [!card]- Why does Finnish give a high eigenvector similarity $\Delta$ against English?
> One lemma's inflected forms **cluster tightly**; English gives a **flatter, more evenly connected** graph.

> [!card]- How does eigenvector similarity $\Delta$ relate to adversarial alignment success?
> It **tracks it**: the lowest-$\Delta$ pair aligns best, the highest-$\Delta$ pairs fail.

> [!card]- What does "static" mean for Word2Vec, GloVe and fastText embeddings?
> **One vector per word type**, whatever the sentence: `bank` is the same in *river bank* and *bank account*.

> [!card]- True or false: In Skip-gram, the hidden layer is the average of the context word embeddings.
> **False.** That is CBOW. Skip-gram's hidden layer is the centre word's embedding.

> [!card]- True or false: CBOW passes its averaged context vector through a non-linear activation before the output layer.
> **False.** CBOW has no non-linearities.

> [!card]- True or false: Word2Vec's context size and number of negative samples can be smaller the more training data there is.
> **True.** More data allows a smaller context and a smaller negative set.

> [!card]- True or false: Raising the unigram distribution to the power 0.75 biases negative sampling towards frequent words.
> **False.** It biases sampling towards **rarer** words.

> [!card]- True or false: GloVe revisits the corpus in every training epoch.
> **False.** One pass builds $X$; training then uses only $X$.

> [!card]- True or false: GloVe's weighting function gives the most frequent co-occurrences unlimited weight.
> **False.** It caps at 1.

> [!card]- True or false: fastText can produce a vector for a word that never occurred in training.
> **True.** It sums the word's character n-gram vectors.

> [!card]- True or false: fastText decomposes both the centre word and the context words into character n-grams.
> **False.** Only the centre word; context vectors stay per word.

> [!card]- True or false: fastText improves over Word2Vec on both semantic and syntactic analogies.
> **False.** Gains are syntactic; semantic analogies do not improve and sometimes get worse.

> [!card]- True or false: Semi-supervised alignment starts from a seed dictionary of thousands of translations.
> **False.** It starts from **hundreds**; thousands is the supervised setting.

> [!card]- True or false: For unit-length vectors and an orthogonal $W$, minimising Euclidean distance is equivalent to maximising cosine similarity.
> **True.** $\lVert Wx - y \rVert^2 = 2 - 2\cos(Wx, y)$.

> [!card]- True or false: An orthogonal mapping can rotate, reflect and scale the source space.
> **False.** No scaling: only rotations and reflections.

> [!card]- True or false: The unconstrained least-squares mapping continuously improves as the embedding dimension grows.
> **False.** It gets worse, since $d^2$ parameters overfit the same dictionary.

> [!card]- True or false: CSLS penalises target words that are close to many mapped source vectors.
> **True.** That is the $r_S(y)$ term.

> [!card]- True or false: CSLS requires extensive hyperparameter tuning.
> **False.** Only the neighbourhood size $K$.

> [!card]- True or false: In MUSE, the adversarial (discriminator) loss is a reliable criterion for choosing which checkpoint to keep.
> **False.** It can look fine while the mapping is poor; MUSE uses average CSLS-neighbour cosine.

> [!card]- True or false: MUSE's generator is a deep non-linear network, as in a standard GAN.
> **False.** It is a single linear $d \times d$ matrix, deliberately.

> [!card]- True or false: Adversarial unsupervised alignment does not suffer from a domain mismatch between the two monolingual corpora.
> **False.** Across domains it collapses to about zero, even for English-Spanish.

> [!card]- True or false: English-Finnish unsupervised alignment fails because the Finnish corpus is too small.
> **False.** It stays at zero with a far larger Finnish corpus.

> [!card]- True or false: In Søgaard et al.'s language comparison, identical-word supervision beats adversarial alignment on every pair involving English.
> **True.** From a small margin (English-Spanish) to working against not working (English-Estonian).

> [!card]- True or false: Eigenvector similarity requires a seed dictionary to align the two graphs.
> **False.** Eigenvalues do not depend on node labels.

> [!card]- True or false: A low eigenvector similarity $\Delta$ proves that two embedding spaces are near-isomorphic.
> **False.** Near-isomorphic spaces give a low $\Delta$, but not the other way round.

> [!card]- Multiple selection: which are true of negative sampling in Word2Vec? (A) It turns prediction into classifying real against noise pairs (B) Each observed pair gets $k$ negatives, typically 5 to 20 (C) It still normalises over the whole vocabulary for each example (D) Negatives are drawn uniformly from the vocabulary
> **A and B.** C: the cost drops to $k+1$ dot products. D: negatives come from $P(w)^{0.75}$.

> [!card]- Multiple selection: which are true of GloVe? (A) It fits dot products plus biases to log co-occurrence counts (B) Its weighting function caps at 1 for frequent pairs (C) It treats each context window as an independent event (D) Its final embedding sums the word and context vectors
> **A, B and D.** C: that is the Word2Vec weakness GloVe addresses.

> [!card]- Multiple selection: which are true of fastText? (A) A word's vector is the sum of its character n-gram vectors and the word itself (B) Context vectors are also built from n-grams (C) Its gains over Word2Vec are mostly on syntactic analogies (D) It returns a table of n-gram vectors
> **A, C and D.** B: only the centre word is decomposed.

> [!card]- Multiple selection: which are true of hubness and CSLS? (A) Hubness worsens as dimensionality increases (B) CSLS penalises candidates that are close to many mapped source vectors (C) CSLS requires extensive hyperparameter tuning (D) Nearest-neighbour retrieval is symmetric
> **A and B.** C: only $K$. D: a hub can be `cat`'s nearest neighbour without `cat` being the hub's.

> [!card]- Multiple selection: which are true of MUSE? (A) The discriminator and the mapping are trained in alternation, each with the other frozen (B) The mapping is kept near-orthogonal after each step (C) The adversarial loss is used to pick the final checkpoint (D) Refinement re-solves the mapping by SVD on mutual nearest neighbours
> **A, B and D.** C: the checkpoint is picked by average CSLS-neighbour cosine.

> [!card]- Multiple selection: which are true of unsupervised (adversarial) alignment, according to Søgaard et al. (2018)? (A) English-Finnish fails even with far more Finnish data (B) It survives a domain mismatch if the languages are close (C) It works when one language uses Skip-gram and the other CBOW (D) Estonian-Finnish aligns adversarially
> **A and D.** B: cross-domain English-Spanish collapses to about zero. C: mixing Skip-gram and CBOW fails.

## Links

- **Course:** [[MNLP - Overview|Course overview]]
- **Previous:** [[MNLP-L04 - Subword Segmentation]] (subword units; fastText's character n-grams are the embedding-side answer to the same vocabulary problem)
- **Next:** [[MNLP-L06 - Contextual Embeddings]] (vectors that depend on the sentence, built with [[Recurrent Neural Network (RNN)|RNNs]], [[Self-Attention]] and [[Transformers]], the basis of [[Large Language Models (LLM)]])
- **Later:** [[MNLP-L07 - Crosslingual NLP]] (crosslingual transfer, building on the aligned spaces of this lecture)
- **Related concepts:** [[Word Embeddings]] · [[Negative Sampling]] · [[Contrastive Learning]] · [[Vector Space Model]] · [[Bag of Words]] · [[Tokenization]]
- **Source:** `embeddings_static.pdf`, Christof Monz, MNLP: Static Embeddings, slides 1 to 71
- **Primary references:** Firth (1957) · Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space* and *Distributed Representations of Words and Phrases and their Compositionality* (Word2Vec, negative sampling) · Mikolov, Le and Sutskever (2013), *Exploiting Similarities among Languages for Machine Translation* (linear translation matrix) · Pennington, Socher and Manning (2014), *GloVe: Global Vectors for Word Representation* · Wattenberg, Viégas and Johnson (2016), *How to Use t-SNE Effectively* · Bojanowski et al. (2017), *Enriching Word Vectors with Subword Information* (fastText) · Xing et al. (2015), *Normalized Word Embedding and Orthogonal Transform for Bilingual Word Translation* · Conneau et al. (2018), *Word Translation Without Parallel Data* (MUSE, CSLS) · Søgaard, Ruder and Vulić (2018), *On the Limitations of Unsupervised Bilingual Dictionary Induction*
