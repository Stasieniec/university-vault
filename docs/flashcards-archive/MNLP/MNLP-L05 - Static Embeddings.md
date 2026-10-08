# Archived flashcards: MNLP-L05 - Static Embeddings

The long-form cards this note carried until 2026-10-08, when they were replaced by short single-fact cards. Kept for reference; not published.

Click a question to reveal its answer, or press **Study** to drill the whole set. Cards marked as exam questions are meant to be answered out loud or on paper first, then checked against the points listed.

> [!exam]- Does unsupervised bilingual lexicon induction work? Answer with the mechanism, the headline result and the conditions under which it fails, with numbers.
> Must hit:
> 1. **Mechanism.** The shared space hypothesis says separately trained embedding spaces are approximately isomorphic, so one linear (orthogonal) map should align them. MUSE (Conneau et al., 2018) bootstraps a rough $W$ adversarially (a linear generator against an MLP discriminator, with re-orthogonalisation), picks the checkpoint with the unsupervised CSLS criterion, then refines with Procrustes ($W = UV^\top$) on a synthetic dictionary of mutual CSLS nearest neighbours, repeated.
> 2. **Headline result.** Fully unsupervised matches or beats supervised Procrustes-CSLS on close, well-resourced pairs (P@1): en-es 81.7 against 81.4, es-en 83.3 against 82.9, en-fr 82.3 against 81.1, en-de 74.0 against 73.5.
> 3. **Where it falls short.** Distant pairs: en-ru 44.0 against 51.7, en-zh 32.5 against 42.7.
> 4. **Where it collapses (Søgaard et al., 2018).** Mixed- or double-marking, case-rich languages: EN-ET 0.00, EN-FI 0.09, EN-EL 0.07. EN-FI stays at 0.0 even when retrained on 1.7 billion Finnish words, so data size is not the explanation. Mismatched domains: 0.0 to 0.13 in every cross-domain en-es cell. Mismatched algorithms: Spanish CBOW against English skipgram gives 0.00 to 0.13.
> 5. **A free baseline beats it.** A seed dictionary of identically spelled words beats adversarial training on every pair involving English (EN-ES 82.62 against 81.89, EN-ET 31.45 against 0.00).
> 6. **Verdict.** It works when the languages are typologically similar, the corpora share a domain and both sides use the same embedding algorithm. Those are the conditions it was evaluated under, and they are the opposite of the low-resource setting that motivates it.
>
> Losing marks: giving only the Conneau et al. headline numbers, or blaming the failures on data size alone.

> [!exam]- Derive the negative sampling objective from a binary classification set-up, then derive its gradient with respect to the input vector $v_w$ and interpret it.
> 1. **Why.** A full softmax needs a score for all $\lvert V \rvert$ words per training example. Replace "which word is the context?" with "is this (word, context) pair real or noise?": 1 positive pair from the data, $k$ negative pairs from a noise distribution.
> 2. **Likelihood.** $\arg\max_\theta \prod_{(w,c) \in D} p(D=1 \mid c,w;\theta) \prod_{(w,c) \in D'} p(D=0 \mid c,w;\theta)$, with $D$ the observed pairs, $D'$ the noise pairs, and $p(D=1 \mid c,w) = \sigma(v_c \cdot v_w)$, where $v_w = Ww$ (column of $W$) and $v_c = W'^\top c$ (row of $W'$).
> 3. **Sigmoid identity.** $1 - \sigma(x) = \frac{e^{-x}}{1+e^{-x}} = \frac{1}{e^x + 1} = \sigma(-x)$, so the objective becomes $\prod_D \sigma(v_c \cdot v_w) \prod_{D'} \sigma(-v_c \cdot v_w)$, and after logs
> $$\sum_{D} \log \sigma(v_c \cdot v_w) + \sum_{D'} \log \sigma(-v_c \cdot v_w)$$
> 4. **Per-example loss.** $\mathcal{L} = -\log \sigma(v_c \cdot v_w) - \sum_{j=1}^{k} \log \sigma(-v_{n_j} \cdot v_w)$.
> 5. **Gradient.** With $\frac{d}{dx}\log\sigma(x) = 1 - \sigma(x)$:
> $$\frac{\partial \mathcal{L}}{\partial v_w} = (\sigma(v_c \cdot v_w) - 1)\, v_c + \sum_{j} \sigma(v_{n_j} \cdot v_w)\, v_{n_j}$$
> 6. **Interpretation.** The positive coefficient is negative, so a descent step moves $v_w$ towards $v_c$; each negative coefficient is positive, so $v_w$ moves away from $v_{n_j}$. Both shrink to zero once the classifier gets the pair right, so the biggest corrections come from negatives the model mistakes for real contexts. Cost per example drops from $\lvert V \rvert$ dot products to $k+1$.

> [!exam]- Compare Word2Vec (CBOW and Skip-gram) with GloVe: what each optimises, what information each uses, how each handles frequency imbalance, and what the final embedding is.
> - **CBOW:** predict the centre word from the average of $2n$ context embeddings, $h = \frac{1}{2n} W w_C$, $\hat{y} = \operatorname{softmax}(W'h)$, cross entropy, SGD.
> - **Skip-gram:** predict each of the $2n$ context words from the centre word, $h = W w_I$, context words assumed independent given $w_I$; in practice trained with negative sampling.
> - **GloVe (Pennington et al., 2014):** weighted least-squares regression on log co-occurrence counts, $J = \sum f(X_{ij})(w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$.
> - **Information used:** Word2Vec sees individual local context windows, treating each as an independent event, so it keeps "rediscovering" the same association. GloVe counts once over the whole corpus (global statistics) and then fits gradient updates to those counts.
> - **Frequency imbalance:** Word2Vec subsamples frequent words (and draws negatives from $P(w)^{0.75}$); GloVe uses the explicit weighting $f(X_{ij})$ with $x_\text{max} = 100$, $\alpha = 0.75$.
> - **Interpretability:** Word2Vec is indirect (it learns to predict contexts); GloVe directly fits log counts.
> - **Final embedding:** Word2Vec typically $W$, or $W^\top + W'$; GloVe $w_i + \tilde{w}_i$. Both sum the two vector sets.
> - Context: prediction-based methods tend to outperform count-based ones; count-based ones use global information directly. GloVe is the hybrid.

> [!exam]- Why does constraining a cross-lingual mapping to be orthogonal help? Give the objective, the two problems it fixes with a derivation for each, its closed-form solution, and the evidence.
> 1. **Objective (Xing et al., 2015).** $W^* = \arg\min_W \sum_i \lVert W x_i - y_i \rVert^2$ subject to $W^\top W = I$, with all embeddings normalised to unit length. An orthogonal $W$ only rotates and reflects: no scaling or shearing.
> 2. **Problem 1, overfitting.** An unconstrained $W$ can stretch, shear and rotate, so it can warp the space to fit the seed pairs and fail to generalise to the words outside the dictionary (the majority). Orthogonal $W$ preserves geometry: $\lVert Wx \rVert^2 = x^\top W^\top W x = x^\top x$, so the source space moves rigidly, which is exactly what the shared space hypothesis says should suffice.
> 3. **Problem 2, objective mismatch.** Training minimises Euclidean distance, retrieval uses cosine. For unit vectors and orthogonal $W$: $\lVert Wx - y \rVert^2 = \lVert Wx \rVert^2 + \lVert y \rVert^2 - 2(Wx)\cdot y = 2 - 2\cos(Wx, y)$, so minimising distance is maximising cosine.
> 4. **Solution.** $U\Sigma V^\top = \operatorname{SVD}(YX^\top)$, $W^* = UV^\top$: one SVD of a $d \times d$ matrix, exact, no learning rate.
> 5. **Evidence (en-es, P@1).** Unconstrained (Mikolov et al., 2013) falls from 30.43% at 300 dimensions to 20.69% at 700, because $W$ has $d^2$ free parameters for the same dictionary. Orthogonal is better everywhere and rises slightly, 38.99% to 41.04%.
>
> Losing marks: claiming Euclidean training and cosine retrieval always agree (they coincide only for unit vectors under an orthogonal map), or calling the unconstrained least-squares map "Procrustes" (in the results tables Procrustes means the orthogonal SVD solution).

> [!exam]- Explain the hubness problem in cross-lingual word retrieval and how CSLS corrects it, including which term of the formula does the work.
> - **Hubness:** in high-dimensional spaces a few vectors become the nearest neighbour of many points regardless of real similarity. It is a general property of high-dimensional geometry and worsens with dimension. A hub sits near the centre of the cloud and is moderately close to everything.
> - **Effect on translation:** one target hub becomes the nearest-neighbour "translation" of many unrelated source words: nn(cat) = nn(car) = nn(house) = thing. Nearest neighbour is asymmetric: `thing` can be the neighbour of `cat` without `cat` being the neighbour of `thing`.
> - **CSLS (Conneau et al., 2018):** $\text{CSLS}(x, y) = 2\cos(x,y) - r_T(x) - r_S(y)$, with $x = Wx_s$, $r_T(x)$ the mean cosine of $x$ to its $K$ nearest target vectors, $r_S(y)$ the mean cosine of $y$ to its $K$ nearest mapped source vectors.
> - **Which term works:** $r_S(y)$ penalises a candidate that is close to lots of things (a hub has high $r_S$). $r_T(x)$ is constant across candidates for a fixed $x$, so it cannot change which $y$ wins; it matters when scores are compared across source words, e.g. ranking pairs to build a dictionary.
> - **Effect:** significantly better retrieval with no parameter tuning beyond $K$ (Conneau et al. use $K = 10$). Over plain NN on the same Procrustes map: en-es 77.4 to 81.4, fr-en 76.1 to 82.4.

> [!exam]- Describe the full MUSE pipeline of Conneau et al. (2018) for aligning two embedding spaces without any bilingual signal, and justify each design choice.
> 1. **Adversarial step.** Generator = the mapping $W$ (a single $d \times d$ linear matrix); discriminator = an MLP (2 hidden layers of 2048, ReLU, input dropout) that tells mapped source vectors $Wx_i$ from real target vectors $y_j$. They train in alternation: discriminator step with $W$ frozen, generator step with labels flipped and the discriminator frozen.
> 2. **Linear generator, deliberately.** A deep generator could match the target *distribution* while scrambling which source word lands where. Only a constrained, near-rigid map forces point-level correspondence to come with the distribution match.
> 3. **Label smoothing $s = 0.2$.** Targets $\{s, 1-s\}$ stop the discriminator becoming overconfident, which would give the generator vanishing gradients.
> 4. **Only the 50k most frequent words** are fed to the discriminator: rare words have poorly estimated embeddings, so frequent words give a cleaner signal.
> 5. **Re-orthogonalisation** after each step, $W \leftarrow (1+\beta)W - \beta(WW^\top)W$ with $\beta = 0.01$, keeps $W$ near orthogonal so it cannot drift into a warping general linear map.
> 6. **Model selection without a dictionary.** The adversarial loss is unreliable. Instead: for the 10k most frequent source words, find each one's CSLS nearest target, average the cosines, keep the checkpoint with the highest value.
> 7. **Refinement.** Build a synthetic dictionary of high-confidence mutual CSLS nearest neighbours among frequent words, re-solve $W = UV^\top$ by SVD (Procrustes), repeat. Translate with CSLS.
> 8. **Contribution of each part (en-es P@1):** adversarial alone 69.8, + CSLS 75.7, + refinement 79.1, both 81.7.

> [!exam]- What is fastText, how does it differ from Skip-gram with negative sampling, and what does the evidence of Bojanowski et al. (2017) say about where it helps and where it does not?
> - **Problem it addresses:** Word2Vec and GloVe give one opaque vector per word type, so `run`, `runs`, `running`, `runner` share no structure; morphologically rich languages (Finnish, Turkish, Russian) split each lemma into many rare forms; OOV words get no vector.
> - **Model:** the centre word vector is the sum of its character n-gram vectors, $v_w = \sum_{g \in G_w} z_g$, where $G_w$ holds all n-grams of `<w>` for $n \in [3, 6]$ plus the whole word `<w>`. `<` and `>` mark word boundaries so prefixes and suffixes are distinct n-grams.
> - **Same as SGNS:** the loss $-\log\sigma(v_w^\top u_c) - \sum_k \log\sigma(-v_w^\top u_{n_k})$, with context vectors $u_c$ still per word.
> - **Different:** every n-gram of $w$ gets the same gradient (since $\partial v_w / \partial z_g = I$); the output is the n-gram table $\{z_g\}$, and any word, seen or not, gets a vector by summing its n-grams.
> - **Word similarity:** sisg (fastText) best or tied-best on 9 of 10 datasets; biggest gains in morphologically rich languages (Russian HJ 59 to 66 over sg) and on English Rare Words (43 to 47). Building OOV vectors from n-grams (sisg against sisg-) adds up to 6 points. Only loss: English WS353 (71 against cbow's 73), frequent words where whole-word vectors are already good.
> - **Analogies:** gains are syntactic (Czech 52.8 to 77.8, German +11.9, Italian +11.2, English +4.8 over sg). Semantic analogies do not improve and sometimes drop (German 66.5 to 62.3), because capital-country relations have nothing to do with character overlap.

> [!exam]- How can the degree of isomorphism between two embedding spaces be measured without a dictionary, what did it show for English and Finnish, and what can it not show?
> 1. **Why a measure is needed:** isomorphism is a strict true/false criterion; Søgaard et al. (2018) want a degree.
> 2. **Eigenvector similarity:** build a nearest-neighbour graph per language (adjacency matrix $A$), take the degree matrix $D$, form the Laplacian $L = D - A$, keep the largest $n\%$ of its eigenvalues, and compute $\Delta = \sum_i (\lambda_i^{(1)} - \lambda_i^{(2)})^2$.
> 3. **Reading it:** larger Laplacian eigenvalues mean denser connectivity; larger $\Delta$ means more structurally different graphs. Eigenvalues are invariant to relabelling the nodes, so no alignment is needed.
> 4. **Results:** EN-ES 2.07 is the lowest; the three pairs where adversarial alignment fails completely have the largest values (EN-ET 6.61, EN-FI 7.33, EN-EL 5.01). $\Delta$ tracks adversarial success.
> 5. **Finnish explanation:** a lemma spreads across dozens of inflected forms that cluster tightly by meaning overlap, while English gives a flatter, more evenly connected graph. Different connectivity means high $\Delta$, and no rotation can fix it, since a rotation preserves the neighbour graph and so the eigenvalues.
> 6. **Limit:** near-isomorphic graphs have low $\Delta$, but a low $\Delta$ does not imply near-isomorphism (non-isomorphic cospectral graphs exist). $\Delta$ can rule isomorphism out, never in.
>
> Losing marks: concluding that a low $\Delta$ proves the spaces are isomorphic.

> [!exam]- How should word embeddings be evaluated? Cover the evaluation axes, the word similarity and analogy tasks, and the limits of 2D visualisations.
> - **Two independent axes:** intrinsic (how good the embeddings are by themselves) against extrinsic (usefulness in downstream tasks such as summarisation, MT, IR); qualitative (inspect selected examples, neighbours, plots) against quantitative (an overall task-dependent score). A 2D plot is intrinsic and qualitative; a similarity correlation is intrinsic and quantitative; BLEU after plugging embeddings into MT is extrinsic and quantitative.
> - **Word similarity:** rank word pairs by cosine, compare with human judgements by Spearman correlation. WS-353 annotates relatedness, SimLex-999 similarity; `coffee`/`cup` is related but not similar, so the two benchmarks can rank the same embeddings differently. Both mix kinds of similarity.
> - **Analogies:** *a : b :: c : X*, answer the word closest by cosine to $v_b - v_a + v_c$ (input words excluded), scored by accuracy; semantic and syntactic sets.
> - **Visualisation:** project to 2D with PCA (linear, top-variance directions) or t-SNE (non-linear, keeps local neighbours, gives up global distances). Useful for showing sense clusters or a consistent gender direction.
> - **Caveats:** non-linear projections distort distances; t-SNE hyperparameters change cluster sizes, distances and even apparent clusters (Wattenberg 2016). Plots complement quantitative and extrinsic evaluation and do not substitute for it.

> [!exam]- Derive the closed-form solution of the unconstrained linear mapping $\min_W \sum_i \lVert W x_i - y_i \rVert^2$, state when it exists, and explain why this mapping gets worse as the embedding dimension grows.
> 1. **Matrix form.** Stack the $n$ dictionary pairs as columns: $X$ is $d_1 \times n$, $Y$ is $d_2 \times n$. Loss $= \lVert WX - Y \rVert_F^2$.
> 2. **Gradient.** $2(WX - Y)X^\top$ (per pair: $2\sum_i (Wx_i - y_i)x_i^\top$).
> 3. **Set to zero.** $WXX^\top = YX^\top$, so $W^* = YX^\top(XX^\top)^{-1}$.
> 4. **Existence.** $XX^\top$ ($d_1 \times d_1$) must be invertible, which needs at least $d_1$ linearly independent source vectors, i.e. $n \geq d_1$ pairs.
> 5. **Closed form against SGD.** Closed form is exact; SGD (used by Mikolov et al., 2013) is approximate, scales better with large dimensions and dictionaries, and never forms or inverts $XX^\top$.
> 6. **Dimension.** $W$ has $d^2$ free parameters for the same seed dictionary, so more dimensions means more room to overfit: en-es P@1 drops from 30.43% (300 dimensions) to 25.76% (500) to 20.69% (700). The unconstrained map can stretch and shear to fit the anchors and generalises poorly.

> [!card]- State the distributional hypothesis, who formulated it, and the two families of methods built on it.
> "You shall know a word by the company it keeps" (**Firth, 1957**). Represent each word by the contexts it occurs in; words in similar contexts get similar vectors and are taken to be similar in meaning.
> 1. **Count based** (distributional semantics): count context words directly.
> 2. **Prediction based** (neural embeddings): train a network to predict words from context or the reverse, and keep its weights.
>
> GloVe is the hybrid of the two.

> [!card]- Along which dimensions should word representations let us compute similarity?
> - **Meaning**, which itself has several dimensions: synonymy (`car`/`automobile`), topical relatedness (`car`/`road`), and others.
> - **Morphology**: `run`, `runs`, `running` should be recognisably related.

> [!card]- Give the count-based recipe for building context vectors, step by step.
> 1. Define a vocabulary $V_c$ of context words.
> 2. Define a vocabulary $V_t$ of target words (may equal $V_c$).
> 3. Define a window of size $n$ (or use the whole sentence or document as context).
> 4. For each occurrence of $w \in V_t$, count how often each $w' \in V_c$ occurs within $n$ words to the left or right.
> 5. Store the counts in a vector $c_w$, with $c_w[i]$ the frequency of context word $i$ in the context of $w$. The vector has $\lvert V_c \rvert$ dimensions.

> [!card]- With window $n = 3$ and $V_c = \{$leash, walk, run, owner, pet, bark$\}$, which words count as context for `dog` in "he took the dog for a walk", and why are `took` and `the` ignored?
> Only **walk** (3 to the right) counts. `took`, `the`, `for` and `a` are within 3 tokens too, but they are not in $V_c$, so they are ignored. A word only contributes if it is both inside the window and in the context vocabulary.

> [!card]- Give the cosine similarity formula, what each part means, its range for count vectors, and why vector length is deliberately ignored.
> $$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert_2 \lVert \mathbf{v} \rVert_2} = \frac{\sum_i u_i v_i}{\sqrt{\sum_i u_i^2}\sqrt{\sum_i v_i^2}}$$
> Numerator: dot product; denominator: product of Euclidean lengths. It is the cosine of the angle: 1 for the same direction, 0 for orthogonal; with non-negative counts it lies in $[0, 1]$.
> Length is ignored because a frequent word has large counts everywhere and a rare word small ones; only the proportions of contexts should decide similarity.

> [!card]- Worked example: context words `runs` and `legs`, with dog $= (1,4)$, cat $= (1,5)$, car $= (4,1)$. Compute the three cosines.
> - $\cos(\text{dog},\text{cat}) = \frac{1 + 20}{\sqrt{17}\sqrt{26}} = \frac{21}{21.02} = 0.999$ (about $2.7^\circ$)
> - $\cos(\text{dog},\text{car}) = \frac{4 + 4}{\sqrt{17}\sqrt{17}} = \frac{8}{17} = 0.471$ (about $61.9^\circ$)
> - $\cos(\text{cat},\text{car}) = \frac{4 + 5}{\sqrt{26}\sqrt{17}} = \frac{9}{21.02} = 0.428$ (about $64.7^\circ$)
>
> The two animals point almost the same way; `car` points elsewhere.

> [!card]- Worked example: over contexts (leash, walk, run, owner, pet, bark), dog $= (3,5,2,5,3,2)$ and cat $= (0,3,3,2,3,0)$. Compute their cosine.
> - Dot product: $0 + 15 + 6 + 10 + 9 + 0 = 40$.
> - $\lVert \text{dog} \rVert = \sqrt{9+25+4+25+9+4} = \sqrt{76} = 8.718$; $\lVert \text{cat} \rVert = \sqrt{0+9+9+4+9+0} = \sqrt{31} = 5.568$.
> - $\cos = 40 / (8.718 \times 5.568) = 40 / 48.54 = 0.824$.

> [!card]- In a count table over (leash, walk, run, owner, pet, bark), bark $= (1,0,0,2,1,0)$ and car $= (0,0,1,3,0,0)$ come out at cosine 0.775, higher than dog and car (0.617). Why, and what goes wrong for a word like `light` with an all-zero row?
> - bark and car share only `owner`: dot product $6$, norms $\sqrt{6} = 2.449$ and $\sqrt{10} = 3.162$, cosine $6/7.746 = 0.775$. With such short vectors one shared context word dominates: the signal-versus-noise problem of count vectors.
> - `light` has none of its contexts in $V_c$, so its vector is zero and its cosine with anything is $0/0$, undefined. A word whose contexts were not included has no representation at all.

> [!card]- What are the properties of count-based context vectors, and their advantages and disadvantages?
> **Properties:** high-dimensional (one dimension per context word, tens or hundreds of thousands), discrete (integer counts), sparse (almost all zeros).
> **Advantages:** an unsupervised way to induce word similarity; interpretable dimensions ($c_w[i]$ is the frequency of an actual word).
> **Disadvantages:** they stay relatively sparse despite tricks like stemming; there is no explicit criterion to separate signal from noise in the context: some words occur by chance, and it is unclear which context words are the important discriminators (`the` co-occurs with everything).

> [!card]- Give the idf formula as a weighting for context words, explain its terms, and compute it for a word in 10 and a word in 900 out of 1000 contexts.
> $$\text{idf}(w) = \log \frac{N}{n_w}$$
> $N$ = number of contexts, $n_w$ = number of contexts containing $w$. A word in nearly every context gets $\log 1 \approx 0$ and is effectively removed.
> With $N = 1000$: $\log(1000/10) = 4.61$ and $\log(1000/900) = 0.105$ (natural log), so the rare, discriminative word counts about 44 times as much.

> [!card]- Are projections the same as embeddings? Give the criterion for a good word embedding and explain the difference between embeddings as by-products and representation learning.
> A projection is the lookup layer of a neural model such as the probabilistic neural language model (PNLM): it maps each one-hot word to a dense vector $C(w)$ learned with the rest of the network.
> **Good embedding:** $C(w) \approx C(w')$ if and only if $w$ and $w'$ mean the same thing **and** show the same syntactic behaviour.
> - **By-products:** in most models the main objective is a task (next-word prediction, classification); embeddings are learned only because they help.
> - **Representation learning:** learning good embeddings is the main objective. Word2Vec is built this way: the prediction task is a pretext and the network is discarded except for the embedding matrices.

> [!card]- How do prediction-based word embeddings differ from count-based context vectors? Give four properties.
> Same intuition (a word is characterised by its contexts), but embeddings are:
> 1. **low dimensional** ($m \approx 100$ to $1000$, against $\lvert V_c \rvert$),
> 2. **dense** (no zeros),
> 3. **continuous** ($c_w \in \mathbb{R}^m$),
> 4. **learned by performing a prediction task**.
>
> Their dimensions are not interpretable, unlike count vectors.

> [!card]- State the CBOW task, how it relates to n-gram language modelling, and its design choices.
> **Task:** given the $n$ words left and $m$ words right of position $t$, predict the word at $t$ (*the man X the road*).
> **Relation to n-gram LM:** an n-gram LM is the special case with $n$ = LM order $- 1$ and $m = 0$ (left context only). Because CBOW also looks right, it is useless as a language model and only good for learning representations.
> **Design choices** (feed-forward network): focus on learning the embeddings; simpler than the PNLM, with no hidden non-linear layer; the embedding layer is brought closer to the output so its gradient is not diluted; typically $n = m$ with $n \in \{2, 5, 10\}$.

> [!card]- Write the CBOW model with every term and matrix shape, and give its loss and training method.
> $$h = \frac{1}{2n} W w_C, \qquad w_C = \sum_{i=t-n,\, i \neq t}^{t+n} w_i, \qquad \hat{y} = \operatorname{softmax}(W'h)$$
> - $w_i$: one-hot vector (length $\lvert V \rvert$) of the word at position $i$; $w_C$: sum of the $2n$ context one-hots (a count vector).
> - $W$: $\lvert h \rvert \times \lvert V \rvert$ input (projection) matrix; $h$: hidden layer of size $\lvert h \rvert$, the embedding dimension.
> - $W'$: $\lvert V \rvert \times \lvert h \rvert$ output matrix, not necessarily shared ($W' \neq W^\top$).
> - $\hat{y} \in \mathbb{R}^{\lvert V \rvert}$: predicted distribution over the centre word.
>
> No non-linearities, one hidden layer. Loss: cross entropy. Training: SGD.

> [!card]- In CBOW, what does multiplying $W$ by a one-hot vector do, what does the cross-entropy loss reduce to, why is it called "continuous bag of words", and what is the computational bottleneck?
> - $W w_i$ selects column $i$ of $W$, so $h$ is the **average** of the context words' columns: lookups, no real matrix multiplication.
> - With a one-hot target, cross entropy is $-\log \hat{y}[w_t]$, the negative log-probability of the correct word (as in the PNLM).
> - **Bag of words:** the context vectors are summed, so word order is lost. **Continuous:** the bag is a sum of dense vectors.
> - **Bottleneck:** the softmax normalises over all $\lvert V \rvert$ words, so each example touches all of $W'$, cost $O(\lvert V \rvert \cdot \lvert h \rvert)$. Negative sampling removes this.

> [!card]- In Word2Vec, where do the embeddings live, and which matrix is used as "the" embedding?
> - **Column $i$ of $W$** ($\lvert h \rvert \times \lvert V \rvert$): word $i$ as an input (context) word.
> - **Row $i$ of $W'$** ($\lvert V \rvert \times \lvert h \rvert$): word $i$ as an output (predicted) word.
>
> Typically $W$ is used, or $W_s = W^\top + W'$, which combines both into one $\lvert V \rvert \times \lvert h \rvert$ matrix whose row $i$ is word $i$'s embedding (the transpose makes the shapes match).

> [!card]- Write the Skip-gram model, its independence assumption, and explain why its $2n$ output layers produce the same distribution.
> $$h = W w_I, \qquad p(w_{t-n} \ldots w_{t+n} \mid w_I) \propto \prod_{i \neq t} p(w_i \mid w_I), \qquad \hat{y}_i = \operatorname{softmax}(W'h)$$
> $w_I$: one-hot of the centre word; $h$ is simply column $w_I$ of $W$, no averaging. Context words are assumed **independent given the input word**.
> All $2n$ output layers use the same $W'$ and the same $h$, so they produce the **same** $\hat{y}$; only the target each is scored against differs. Loss $= -\sum_{i \neq t} \log \hat{y}[w_i]$ (cross entropy, SGD). Equivalently, $2n$ independent (input word, context word) pairs per position, which is how negative sampling treats it.

> [!card]- Compare CBOW and Skip-gram on input, output, hidden layer and predictions per position.
> - **CBOW:** input $2n$ context words; output the centre word; hidden layer = average of the context embeddings; 1 prediction per position.
> - **Skip-gram:** input 1 centre word; output each of the $2n$ context words; hidden layer = the centre word's embedding; $2n$ predictions per position.

> [!card]- What problem does negative sampling solve, and what task replaces the original prediction task?
> Both CBOW and Skip-gram benefit from large data, but the full softmax needs a score for every word in $V$ for every training example.
> Negative sampling turns it into a **binary classification**: distinguish words that do and do not occur in the context of the input word, using **1 positive** example (the word that actually occurred) and **$k$ negatives** drawn from a noise distribution. Cost per example drops from $\lvert V \rvert$ dot products to $k + 1$. It belongs to the contrastive learning family: pull the true pair together, push sampled pairs apart.

> [!card]- In the negative sampling objective, what are $D$, $D'$, $\theta$, $v_w$ and $v_c$, and what is $p(D = 1 \mid c, w; \theta)$?
> - $D$: observed (word, context) pairs from the data; $D'$: pairs drawn from a noise distribution.
> - $D = 1$: the pair came from the data; $D = 0$: noise.
> - $\theta$: the parameters, the matrices $W$ and $W'$.
> - $v_w = Ww$ is column $w$ of $W$ (input embedding); $v_c = W'^\top c$ is row $c$ of $W'$ (output embedding).
> - $p(D=1 \mid c,w;\theta) = \sigma(v_c \cdot v_w)$ with $\sigma(x) = 1/(1+e^{-x})$, and $p(D=0) = 1 - \sigma(v_c \cdot v_w) = \sigma(-v_c \cdot v_w)$.
>
> The dot product is a compatibility score pushed up for real pairs and down for random ones; words sharing many contexts get pushed towards the same output vectors and so towards each other.

> [!card]- What are the practical Word2Vec settings for negative sampling and context size?
> - Skip-gram: add **5 to 20 negative samples** per observed $(w, c)$.
> - Draw negatives from the unigram distribution scaled to **$P(w)^{0.75}$**, to bias towards rarer words.
> - Context size typically **2 to 5**.
> - The more data, the smaller the context and the negative sample set can be.

> [!card]- Worked example: three words have unigram probabilities 0.9, 0.09 and 0.01. Compute the negative sampling distribution after raising to the power 0.75, and say why this matters.
> - $0.9^{0.75} = 0.924$, $0.09^{0.75} = 0.164$, $0.01^{0.75} = 0.032$; sum $= 1.12$.
> - Renormalised: $0.825$, $0.147$, $0.028$.
>
> The rare word's probability almost triples (1% to 2.8%) and the frequent word's drops (90% to 82.5%). Without it nearly every negative would be `the`, `of` or `and`, and the model would learn little about the rest of the vocabulary.

> [!card]- Give the Skip-gram with negative sampling training loop as steps.
> 1. Initialise $W$ (input, columns $v_w$) and $W'$ (output, rows $v_c$) with small random values; set the noise distribution $P_n(w) \propto P(w)^{0.75}$.
> 2. For each position $t$ with word $w$, and each context position $i \in [t-n, t+n]$, $i \neq t$: take the positive pair $(w, c)$ and draw $k$ negatives $n_1 \ldots n_k$ from $P_n$.
> 3. Positive: error $e = \sigma(v_c \cdot v_w) - 1$; accumulate $e\, v_c$ into the gradient for $v_w$; update $v_c \leftarrow v_c - \eta\, e\, v_w$.
> 4. Each negative: $e = \sigma(v_{n_j} \cdot v_w)$; accumulate $e\, v_{n_j}$; update $v_{n_j} \leftarrow v_{n_j} - \eta\, e\, v_w$.
> 5. Update $v_w \leftarrow v_w - \eta\, g_w$ with the accumulated gradient.
> 6. Return $W$ (or $W^\top + W'$).

> [!card]- Worked example: one positive pair scores $v_c \cdot v_w = 2$, two negatives score $0.5$ and $-1$. Compute the negative sampling loss and the gradient with respect to $v_w$.
> - Positive: $\sigma(2) = 0.881$, loss $-\log 0.881 = 0.127$.
> - Negative 1: $\sigma(-0.5) = 0.378$, loss $0.974$.
> - Negative 2: $\sigma(1) = 0.731$, loss $0.313$.
> - Total $\mathcal{L} = 1.414$.
> - Gradient: $(0.881 - 1)v_c + \sigma(0.5)v_{n_1} + \sigma(-1)v_{n_2} = -0.119\,v_c + 0.622\,v_{n_1} + 0.269\,v_{n_2}$.
>
> The largest correction comes from $n_1$, the negative the model currently mistakes for a real context; the positive is already mostly right.

> [!card]- Worked example: at random initialisation all scores are 0, with one positive and two negatives. What are the negative sampling loss and the gradient with respect to $v_w$?
> $\sigma(0) = 0.5$ for every pair, so each term contributes $-\log 0.5 = \log 2 = 0.693$ and $\mathcal{L} = 3 \times 0.693 = 2.079$.
> Gradient: $(0.5 - 1)v_c + 0.5\,v_{n_1} + 0.5\,v_{n_2} = -0.5\,v_c + 0.5\,v_{n_1} + 0.5\,v_{n_2}$: equal pull towards the true context and push away from each negative.

> [!card]- Contrast count-based and prediction-based learning of word representations, including global matrix factorisation, and say what GloVe takes from each.
> - **Count-based:** directly, or as global matrix factorisation (latent semantic analysis: factorise the co-occurrence matrix with SVD). Uses global co-occurrence information directly, but tends to perform worse.
> - **Prediction-based** (CBOW, Skip-gram): local context windows. Tends to outperform count-based methods, but treats each window as an independent event, repeatedly rediscovering the same association.
> - **GloVe** (Pennington et al., 2014) combines global co-occurrence statistics with local prediction-style updating: count once over the corpus, then train vectors by gradient updates to fit those counts.

> [!card]- Define $X_{ij}$, $X_i$ and $P_{ij}$ in GloVe.
> - $X_{ij}$: number of times word $j$ occurs in the context of word $i$ (the co-occurrence matrix).
> - $X_i = \sum_k X_{ik}$: number of times any word occurs in the context of word $i$.
> - $P_{ij} = P(j \mid i) = X_{ij} / X_i$: probability of word $j$ occurring in the context of word $i$.

> [!card]- Write the GloVe objective and explain every term.
> $$J = \sum_{(i,j)} f(X_{ij})\left(w_i \cdot \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$
> - $w_i$: vector for word $i$ as target; $\tilde{w}_j$: vector for context word $j$.
> - $b_i$, $\tilde{b}_j$: scalar biases.
> - $\log X_{ij}$: log co-occurrence count, the regression target.
> - $f(X_{ij})$: weighting function.
>
> It is weighted least-squares regression: the dot product plus biases should predict the log co-occurrence count, so words with similar co-occurrence rows get similar vectors.

> [!card]- Why does GloVe regress on log counts? Use the ice and steam example.
> If $w_i \cdot \tilde{w}_k \approx \log X_{ik}$, then $(w_i - w_j)\cdot \tilde{w}_k \approx \log \frac{X_{ik}}{X_{jk}}$, a log ratio of co-occurrence probabilities. Ratios are what separate meaning: `ice` and `steam` both co-occur with `water`, but the ratio is large for `solid` and small for `gas`. Vector differences then encode such ratios, which is also why analogy arithmetic works.

> [!card]- Give the GloVe weighting function with its typical parameters, its purpose, and compute $f(10)$ and $f(50)$.
> $$f(x) = \begin{cases} (x / x_\text{max})^\alpha & x < x_\text{max} \\ 1 & \text{otherwise} \end{cases}, \qquad x_\text{max} = 100,\ \alpha = 0.75$$
> Purpose: de-emphasise very rare (noisy) and very frequent (uninformative) co-occurrences. It rises smoothly from zero for rare pairs and caps at 1 so frequent pairs cannot take over.
> $f(10) = 0.1^{0.75} = 0.178$; $f(50) = 0.5^{0.75} = 0.595$; $f(1) = 0.032$; $f(x) = 1$ for $x \geq 100$.

> [!card]- Why is the GloVe objective well defined for word pairs that never co-occur, and why does that make GloVe cheap?
> $\log X_{ij} = -\infty$ when $X_{ij} = 0$, but $f(0) = 0$, so those terms vanish. The sum effectively runs only over the non-zero cells of $X$, and since $X$ is sparse, training is cheap.

> [!card]- Worked example: in GloVe, $w_i \cdot \tilde{w}_j = 2.0$, $b_i = 0.3$, $\tilde{b}_j = 0.2$ and $X_{ij} = 20$. Compute the weight, error and loss term (natural log).
> - Weight: $f(20) = (20/100)^{0.75} = 0.2^{0.75} = 0.299$.
> - Error: $2.0 + 0.3 + 0.2 - \log 20 = 2.5 - 2.996 = -0.496$.
> - Loss term: $0.299 \times 0.496^2 = 0.299 \times 0.246 = 0.074$.
>
> The prediction is too low, so the update increases the dot product and biases for this pair.

> [!card]- Give the GloVe training algorithm as steps, and explain the distance weighting, AdaGrad and the final embedding.
> 1. One pass over the corpus builds $X$: for each centre word $i$ and context word $j$ in the window, $X_{ij}$ += weight, often $1/\text{distance}$ (1 for adjacent words, $1/3$ at distance 3), so $X_{ij}$ need not be an integer.
> 2. Initialise $w_i, \tilde{w}_j, b_i, \tilde{b}_j$ randomly.
> 3. For several epochs, for each pair $(i,j)$ in shuffled order: weight $= f(X_{ij})$, error $= w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}$, loss $=$ weight $\cdot$ error$^2$; update all four by AdaGrad.
> 4. Final embedding $w_i + \tilde{w}_i$.
>
> After step 1 the corpus is never read again (the "global" part). AdaGrad is SGD with per-parameter learning rates that shrink for parameters with large past gradients, useful since frequent words get many updates and rare words few. With a symmetric window $X$ is symmetric, so the two vector sets differ only by random initialisation, and summing averages out that noise.

> [!card]- What is subsampling of frequent words in Word2Vec?
> Randomly discarding occurrences of very frequent words before training, with each occurrence of $w$ dropped with a probability that grows with $w$'s frequency, so that words like `the` do not dominate the training pairs. It is Word2Vec's way of handling frequency imbalance, where GloVe uses $f(X_{ij})$.

> [!card]- Define the Spearman correlation used in word similarity evaluation, and compute it for four pairs with human ranks 1, 2, 3, 4 and model ranks 2, 1, 3, 4.
> Pearson correlation between the ranks of the two lists, so only the ordering of pairs matters and the scale of the cosine is irrelevant. With no ties:
> $$\rho = 1 - \frac{6\sum_k d_k^2}{N(N^2-1)}$$
> with $d_k$ the rank difference for pair $k$ and $N$ the number of pairs.
> Here $d = (-1, 1, 0, 0)$, $\sum d_k^2 = 2$, so $\rho = 1 - \frac{12}{4 \times 15} = 1 - 0.2 = 0.8$.

> [!card]- What do WS-353 and SimLex-999 each annotate, and what do the WS-353 scores for media/radio and bread/butter show?
> **WS-353** annotates **relatedness**; **SimLex-999** annotates **similarity**. Both mix kinds of similarity (synonyms, topical, unrelated).
> WS-353 scores (0 to 10): media/radio 7.42 is higher than television/radio 6.77, and bread/butter gets 6.19 although bread and butter are not the same kind of thing. These are relatedness judgements. `coffee`/`cup` is the same case: highly related, not similar, rewarded by WS-353 and penalised by SimLex-999.

> [!card]- Describe the analogy task: the question form, how the answer is computed, how it is scored, and the two kinds of analogy.
> *a is to b as c is to X* (Paris : France :: Berlin : X). Compute $X = v_b - v_a + v_c$ (here $v_\text{France} - v_\text{Paris} + v_\text{Berlin}$) and return the vocabulary word closest by cosine, excluding the three input words (the vector usually stays closest to one of them). Scored by **accuracy**: is the top answer exactly right. Classic case: $v_\text{king} - v_\text{man} + v_\text{woman} \approx v_\text{queen}$.
> Semantic analogies (capital-country) and **syntactic** ones (*acquired : acquire :: tried : try*).

> [!card]- Worked example: $v_\text{king} = (4,1)$, $v_\text{man} = (2,0)$, $v_\text{woman} = (2,2)$. Compute the analogy vector and decide between candidates queen $= (4, 3.2)$ and prince $= (5, 1)$.
> $X = (4,1) - (2,0) + (2,2) = (4,3)$, $\lVert X \rVert = 5$.
> - $\cos(X, \text{queen}) = \frac{16 + 9.6}{5\sqrt{26.24}} = \frac{25.6}{25.61} = 0.9995$
> - $\cos(X, \text{prince}) = \frac{20 + 3}{5\sqrt{26}} = \frac{23}{25.50} = 0.902$
>
> Answer: queen.

> [!card]- How do PCA and t-SNE differ as ways of visualising embeddings, and why is visualisation hard in the first place?
> Embeddings have many dimensions (128, 512, 1024...), so they must be projected to 2D.
> - **PCA:** linear projection onto the two directions of largest variance.
> - **t-SNE:** non-linear; keeps each point's nearest neighbours near it in 2D and gives up preserving global distances. A full-vocabulary t-SNE map shows local cluster structure but no readable global layout.

> [!card]- What do 2D projections of the neighbourhood of `power` and of male/female word pairs show about embedding spaces?
> - **power:** different senses separate into regions: an electricity cluster (voltage, battery, solar), an energy/systems cluster (motor, fuel, supply), a control/operation cluster, and an abstract ability/authority cluster (strength, authority, ability). Capitalised `Power` sits isolated from `power`: the vocabulary is case-sensitive and the two forms get different vectors.
> - **male/female pairs** (brother/sister, uncle/aunt, king/queen, duke/duchess...): lines joining each pair all point roughly the same way, a consistent gender direction. This is the geometric fact behind $v_\text{king} - v_\text{man} + v_\text{woman} \approx v_\text{queen}$ and the property cross-lingual alignment relies on.

> [!card]- Why should 2D embedding visualisations not be taken at face value?
> - Non-linear projections group points that are close in high-dimensional space, so distances in the plot are not distances in the space.
> - t-SNE hyperparameters have a substantial impact (Wattenberg 2016): the same data can show different cluster sizes, distances, or apparent clusters that are not there.
>
> They complement quantitative and extrinsic evaluation and do not replace it.

> [!card]- What three problems come from giving each word type one opaque vector, as Word2Vec and GloVe do?
> 1. Related forms (`run`, `runs`, `running`, `runner`) get potentially unrelated vectors with no shared structure.
> 2. Morphologically rich languages (Finnish, Turkish, Russian) suffer most: huge word-form families, each form a separate, rarer word with a worse-estimated vector.
> 3. No fallback for out-of-vocabulary words: an unseen word has no vector.

> [!card]- Name the two options for making word embeddings morphologically sensitive.
> 1. **Sub-segment words** (with BPE or a morphological analyser) and learn an embedding for each sub-segment.
> 2. **Sliding character window, fastText** (Facebook AI, Bojanowski et al., 2017): represent a word by all its character n-grams.

> [!card]- Why does fastText wrap words in `<` and `>`, and why is the whole word included in $G_w$?
> **Boundary markers** make prefixes and suffixes distinct n-grams: `<ru` can only be word-initial, `ns>` only word-final, and the trigram `her` inside `<where>` differs from the word `<her>`.
> The **whole word** `<w>` is added so frequent words still get a word-specific vector component on top of their n-grams.

> [!card]- Give the fastText training algorithm as steps.
> 1. Inputs: corpus, window $m$, n-gram range $[n_\text{min}, n_\text{max}]$ (default 3 to 6), dimension $d$, negatives $K$.
> 2. For each vocabulary word, $G_w$ = all character n-grams of `<w>` in that range, plus `<w>`; initialise n-gram vectors $z_g$ and context vectors $u_c$ randomly.
> 3. For each (centre $w$, context $c$) pair: sample $K$ negatives from $P_n$; compose $v_w = \sum_{g \in G_w} z_g$; loss $= -\log\sigma(v_w^\top u_c) - \sum_k \log\sigma(-v_w^\top u_{n_k})$.
> 4. Update every $z_g$, $g \in G_w$, with the same gradient $\partial\,\text{loss}/\partial v_w$; update $u_c$ and every $u_{n_k}$.
> 5. Return the n-gram table $\{z_g\}$. There is no word table: any word's vector is computed on demand by summing its n-grams.

> [!card]- Worked example: list the fastText character n-grams of `runs` for $n \in [3, 6]$, count them, and say what `runs` shares with `running`.
> Wrapped: `<runs>` (6 characters).
> - $n=3$: `<ru`, `run`, `uns`, `ns>`
> - $n=4$: `<run`, `runs`, `uns>`
> - $n=5$: `<runs`, `runs>`
> - $n=6$: `<runs>` (also the whole-word token)
>
> $\lvert G_\text{runs} \rvert = 10$. `<running>` has 23 n-grams (22 of length 3 to 6, plus the 9-character whole word) and shares `<ru`, `<run` and `run` with `<runs>`, so the two vectors share three summands and are pulled together automatically.

> [!card]- Worked example: how many elements does $G_w$ have for the word `cat` with $n \in [3, 6]$?
> Wrapped: `<cat>` (5 characters). $n=3$: `<ca`, `cat`, `at>` (3); $n=4$: `<cat`, `cat>` (2); $n=5$: `<cat>` (1); $n=6$: none. The whole word `<cat>` is already the 5-gram, so $\lvert G_\text{cat} \rvert = 6$.

> [!card]- How does fastText handle an unseen word such as `runnings`, and how does it cope with the huge number of distinct n-grams?
> The OOV word still has n-grams like `<run`, `runn`, `ning`, `ings>`, most of them seen in training, so summing their vectors gives a sensible vector where Word2Vec would have nothing.
> In practice the n-grams are hashed into a fixed number of buckets (about 2 million), with one shared vector per bucket.

> [!card]- In the fastText evaluation of Bojanowski et al. (2017), what are sg, cbow, sisg and sisg-, and what does sisg against sisg- measure?
> - **sg**, **cbow**: the Word2Vec baselines.
> - **sisg** (subword information skip-gram): fastText.
> - **sisg-**: fastText where words absent from the training vocabulary get a null vector instead of an n-gram vector.
>
> sisg against sisg- isolates the value of building OOV vectors from n-grams: up to 6 points of Spearman × 100 (German GUR350 64 to 70, Russian HJ 60 to 66).

> [!card]- What are the cross-lingual embedding goal, its main uses, the three training set-ups, and the hope behind alignment?
> **Goal:** put translation-equivalent words close together, e.g. $E(\text{house}) \approx E(\text{gebouw})$, as monolingual spaces already do for synonyms.
> **Uses:** bilingual lexicon induction (dictionary by nearest-neighbour search) and transferring models across languages.
> **Set-ups:**
> 1. Train each language separately, then align the spaces.
> 2. Train all languages together (benefiting from shared words like names and numbers), then align the regions.
> 3. Train on data aligned at the word or sentence level.
>
> Either way some alignment is needed: a mapping that brings translation equivalents close. **Hope:** large monolingual corpora plus only a small bilingual signal, since monolingual text is plentiful and parallel text is not.

> [!card]- State the shared space hypothesis and what follows if it is true.
> - Languages trained separately still encode **similar relational structure**: king/queen in English relates like rey/reina in Spanish.
> - There is an **approximate isomorphism** between the spaces: the overall shape of the point cloud is similar, while coordinates and orientations differ (random initialisation makes axes meaningless).
> - **If true** (it is a hypothesis), a single geometric transformation, a general linear map, can align the spaces and bring translation pairs close together.

> [!card]- What does the 2D comparison of English and Spanish numbers and animals show?
> Projections of one to five and horse, cow, pig, dog, cat in English and their Spanish translations (uno...cinco, caballo, vaca, cerdo, perro, gato): the Spanish plot is stretched, but the arrangement is the same. `one` lies far right of the other numbers, `four` at the top, `two` at the bottom; `cat` isolated at bottom left, `horse` and `cow` together at top left, `dog` to the right. Two independently trained spaces, one shape.

> [!card]- State the structure-preservation condition for a mapping $m$, apply it to the king/queen analogy, and say why it motivates a linear map.
> $m(a \circ b) = m(a) \circ m(b)$, with $\circ$ vector addition and subtraction. So
> $$m(v_\text{king} - v_\text{man} + v_\text{woman}) \approx m(v_\text{king}) - m(v_\text{man}) + m(v_\text{woman})$$
> and the English gender offset maps to the Spanish one (rey, reina, hombre, mujer may be rotated and scaled differently, with the same relational structure).
> For linear $m(v) = Wv$ this holds exactly, $W(a - b + c) = Wa - Wb + Wc$; the $\approx$ reflects that the spaces are only approximately isomorphic. Strictly, a structure-preserving map is a homomorphism; an isomorphism must also be invertible.

> [!card]- Name the three types of cross-lingual alignment method with the bilingual signal each uses.
> - **Supervised:** a seed bilingual dictionary as anchors, typically thousands of translations; solve for the mapping directly.
> - **Semi-supervised:** a very small seed dictionary (hundreds); bootstrap new translations by statistical confidence, then re-solve.
> - **Unsupervised:** no bilingual signal; uses adversarial training.

> [!card]- Where can seed dictionaries come from, and what is the problem with each source?
> 1. **Human-compiled machine-readable dictionaries.** They list lemmas while embedding vocabularies are full of inflected forms (`hablamos`), contain many senses per entry, cover low-resource languages poorly and lack multi-word expressions.
> 2. **Learned from data:** needs a sentence-level parallel corpus, then word alignment of each pair; alignments are often not one-to-one, so the dictionary is noisy.
> 3. **Words spelled the same in both languages** (names, numbers, borrowings): false friends (Dutch `bad` means bath), only works for languages sharing a script, skewed to names and numbers. Despite this, identical-word dictionaries work surprisingly well.

> [!card]- In a word alignment between "The Secretary of State visits The Netherlands" and "De minister van buitenlandse zaken brengt een bezoek aan Nederland", which alignments are not one-to-one, and why does it matter?
> - `buitenlandse zaken` (foreign affairs): both words align to `State`.
> - `brengt een bezoek aan` ("brings a visit to"): four Dutch words align to `visits`.
> - `Nederland` aligns to the two words `The Netherlands`.
>
> A word aligner over a parallel corpus produces such links and the most frequent link per word gives a dictionary; the many-to-one cases make that dictionary noisy at the word level.

> [!card]- State the general alignment problem with every term, and the three basic questions it raises.
> $$W^* = \arg\min_W \sum_i \lVert Wx_i - y_i \rVert^2$$
> $(x_i, y_i)$: paired vectors, a source embedding $x_i \in \mathbb{R}^{d_1}$ and the target embedding $y_i \in \mathbb{R}^{d_2}$ of its translation; $W$: a $d_2 \times d_1$ matrix applied to every source vector; Euclidean norm.
> Questions: what should $W$ be (linear)? what constraints should it satisfy (orthogonality)? where do the pairs come from (a seed dictionary, or nowhere)?

> [!card]- What does "Procrustes" refer to in cross-lingual mapping, and which method belongs to which paper?
> Without a constraint on $W$, $\min_W \sum \lVert Wx_i - y_i \rVert^2$ is ordinary multivariate least squares: Mikolov et al. (2013), the "translation matrix". The **Procrustes problem** proper, and the "Procrustes" rows of the Conneau et al. (2018) results, is the **orthogonally constrained** version solved by SVD: Xing et al. (2015) onwards. The term is sometimes applied loosely to the unconstrained version.

> [!card]- What is a seed dictionary in practice, what are its limits, and what is the real goal of learning a mapping from it?
> A list of (source word, target word) pairs, often the top few thousand most frequent words translated with an existing dictionary; automatic translation can be used with caution (errors become wrong anchors). Limited to single words, since multi-word expressions have no embeddings. Each pair is one anchor point, e.g. (cat, gato).
> The goal is a mapping that **generalises beyond the anchors**: 5000 pairs are useless on their own, while a mapping learned from them that translates the other 195 000 words is the point.

> [!card]- State the linear transformation hypothesis and justify why the map should be linear.
> **Strong assumption:** a single linear transformation $W$ suffices to map one language's space onto another's.
> **Why linear:** king/queen and man/woman analogies are already linear regularities within one space, and the assumption is that the same regularities hold across spaces; a linear map preserves them exactly. It also reduces alignment to classic regression: find the matrix that best maps one paired point set onto the other.

> [!card]- Give the least-squares mapping algorithm (Mikolov et al., 2013) as steps, its gradient, and how a word is translated at inference.
> 1. For each dictionary pair, $x_i$ = source embedding, $y_i$ = target embedding; assemble $X$ ($d_1 \times n$) and $Y$ ($d_2 \times n$).
> 2. Initialise $W$ randomly.
> 3. Until convergence, per mini-batch: loss $= \sum \lVert Wx_i - y_i \rVert^2$, $W \leftarrow W - \eta\, \partial\text{loss}/\partial W$, with gradient $2\sum_i (Wx_i - y_i)x_i^\top$.
> 4. Return $W$.
>
> Inference: compute $Wx$ and return the target word whose embedding is nearest by cosine.

> [!card]- Worked example: in one dimension, the dictionary pairs are $(x, y) = (1, 2)$ and $(2, 5)$. Find the least-squares mapping $W$ and its loss.
> In 1D, $W^* = YX^\top(XX^\top)^{-1} = \frac{\sum_i y_i x_i}{\sum_i x_i^2} = \frac{2 + 10}{1 + 4} = 2.4$.
> Residuals: $2.4 - 2 = 0.4$ and $4.8 - 5 = -0.2$; loss $= 0.16 + 0.04 = 0.20$.
> Check: gradient $2(0.4 \cdot 1 + (-0.2) \cdot 2) = 0$.

> [!card]- In Mikolov-style word translation results (En to Sp, Sp to En, En to Cz, Cz to En), how does the translation matrix compare with edit-distance and co-occurrence baselines, and what does combining it with edit distance do?
> - Edit distance translates to the most similarly spelled word (only works via cognates); word co-occurrence uses count vectors; the translation matrix is the learned linear $W$.
> - En to Sp P@1: translation matrix 33% against 13% (edit distance) and 19% (co-occurrence).
> - Adding edit distance to the matrix gains about 10 points for Spanish (33 to 43) but only 2 for Czech (27 to 29), since Spanish shares far more cognates with English.
> - Czech, the more distant language, is harder for every method.

> [!card]- How does linear-mapping translation accuracy depend on monolingual data size and on word frequency?
> - **Data size:** P@1 rises from about 9 at $2 \cdot 10^7$ training words to about 53 at $2.5 \cdot 10^{10}$ (P@5 from 19 to 75), roughly linear in the log of data size and flattening beyond about $2 \cdot 10^9$. Better monolingual embeddings make the mapping work better.
> - **Frequency:** accuracy drops as test words get rarer, from about 53 in the 5–7K frequency-rank bin to about 40 in the 17–19K bin, because rarer words have worse embeddings and less reliable mapped positions.

> [!card]- What do the errors of a linear Spanish-to-English mapping (imperio, millas, hablamos, protegida, determinante) reveal?
> - `imperio` gives dictatorship, imperialism, tyranny: topically related, none is `empire`.
> - `determinante` gives crucial, key, important: the running-text sense, against the dictionary's `determinant`.
> - `millas` ranks kilometers ahead of miles: the units occur in identical contexts.
> - `protegida` ranks wetland first, presumably from contexts like *zona protegida*.
> - `hablamos` (we talk) gives talking, talked, talk: morphology does not line up one-to-one, so the dictionary's `talk` is only rank 3.
>
> Embedding nearness captures relatedness, while a dictionary demands exact equivalence.

> [!card]- Give the four limitations of an unconstrained linear mapping $W$.
> 1. It can stretch, shear and rotate the space, with nothing preserving the source language's internal geometry (distances, angles).
> 2. This allows overfitting to the seed dictionary: words near the seed pairs map well, the majority outside it do not.
> 3. Training minimises Euclidean distance $\lVert Wx_i - y_i \rVert^2$ while inference retrieves by cosine; the two are not equivalent unless vectors are unit-normalised.
> 4. So additional constraints on the form of $W$ are needed.

> [!card]- State the orthogonal mapping objective of Xing et al. (2015), its normalisation, and what an orthogonal $W$ can and cannot do.
> $$W^* = \arg\min_W \sum_i \lVert Wx_i - y_i \rVert^2 \quad \text{s.t.} \quad W^\top W = I$$
> with all embeddings unit length, $\lVert x \rVert^2 = 1$. An orthogonal $W$ allows only rotations and reflections, no scaling or shearing, so it preserves angles and distances: $\lVert Wx \rVert^2 = x^\top W^\top W x = \lVert x \rVert^2$. It can be solved exactly with SVD. After normalisation all points lie on the unit sphere and the map is literally a rotation (or reflection) of that sphere.

> [!card]- Derive that for unit vectors and orthogonal $W$, minimising Euclidean distance equals maximising cosine, and compute $\lVert Wx - y \rVert^2$ when $\cos(Wx, y) = 0.8$.
> $$\lVert Wx - y \rVert^2 = \lVert Wx \rVert^2 + \lVert y \rVert^2 - 2(Wx)\cdot y = 1 + 1 - 2\cos(Wx, y) = 2 - 2\cos(Wx,y)$$
> using $\lVert Wx \rVert = \lVert x \rVert = 1$ and $(Wx)\cdot y = \cos(Wx, y)$ for unit vectors. Distance is a decreasing function of cosine.
> With $\cos = 0.8$: $2 - 1.6 = 0.4$. (Cosine 0 gives 2; cosine $-1$ gives 4.)

> [!card]- Give the closed-form solution of orthogonal Procrustes and derive it.
> $U\Sigma V^\top = \operatorname{SVD}(YX^\top)$, $W^* = UV^\top$, with $X$, $Y$ ($d \times n$) holding the pairs as columns.
> 1. $\lVert WX - Y \rVert_F^2 = \lVert WX \rVert_F^2 + \lVert Y \rVert_F^2 - 2\operatorname{tr}(W^\top YX^\top)$.
> 2. For orthogonal $W$, $\lVert WX \rVert_F^2 = \lVert X \rVert_F^2$, so minimising means maximising $\operatorname{tr}(W^\top U\Sigma V^\top) = \operatorname{tr}(V^\top W^\top U\, \Sigma)$.
> 3. $Z = V^\top W^\top U$ is orthogonal, so $Z_{kk} \leq 1$ and $\operatorname{tr}(Z\Sigma) = \sum_k Z_{kk}\sigma_k \leq \sum_k \sigma_k$, with equality at $Z = I$.
> 4. Hence $W^\top = VU^\top$, so $W = UV^\top$.
>
> One SVD of a $d \times d$ matrix, no iteration, no learning rate.

> [!card]- Worked example: two dictionary pairs with $x_1 = (1,0)$, $x_2 = (0,1)$, $y_1 = (0,1)$, $y_2 = (-1,0)$. What does orthogonal Procrustes return?
> $X = I$, so $YX^\top = Y = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$, which is already orthogonal: an SVD is $U = Y$, $\Sigma = I$, $V = I$.
> $W = UV^\top = Y$, the 90° rotation. Check: $Wx_1 = (0,1) = y_1$, $Wx_2 = (-1,0) = y_2$, loss 0, and $W^\top W = I$.

> [!card]- Compare unconstrained (Mikolov et al., 2013) and orthogonal (Xing et al., 2015) en-es mappings across embedding dimensions.
> P@1: at 300/300 dimensions 30.43% against 38.99%; 500/500 25.76% against 39.91%; 700/700 20.69% against 41.04%; 800 (EN)/200 (ES) 35.36% against 40.06%.
> - The unconstrained map gets worse as dimension grows ($d^2$ free parameters for the same dictionary, more room to overfit); its best setting is the asymmetric 800/200.
> - The orthogonal map is better everywhere and improves slightly with dimension.

> [!card]- Why can the orthogonality constraint $W^\top W = I$ not hold as written when English has 800 dimensions and Spanish 200?
> $W$ is then non-square ($200 \times 800$ for English to Spanish), with rank at most 200, so $W^\top W$ ($800 \times 800$) cannot be the identity. The constraint can only hold as $WW^\top = I$ (orthonormal rows) or with roles reversed. How Xing et al. handled this is not settled in the course material; the constraint as written applies cleanly only to square cases.

> [!card]- Worked example: translating `cat` with $r_T(x) = 0.40$. Candidate A has $\cos = 0.70$, $r_S = 0.75$; candidate B has $\cos = 0.62$, $r_S = 0.45$. Which wins under nearest neighbour and which under CSLS?
> - Nearest neighbour: A (0.70 > 0.62).
> - CSLS: A $= 1.40 - 0.40 - 0.75 = 0.25$; B $= 1.24 - 0.40 - 0.45 = 0.39$. **B wins.**
>
> A is a hub: close to many mapped source words on average, so its high $r_S$ cancels its slightly higher cosine.

> [!card]- Give the definition of $r_T(Wx_s)$ in CSLS and say what $r_S(y)$ is.
> $$r_T(Wx_s) = \frac{1}{K}\sum_{y_t \in \mathcal{N}_T(Wx_s)} \cos(Wx_s, y_t)$$
> the mean cosine between the mapped source vector and its $K$ nearest target vectors $\mathcal{N}_T(Wx_s)$. $r_S(y)$ is the mean cosine between target $y$ and its $K$ nearest neighbours in the mapped source space. The translation of $x_s$ is the $y$ maximising $\text{CSLS}(Wx_s, y)$.

> [!card]- With supervised Procrustes on fastText embeddings, how do NN, ISF and CSLS retrieval compare, and how does accuracy vary by language (Conneau et al., 2018)?
> - **NN:** plain cosine nearest neighbour. **ISF:** inverted softmax, another hubness correction. **CSLS** as defined.
> - CSLS beats NN on every pair: en-es 77.4 to 81.4 (+4.0), fr-en 76.1 to 82.4 (+6.3), en-de 68.4 to 73.5 (+5.1), ru-en 58.2 to 63.7 (+5.5).
> - Accuracy follows language distance from English: Spanish and French around 80, German low 70s, Russian 47 to 64, Chinese 30 to 43, Esperanto 20 to 29.

> [!card]- Why is unsupervised cross-lingual alignment wanted, and what is the chicken-and-egg problem?
> Supervised methods need thousands of translation pairs: for many low-resource pairs even a small dictionary may not exist, and dictionaries may not cover inflected forms, which are the forms most often seen in data. So: align two point clouds, hypothesised to be approximately isomorphic, from geometry alone.
> Chicken and egg: a mapping is needed to find translation pairs, and translation pairs to learn the mapping. Conneau et al. (2018) break it with adversarial training, which bootstraps a rough alignment from distributional structure, then feeds translation pairs to the supervised machinery.

> [!card]- In adversarial cross-lingual alignment, what are the generator, the discriminator, the real and fake data, and why would fooling the discriminator align translations?
> - **Generator:** the mapping $W$, sending source embeddings into the target space; aim: make mapped vectors indistinguishable from real target vectors.
> - **Discriminator:** a classifier trained simultaneously to tell mapped source vectors from real target vectors.
> - **Real data:** target embeddings $y_1 \ldots y_n$; **fake data:** mapped source vectors $Wx_1 \ldots Wx_m$.
>
> If the discriminator cannot tell them apart, the mapped cloud has the target cloud's shape and position; under the shared space hypothesis, the only rotation achieving this puts translations on top of each other.

> [!card]- Give the discriminator architecture and sampling choices in MUSE, with the reason for each.
> - MLP with **2 hidden layers of 2048 units**, **ReLU**, **dropout on the input layer**.
> - Two-class output with **label smoothing $s = 0.2$**: targets $\{s, 1-s\}$ instead of $\{0, 1\}$, preventing pathological overconfidence (a discriminator outputting exactly 0 or 1 gives the generator vanishing gradients).
> - Source and target vectors drawn from the **50k most frequent words** of each language: rare words have poorly estimated embeddings, so frequent words give a cleaner signal.

> [!card]- Write the discriminator and generator losses of adversarial alignment and explain each.
> $P_{\theta_D}(\text{source}=1 \mid v)$: probability that $v$ is a mapped source (fake) vector.
> $$\mathcal{L}_D = -\frac{1}{n}\sum_i \log P_{\theta_D}(\text{source}=1 \mid Wx_i) - \frac{1}{m}\sum_j \log P_{\theta_D}(\text{source}=0 \mid y_j)$$
> $$\mathcal{L}_W = -\frac{1}{n}\sum_i \log P_{\theta_D}(\text{source}=0 \mid Wx_i) - \frac{1}{m}\sum_j \log P_{\theta_D}(\text{source}=1 \mid y_j)$$
> - $\mathcal{L}_D$: binary cross entropy with true labels, updated with $W$ frozen.
> - $\mathcal{L}_W$: same cross entropy with labels flipped, updated with $\theta_D$ frozen; the $y_j$ term does not depend on $W$ and gives no gradient.
> - Gradient for $W$: $\sum_i \delta_i x_i^\top$, with $\delta_i = \partial\mathcal{L}_W / \partial(Wx_i)$, backpropagated through the frozen discriminator.
> - Initialise $W$ as the identity (or a random orthogonal matrix).

> [!card]- Give the re-orthogonalisation update in MUSE, show that orthogonal matrices are fixed points, and compute its effect on singular values 1.2 and 0.8.
> $$W \leftarrow (1+\beta)W - \beta(WW^\top)W, \qquad \beta = 0.01$$
> If $W$ is orthogonal, $WW^\top = I$ and the update gives $(1+\beta)W - \beta W = W$.
> Per singular value: $s \leftarrow (1+\beta)s - \beta s^3$, pushing $s$ towards 1.
> - $s = 1.2$: $1.212 - 0.01728 = 1.195$
> - $s = 0.8$: $0.808 - 0.00512 = 0.803$
> - $s = 1$: unchanged.
>
> Without it, gradient steps would let $W$ drift into a general linear map with the warping and overfitting problems of unconstrained mappings.

> [!card]- Why is model selection hard in unsupervised alignment, and what criterion does MUSE use instead?
> The adversarial loss is not a reliable indicator: the discriminator loss can look fine while the mapping is poor, and there is no validation dictionary.
> Criterion:
> 1. Take the 10k most frequent source words.
> 2. With the current $W$, find each word's CSLS nearest neighbour in the target space.
> 3. Average the cosine similarities of these induced pairs; keep the checkpoint with the highest value.
>
> Rationale: a poor mapping puts mapped words far from any real target word, so even best matches have low cosine. It is a proxy: high cosine does not prove the pairs are translations.

> [!card]- Give the MUSE refinement (self-learning) loop as steps, and say why it counts as semi-supervised.
> 1. Start from the rough $W$ from adversarial training (global orientation roughly right, too imprecise for good retrieval).
> 2. For each frequent source word $s$, find $t = \arg\max_t \text{CSLS}(Wx_s, y_t)$; keep $(s,t)$ only if $s$ is also the best source for $t$ (mutual nearest neighbours), and only high-confidence pairs.
> 3. Re-solve exactly: $U\Sigma V^\top = \operatorname{SVD}(Y_D X_D^\top)$, $W = UV^\top$.
> 4. Repeat 2 and 3.
>
> Steps 2 to 4 are the supervised Procrustes method run on a dictionary the system built itself; starting from a small human seed dictionary instead of adversarial training gives the semi-supervised method. Mutuality filters out hub matches.

> [!card]- Who criticised the unsupervised alignment results of Conneau et al. (2018), and along which five lines?
> **Søgaard, Ruder and Vulić (2018)**: the choice of languages, the role of morphology, data size, domain, and the embedding algorithm.

> [!card]- How did Søgaard et al. (2018) change the language sample relative to Conneau et al. (2018)?
> Conneau et al.'s languages (EN, FR, DE, ZH, RU, ES) are mostly dependent-marking, have few or no cases (German 4, Russian 6–7, the rest none) and none is agglutinative. Søgaard et al. added Estonian and Finnish (mixed marking, agglutinative, 10+ cases), Greek (double marking, fusional, 3 cases), Hungarian (dependent, agglutinative, 10+), Polish (dependent, fusional, 6–7) and Turkish (dependent, agglutinative, 6–7).

> [!card]- Define head-marking, dependent-marking, double-marking and zero-marking, and identify head and dependents in a possessive phrase and a clause.
> Which word in a grammatical relationship carries the marker?
> - **Dependent-marking:** the dependent. **Head-marking:** the head. **Double-marking:** both. **Zero-marking:** neither (word order alone).
> - Possessive phrase: head = possessed noun, dependent = possessor.
> - Clause: head = verb, dependents = subject and object.

> [!card]- Explain the marking in German "das Haus des Mannes" and "der Mann sieht den Hund", and contrast English.
> - *das Haus des Mannes*: head `Haus`, dependent `Mann`; genitive marking on the possessor *des Mannes* (dependent-marking).
> - *der Mann sieht den Hund*: head `sieht`, dependents `Mann`, `Hund`; nominative and accusative case on subject and object (dependent-marking). The verb also agrees with the subject (third person singular), so there is some head-marking, but dependent marking predominates.
> - English marks the same relations mostly by word order (*the man sees the dog* against *the dog sees the man*) and a few prepositions (*of the man*).

> [!card]- What did Søgaard et al. (2018) find when comparing adversarial alignment with identical-word supervision (P@1)?
> - Mixed or double marking pairs fail completely with adversarial: EN-ET 0.00, EN-FI 0.09, EN-EL 0.07; identical-word supervision gets 31.45, 28.01, 42.96.
> - Identical beats adversarial on every pair with English: EN-ES 82.62 against 81.89, EN-HU 46.56 against 45.06, EN-PL 52.63 against 46.83, EN-TR 39.22 against 32.71.
> - ET-FI, two structurally similar languages, works adversarially (29.62, against 24.35 identical): the one case where adversarial wins.
> - Identical-word supervision needs no human dictionary, so it is a fair free baseline.

> [!card]- French is listed as mixed-marking, yet English-French unsupervised alignment works well. What does that say about the explanation for the failures?
> Marking type is not the whole story. The failing languages are mixed or double marking **and** case-rich (Estonian, Finnish: 10+ cases, agglutinative; Greek: 3 cases, fusional). The real factor is how many distinct surface forms a lemma has, to which marking type contributes. Hungarian, dependent-marking but agglutinative with 10+ cases, sits in between (45.06).

> [!card]- According to Søgaard et al. (2018), why does unsupervised alignment fail for mixed- and double-marking languages?
> - Grammatical information is distributed across more word forms.
> - A lemma is realised in many more distinct surface forms, each with its own embedding and each of lower frequency.
> - This makes the mapping between a language that marks relations by word order and one that marks them by morphology more complex: non-isomorphic.
>
> English `house` corresponds to Finnish `talo`, `talon`, `taloa`, `talossa`, `talosta`, `taloon`, `talolla`...: no rotation maps one point onto a dozen.

> [!card]- Give the eigenvector similarity algorithm as steps, and the correct form of the Laplacian.
> 1. For each language, compute each word's nearest neighbours and record them in an adjacency matrix $A$ (undirected nearest-neighbour graph).
> 2. Degree matrix $D$: diagonal, each entry the number of neighbours of that node.
> 3. Laplacian $L = D - A$ (eigenvalues all $\geq 0$). Writing $L = A - D$ flips every eigenvalue's sign, so "keep the largest" would pick the values nearest zero.
> 4. Keep the largest $n\%$ of the eigenvalues of $L$.
> 5. $\Delta = \sum_i (\lambda_i^{(1)} - \lambda_i^{(2)})^2$ over the kept eigenvalues.

> [!card]- Worked example: compute $\Delta$ between a 4-node star and a 4-node path, keeping all Laplacian eigenvalues.
> - Star (one hub, three leaves): eigenvalues $4, 1, 1, 0$.
> - Path: $2+\sqrt{2} = 3.414$, $2$, $2-\sqrt{2} = 0.586$, $0$.
>
> $$\Delta = (4 - 3.414)^2 + (1-2)^2 + (1 - 0.586)^2 + 0 = 0.343 + 1 + 0.171 = 1.515$$
> Two stars with nodes numbered in any order give $\Delta = 0$. The star's top eigenvalue (4) is its hub, one densely connected node, like tight clusters of Finnish inflected forms against a flatter English graph.

> [!card]- Worked example: compute $\Delta$ between a triangle (3 nodes, all connected) and a 3-node path, keeping all Laplacian eigenvalues.
> - Triangle: $L = D - A$ has eigenvalues $3, 3, 0$.
> - Path: eigenvalues $3, 1, 0$.
>
> $\Delta = (3-3)^2 + (3-1)^2 + (0-0)^2 = 4$. The triangle is more densely connected, which shows up as a larger second eigenvalue.

> [!card]- What does a high eigenvector-similarity $\Delta$ between two languages mean, and why can no rotation fix it?
> - Words in language A cluster in a way structurally unlike language B.
> - The isomorphism assumption between A and B does not hold.
> - A rotation preserves the nearest-neighbour graph exactly, so it cannot change the Laplacian eigenvalues.

> [!card]- Is the failure of English-Finnish unsupervised alignment a data size problem? Give the evidence.
> No. Finnish Wikipedia is 12M words against 363M for Spanish, but retraining on the Finnish WaC corpus (1.7 billion words) left English-Finnish P@1 at 0.0. Meanwhile Estonian-Finnish, both mixed-marking agglutinative languages, reaches 29.62. Pairing structurally dissimilar languages is the problem.

> [!card]- How does unsupervised alignment accuracy differ by part of speech for en-es, en-hu and en-fi, and why?
> - **en-hu:** nouns 26.87 and verbs 25.44 are worst; adjectives 53.28, adverbs 51.57, other 53.40 are about twice as good. Nouns (case) and verbs (person, number, tense) inflect most in Hungarian: the form explosion per lemma shows up per POS.
> - **en-fi:** 0.00 for every POS.
> - **en-es:** verbs weakest (66.05 against nouns 80.94, adjectives 85.53), since Spanish verbs are its most heavily inflected words.

> [!card]- How sensitive is bilingual lexicon induction to domain, for identical-word supervision and for adversarial alignment (en-es, EuroParl, Wikipedia, EMEA)?
> Monolingual corpora of 1.1M sentences each from EuroParl (political), Wikipedia (general) and EMEA (medical).
> - **Same domain both sides:** both work, 41 to 64 P@1 (identical: EP 64.09, Wiki 46.52, EMEA 49.24; adversarial: 61.01, 41.38, 49.43).
> - **Different domains:** identical-word supervision degrades but survives (25.17 and 25.48 between EP and Wiki; 4.84 to 9.63 involving medical). Adversarial collapses to 0.0 to 0.13 in every cross-domain cell.
>
> This matters because a low-resource language rarely lets you pick a monolingual corpus matching the English domain, and a domain mismatch alone breaks even English-Spanish.

> [!card]- How sensitive is unsupervised alignment to the embedding algorithm and its hyperparameters?
> English fixed at fastText skipgram, window 2, n-grams 3–6; Spanish varied.
> - **Variations within skipgram** (window 10, n-grams 2–7, or both): 81.89 down to 80.15 at worst, a loss of at most 1.74 points.
> - **Spanish CBOW against English skipgram:** 0.00 to 0.13, even with identical hyperparameters.
>
> The two algorithms produce spaces with different geometry on the same kind of data, so the approximate isomorphism unsupervised alignment depends on is partly an artefact of using the same algorithm on both sides.

> [!card]- What does "static" mean for Word2Vec, GloVe and fastText embeddings, and what is the standard example of its limitation?
> One vector per word type, whatever the sentence. `bank` has the same vector in *river bank* and *bank account*. Contextual embedding models (an LSTM or, in most current models, a Transformer encoder such as BERT) remove this restriction.
