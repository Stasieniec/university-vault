---
type: lecture
course: 5354PHSC6Y
week: 4
lecture: 4
date: 2026-09-22
lecturer: Enrico Cinti
status: complete
topics:
  - Hempel's deductive-nomological model
  - Inductive-statistical explanation
  - Asymmetry, irrelevance and correlation objections
  - Causal-mechanical explanation (Salmon)
  - Unification (Friedman, Kitcher)
  - Van Fraassen's pragmatic account of explanation
  - Godfrey-Smith's contextualism about explanation
  - Eliminativism and reductivism about understanding
  - Pragmatic theories of understanding
  - De Regt and Dieks, CUP and CIT
  - Intelligibility and visualizability (Anschaulichkeit)
---

# PhilSci-L04: Scientific Explanation and Scientific Understanding

> [!abstract] In short
> What is it for science to tell us *why* something happens?
>
> **Explanation.** Hempel: to explain something is to deduce it from laws of nature plus the circumstances, showing it was to be expected. That fails, most clearly because you can deduce a flagpole's height from its shadow, and the shadow does not explain the pole. The replacements say explanation shows causes (Salmon) or fits things into a broader pattern (Kitcher). Godfrey-Smith says each field sets its own standards; van Fraassen says explanation is only a use of science.
>
> **Understanding.** De Regt and Dieks: understanding is more than having an explanation. It is a skill, the ability to use a theory, and a theory is intelligible when scientists can see what it implies without calculating. What counts as intelligible has changed through history.
>
> Mock exam: Q7 (Hempel plus one objection). Van Fraassen also feeds Q8 (week 5).

## Why explanation was a problem for empiricists

Explanation matters for two reasons. We expect science to tell us why things happen. And explanation is tied to evidence: scientists prefer theories that explain puzzling phenomena well, and much scientific reasoning is **explanatory inference** (from data to a hypothesis about a hidden structure or process that would explain the data). So an account of explanation feeds into an account of evidence.

It is problematic because empiricists distrusted it. "Explaining" sounds like claiming insight into hidden causes and essences behind the appearances: **metaphysics** (claims about the ultimate nature of reality that no observation can settle), which the logical empiricists of [[PhilSci-L01 - Introduction and Logical Empiricism|L01]] wanted science to do without.

- **Pierre Duhem** (French physicist and philosopher, see [[PhilSci-L03 - Under-determination|L03]]): explaining "gives hostages to fortune", making physics depend on metaphysical claims like "atoms exist" or "there is an aether", which science can neither confirm nor refute. So explaining is not physics's job.
- **Rudolf Carnap** rejected metaphysical why-questions, of the kind he found in Hegel, as pseudo-explanations. But he admitted a respectable **empiricist explanation**, closely linked with **prediction from laws**.

In Godfrey-Smith's words, the positivists "made peace with the idea that science explains" by construing it "in a low-key way that fitted into their empiricist picture". The result, in 1948, was Carl Hempel and Paul Oppenheim's **covering law model**, a "philosophically light" account: nothing about hidden causes, only how statements relate logically.

## Hempel: to explain is to show it was to be expected

Hempel's idea: to explain something is to show that, given the laws of nature and the circumstances, it was **to be expected**. You write an argument whose premises are laws and particular facts, and the thing to be explained follows as the conclusion.

> [!example] The sticking front door
> Lately my front door is hard to open and close. Why?
> $$\begin{array}{ll} C_1: & \text{My front door is made of wood.} \\ L_1: & \text{Wooden objects expand if humidity increases.} \\ C_2: & \text{Humidity has increased this autumn.} \\ \hline E: & \text{My front door has expanded this autumn.} \end{array}$$

If we ask "why $X$?", $X$ is the **explanandum** (what is explained). If we answer "because $Y$", $Y$ is the **explanans** (what does the explaining).

> [!definition] Deductive-nomological (D-N) explanation
> An **explanation** is a deductive argument in which the **explanandum** (the phenomenon or law to be explained) is deduced from premises containing **general laws** and **particular facts and initial/boundary conditions** (the **explanans**).
> $$\begin{array}{lll} \textit{Explanans:} & C_1, \dots, C_k & \text{particular facts / conditions} \\ & L_1, \dots, L_r & \text{general laws} \\ \hline \textit{Explanandum:} & E & \text{phenomenon} \end{array}$$
> **Conditions of adequacy:**
> 1. The explanandum must be a **logical consequence** of the explanans.
> 2. The explanans must contain a **general law**, in an **essential** way (remove it and the deduction fails).
> 3. The explanans must have **empirical content**: testable, at least in principle, by experiment or observation.
> 4. The explanans must be **true**.

**Deductive** because the conclusion follows logically; **nomological** because it uses laws (Greek *nomos*, law). It is the deductive version of the **covering law model**: the phenomenon is "covered" by a law. The **inductive-statistical (I-S)** version handles **probabilistic** laws: the explanans shows the explanandum was **highly probable**.

**Explanation and prediction are symmetric**: for Hempel they have the same logical structure. Had I known in summer that the door is wooden and that humidity would rise, the same argument would have *predicted* the sticking door. The only difference is whether you already know the conclusion.

The model covers **regularities** as well as single events: from the periodic table and the laws of electrons in atoms you can deduce that noble gases barely react and metals conduct electricity.

Three clarifications:

- **Laws need not be causal.** Functional laws count: $PV = nRT$ explains without saying whether pressure causes volume or the reverse.
- **The model is normative.** It states the ideal of a good scientific explanation, like the logician's concept of **proof**, which stays useful though mathematicians rarely write fully formal proofs.
- **Elliptical explanations.** Scientists often leave laws or conditions unstated; Hempel treats these as abbreviations of a full D-N argument.

## What goes wrong: a good deduction need not explain

Each objection is an argument that meets all of Hempel's conditions and still does not explain. So the conditions are not sufficient.

### Asymmetry: the flagpole and its shadow

Why is the flagpole's shadow 12 metres long? From the pole's height, the sun's elevation, the laws of optics and some trigonometry, the length follows. A perfect D-N explanation.

But the deduction runs just as well backwards: from the shadow's length, the sun's position and the same laws, you can deduce the pole's height.

```
  Laws of optics              Laws of optics
  Laws of geometry            Laws of geometry
  Position of the sun         Position of the sun
  Length of the flagpole      Length of the shadow
  ----------------------      ----------------------
  Length of the shadow        Length of the flagpole

  explains                    predicts, does not explain
```

Both arguments satisfy all four conditions. Only the left one explains: the pole explains the shadow, and the shadow does not explain the pole. We can swap them **to predict, but not (always) to explain**. Explanation has a direction that deduction lacks, which is exactly where Hempel's symmetry of explanation and prediction breaks. Godfrey-Smith calls this "something close to a knockdown argument" and "the killer".

The same goes for symptoms. If only disease $D$ produces symptom $S$, you can infer $D$ from $S$ with a law, but a symptom never explains its disease.

Hempel **bit the bullet**: if his theory lets an explanation run both ways, both directions must be fine. That is defensible in some physics cases where the direction is genuinely unclear, and hopeless for the flagpole.

### Irrelevance: hexed salt

> [!example] Hexed salt (Salmon, Achinstein)
> $$\begin{array}{ll} L: & \text{Every sample of salt hexed by a magician dissolves in water.} \\ C_1: & \text{This sample was hexed by a magician.} \\ C_2: & \text{This sample was placed in water.} \\ \hline E: & \text{This sample dissolved.} \end{array}$$

The law is true (all salt dissolves, hexed or not), and the deduction is valid. But the hex explains nothing. An argument can meet the D-N conditions while containing **irrelevant information that robs it of explanatory power**.

### Correlation: the barometer

> [!example] The barometer
> $$\begin{array}{ll} L: & \text{If the barometer drops, a storm will occur.} \\ C: & \text{The barometer drops.} \\ \hline E: & \text{A storm will occur.} \end{array}$$

An excellent prediction, and no explanation: the barometer does not cause the storm. Both are effects of a **common cause**, the drop in atmospheric pressure. **Correlation is not explanation**: showing that something was *to be expected* is not showing *why* it happened. The mock key calls this "apparently missing causation (correlation or expectation $\neq$ explanation)".

### Two more problems

- **Explanations do not always involve laws.** "The window broke because the ball hit it" explains by citing a causally relevant fact, with no law. So Hempel's conditions are not necessary either.
- **High probability is not needed** (against the I-S version). Syphilis explains some of its symptoms though only some patients develop them.

> [!summary] What the counterexamples share
> **Causation** is missing. The pole causes the shadow, not the reverse. The hex causes nothing. The barometer and the storm share a cause.

## First replacement: to explain is to show the causes

> [!definition] Causal theory of explanation (Wesley Salmon, 1984)
> **Explanation = uncovering the causes of phenomena.** "Causal processes, causal interactions, and causal laws provide the **mechanisms** by which the world works; to understand why certain things happen, we need to see how they are produced by these mechanisms."

On Salmon's **causal-mechanical** version, an explanation situates the explanandum in a **nexus of causal relations**, the causal structure of the world. "The water evaporated because heat was applied to it, and the van der Waals bonds between the molecules were broken." Answering a why-question with "because..." means citing causes.

This solves the flagpole at once: sunlight hitting the pole causes the shadow, so explanation runs from pole to shadow.

The price is an account of **what causation is**, which empiricists since David Hume (18th-century Scottish empiricist) have found suspect: a hidden, unobservable connection. Three families of answer:

- **Regularity (Hume):** causation is **constant conjunction**, $C$ regularly followed by $E$. That produces an **expectation** in us, not a necessary connection in nature. It brings the barometer back: barometer drops are constantly conjoined with storms.
- **Counterfactual (Hume, Woodward):** **if the cause $C$ had not occurred, the effect $E$ would not have occurred**.
- **Process (Salmon):** we can identify actual **causal processes and interactions** in nature, like a chain of falling dominoes.

Two problems remain. **How do we tell causation from correlation?** That is **Hume's problem**, and the causal theory inherits it. And it may be **too narrow**: explanations in quantum theory, or of a law by a more general law, are hard to put causally.

## Second replacement: to explain is to unify

> [!definition] Unificationist theory (Michael Friedman 1974, Philip Kitcher 1981, 1989)
> Phenomena are explained by **fitting them into a broader pattern**. "Science advances our understanding of nature by showing us how to derive descriptions of many phenomena, using the same patterns of derivation again and again, and, in demonstrating this, it teaches us how to **reduce the number of types of facts we have to accept as ultimate** (or brute)." (Kitcher 1989)

A **brute fact** is one we must simply accept, with nothing explaining it. For Kitcher, understanding means "seeing **connections, common patterns**, in what initially appeared to be different situations", and an explanation is an **argument pattern** running from a few unified principles to many miscellaneous events. Think of physics merging electricity and magnetism into one theory.

- **Advantage:** it covers **non-causal explanations**, for example in quantum theory.
- **Historical support:** Darwin's evolution and Newton's later work on matter convinced scientists before making many new predictions, through their **explanatory promise**: unifying many phenomena with few principles.
- **The flagpole:** Kitcher argues causal talk is a loose summary of deeper asymmetries in unification: deriving shadows from poles fits a pattern covering far more phenomena than deriving poles from shadows.

## Godfrey-Smith: no single explanatory relation

Peter Godfrey-Smith (the week's reading) argues the debate was set up wrongly.

Causation and unification have been treated as rivals, but we need not choose. Much of the time, explaining means describing causal mechanisms or history; sometimes patterns or principles explain where causal language is hard to apply, and unification does the work. The champions converged: Salmon eventually accepted unification, Kitcher eventually accepted causation.

That is **pluralism** (several explanatory relations are legitimate), a step in the right direction. Godfrey-Smith goes further: the mistake is to think there is **one** special explanatory relation, or a fixed short list of two or three.

> [!definition] Godfrey-Smith's contextualism about explanation
> The standards for good explanation are **partially dependent on the scientific context**. Different fields, and the same field at different times, establish their own criteria. The standards in field A need not suffice in field B.

This is Thomas Kuhn's view (see [[PhilSci-L02 - Kuhn on Scientific Practice|L02]]), with Kuhn's example: Newton's gravity gave a mathematical law and no mechanism. Did it explain why things fall? Some said no; over time it became part of Newtonianism that the right kind of mathematical law **counts** as an explanation.

It is not "anything goes": a conception of explanation can embed a **factual error**. If good explanations must cite God's will and there is no God, that conception is mistaken.

So the covering law theory is dead **as a general account**, but some explanations do roughly have its form. The mistake was applying it everywhere.

**Is it convincing?** For: causation and unification each handle cases the other struggles with, and their champions conceded ground. Against: if "explanation" means different things in different fields, what makes them all explanation? De Regt and Dieks, below, answer that **understanding** is the common aim the kinds of explanation serve.

## Van Fraassen: explanation is outside science

Bas van Fraassen (empiricist, *The Scientific Image*, 1980) also makes explanation contextual, and draws the opposite conclusion about its place.

- Explanation and understanding belong to the **pragmatic dimension** of science: **contextual**, depending on our aims and preferences.
- Explanation is among the reasons to **accept** a theory as useful for particular purposes (predicting, answering questions).
- Explanations **do not add to our beliefs** about the relation between theory and world.
- So **to explain is not an epistemic aim of science** (an aim concerning knowledge). It is a pragmatic dimension of theory acceptance.

> [!definition] Van Fraassen: explanation as a three-term relation
> "The discussion of explanation went wrong at the very beginning when explanation was conceived of as a relationship like description: a relation between theory and fact. Really it is a **three-term relation, between theory, fact, and context**. An explanation is an answer... So **scientific explanation is not (pure) science but an application of science. It is a use of science to satisfy certain of our desires.**"

> [!warning] Two contextual views that are opposites
> Godfrey-Smith and van Fraassen both say explanation varies with context. For **van Fraassen** it is **external** to science: something people do *with* a theory to answer questions from outside scientific discussion, adding nothing to what we believe. For **Godfrey-Smith** it is **thoroughly internal**: assessing what explains what is an important part of scientific reasoning, with standards that differ by field.

Why it matters in [[PhilSci-L05 - Scientific Realism and its Critiques|L05]]: van Fraassen's anti-realism (constructive empiricism) needs explanation to be non-epistemic, or **inference to the best explanation** would push him to believe in unobservables. Mock Q8 is Musgrave's reply.

| | Explaining is... | Part of science? |
|---|---|---|
| **Hempel** | Deducing from laws | Yes |
| **Salmon** | Showing causes | Yes |
| **Kitcher** | Fitting into wide patterns | Yes |
| **Godfrey-Smith** | Set by each field | Yes, variously |
| **Van Fraassen** | Answering in a context | No, an application |

## Is understanding anything more than explanation?

Explanation relates a theory and a phenomenon; understanding is something a person achieves. Does it need its own theory?

**Hempel: no, it is psychology (eliminativism).** "Such expressions as 'realm of understanding' and 'comprehensible' do not belong to the vocabulary of logic, for they refer to the psychological and pragmatic aspects of explanation." Explanation is an **objective** relation between theory $T$ and phenomenon $P$; whether a subject $S$ understands is **epistemically irrelevant**. Hempel uses "pragmatic" as a synonym for **"subjective"**: studying understanding tells you about people's preferences, so it is for **psychology, not philosophy**.

**Reductivism: explanation is understanding enough.**

- **Khalifa (2012):** understanding is a **form of knowledge**, reducible to explanation and related notions, yet still **philosophically interesting** (unlike for Hempel).
- **Lipton (2004):** "understanding is not some sort of super-knowledge, but simply more knowledge: knowledge of causes."
- **Trout (2002):** the *Eureka!* feeling of understanding results from **cognitive biases** like overconfidence and hindsight, so it is no epistemic guide.

**Carnap's three senses of "understanding":**

1. **Pragmatic:** the **capability to use** a theory to describe and predict facts. Legitimate.
2. **Subjective:** **intuitive understanding**, the feeling of grasping. Rejected.
3. **Epistemic:** **knowing the semantic rules** (truth conditions) of a language system. Legitimate.

Even a logical empiricist allowed understanding as the ability to use a theory. That is where the lecture ends up.

**The equivocation on "pragmatic".** Friedman (1974) and Woodward and Ross (2021) say Hempel and van Fraassen slide between two meanings:

1. **Subjective**: varying with an individual's psychology.
2. **Use(able) for certain aims**: **compatible with an objective (inter-subjective) notion** of understanding.

On the second, **understanding can be an epistemic aim of science, achieved by adequate explanations**. Hempel's step from "pragmatic" to "epistemically irrelevant" only works on the first. Hence De Regt and Dieks: "Contra Hempel, van Fraassen, and Trout, we hold that the pragmatic nature of understanding is not inconsistent with it being epistemically relevant."

## Understanding as a skill: De Regt and Dieks

### What understanding adds

Pragmatic theories of understanding share three requirements:

1. **Explanation:** there is a scientific explanation of the phenomenon (a deduction, argument or model), often in Hempel's sense.
2. **Requirements of adequacy:** epistemic values, usually **internal consistency** and **empirical confirmation**.
3. **Useability:** it must be **possible to construct** adequate explanations.

Reductivists accept only (1) and (2): understanding = adequate explanation = knowledge. Those for whom understanding is **more than** explanation add (3), which is **not about knowledge but about the ability to act**: it involves **skills and judgement**.

So understanding a phenomenon requires an explanation, plus the ability to construct and use explanations. The explanation is the product; understanding is the achievement of a scientist who can produce it. That is why understanding can **unify** a pluralist view of explanation: causal, unificatory and law-based explanations are all routes by which a skilled scientist comes to understand.

> [!definition] The "grammar" of understanding
> **Scientist $S$ (in context $C$) understands phenomenon $P$ on the basis of theory $T$.**

Hempel's explanation has two places ($T$, $P$); understanding has at least four. It is **pragmatic** (it involves a subject and context) and **contextual** (the same theory can give understanding in one community and not another).

Why does using a theory take skill? Because a theory meets a phenomenon through a **model**:

$$T \sim M \sim P$$

where $T$ is the theory, $P$ the phenomenon, and $M$ a model that **represents $P$ so that $T$ can be applied to it** (a planet treated as a point mass). There are **no algorithms or strict rules** for building models, only **approximation, idealisation and pragmatic decisions**.

> [!definition] Understanding (De Regt)
> **Understanding = the skill to use a theory for building models to explain phenomena.**

### Intelligibility, CUP and CIT

To explain $P$ with $T$, $S$ needs the skills to use $T$, so $T$ must be **intelligible** to $S$.

> [!definition] Intelligibility
> **Intelligibility** = the value that scientists attribute to the qualities of a theory $T$ that facilitate the use of the theory. Not an intrinsic property of theories, but a **context-dependent value related to scientists' skills**. Example: **visualizability**.

CUP says when a phenomenon is understood; CIT tests when a theory is intelligible.

> [!definition] CUP: Criterion for Understanding Phenomena
> A phenomenon $P$ is **understood** scientifically **iff** there is an explanation of $P$ that is based on an **intelligible theory** $T$ and conforms to the basic epistemic values of **empirical adequacy** and **internal consistency**.

> [!definition] CIT: Criterion (test) for the Intelligibility of Theories
> A scientific theory $T$ (in one or more of its representations) is **intelligible** for scientists (in context $C$) if they can **recognise qualitatively characteristic consequences of $T$ without performing exact calculations** (or fully explicit theoretical argumentation).

The idea: scientists with "insight" into how a theory works can use it to build empirically adequate, consistent models.

> [!example] Passing the CIT
> A standard illustration from outside the lecture uses the kinetic theory of gases (a gas as many tiny moving molecules). A physicist who has it can say without calculating that heating a gas in a closed container raises its pressure: the molecules move faster and hit the walls harder and more often. Someone who can only get there by solving the equations has the theory but, by CIT, does not find it intelligible.

### Does this make understanding subjective?

Skills belong to individuals, so understanding looks subjective again. De Regt and Dieks answer with **three levels** of analysis of science. "The three-level distinction reconciles the existence of **universal aims of science** with the existence of **variation**" in how those aims are specified and applied.

- **Macro** (science as a whole): understanding is a **universal aim** of science.
- **Meso** (scientific communities, such as a discipline in a period): **standards of intelligibility** are set here, and vary from one community to another. They are **objective, but relative to the level of progress** of the discipline: contextual.
- **Micro** (individual scientists): individuals have or lack the skills.

So CIT does not make understanding subjective. Its context is the meso-level community with shared, trained standards, and whether a physicist can draw qualitative consequences without calculating is a public, testable fact about her competence. Trout's feeling is not. CIT does make understanding **contextual**, and the pressure point is a case like 1926 (below), where two groups in one field disagreed about intelligibility. The defence is that the community settled it over time.

**Tools for qualitative reasoning** (diagrams such as Feynman diagrams or the Bohr picture of the atom, causal-mechanical stories, analogies, thought experiments, toy models, simulations) let scientists anticipate a theory's consequences without calculating. They are taught and shared within a discipline, so they operate at the **meso-level**; individuals master them to different degrees.

**Is understanding an additional epistemic aim?** For: the account explains historical variation that one fixed standard cannot, and it answers Trout, since a skill that produces empirically adequate models is not a feeling. Against: a skill may be a **means** to the epistemic aims (truth, empirical adequacy) rather than an aim itself; a reductivist like Khalifa can call it knowledge plus competence.

## Standards of intelligibility change

Gravitation shows standards varying **diachronically** (over time); quantum theory around 1926 shows them varying **synchronically** (between scientists at the same time).

### Gravitation: from Newton to Einstein

In Newton's theory (1687), two masses attract with force $F = G\,m_1 m_2 / r^2$ ($r$ the distance between them, $G$ a constant), across empty space with nothing in between: **action at a distance**.

Christiaan Huygens (Dutch physicist, a follower of Descartes's view that bodies act only by contact) found this unintelligible: to say "that bodies fall down because of some gravitational attraction... is to **say nothing**." His standard was **contact action**.

**Around 1800** the standard had flipped: **action at a distance became the ideal of understanding**. Coulomb's law for electric charges, $F = k\,q_1 q_2 / r^2$, was intelligible because it had Newton's form. **After 1850** it flipped back: **action by contact**, through a field, became acceptable with Maxwell, and Einstein's relativity (1905, 1915) made gravity the curvature of space-time.

The phenomenon stayed the same; the community's standard of intelligibility changed. (Kuhn's Newton example above is the same history.)

### Quantum mechanics around 1926

Between 1920 and 1925 physics lost its pictures. Light and matter behave both as waves and as particles (wave-particle duality), so there was **no unambiguous visualization** and **no particle trajectories**. Pauli's fourth quantum number (spin, 1925) had no picture in Bohr's model of electron orbits, so atoms became completely non-visualizable.

Around 1926 two theories of the atom competed:

- **Matrix mechanics** (Werner Heisenberg, Wolfgang Pauli): **abstract**.
- **Wave mechanics** (Erwin Schrödinger): **visualizable**.

The debate was over when a theory is *anschaulich*. ***Anschaulichkeit*** is German for visualizability, intelligibility.

Schrödinger was "discouraged, if not repelled" by Heisenberg's "very difficult methods of transcendental algebra, and by the lack of *Anschaulichkeit*". Physicists did find his theory easier to work with, so it was more widely used.

Pauli (1924), on the demand for pictures: it "is partly legitimate and healthy", but "should never count as an argument for the retention of fixed conceptual systems in physics. Once the new conceptual systems are settled, then also these will be *anschaulich*." Visualizability is no fixed standard: new theories become intelligible once people have the skills to use them. That is De Regt's contextual view in a 1924 letter.

**Outcome.** Schrödinger's realistic reading of the wave function as a charge density failed; it is a **probability density** (Born). So his visualization was problematic, while Heisenberg went on to use visualizable concepts. The result was a new quantum mechanics combining both theories, plus Dirac's mathematical unification and Born's interpretation.

The lesson: **visualizability was valued as a property that increases the intelligibility** of theories, but it is **not necessary** for understanding. It is one tool, valued by some communities at some times.

## What to be able to write

- Hempel's D-N model: definition, the four conditions of adequacy, and the symmetry of explanation and prediction ("to be expected").
- The flagpole asymmetry argued fully; the barometer and hexed salt as backups.
- Salmon's causal theory and its problems (Hume's problem, too narrow); Kitcher's unification and its advantage (non-causal cases).
- Godfrey-Smith's contextualism, against pluralism and against van Fraassen (internal versus external).
- Van Fraassen: three-term relation, an application of science, not an epistemic aim.
- Hempel's eliminativism, the reductivists, and the two senses of "pragmatic".
- De Regt and Dieks: the grammar of understanding, understanding as skill ($T \sim M \sim P$), CUP and CIT near-verbatim, the three levels against the subjectivity charge.
- One historical case, diachronic (Huygens) or synchronic (Schrödinger against Heisenberg and Pauli).

The model answer for mock Q7 is the first exam question below.

## Exam questions

> [!exam]- Mock exam Q7: "Carl Hempel's "deductive-nomological model" for a long time was the "received view" of scientific explanation. Briefly summarize Hempel's model, and give at least one argument *against* the model's being a satisfactory account of scientific explanation." (half a page)
> **Key points:** explanandum deduced from general laws plus particular conditions; law used essentially, empirical content, explanans true; explanation and prediction symmetric; asymmetry objection (flagpole); barometer: correlation is not explanation.
>
> Follow the official key:
> 1. **The model.** An explanation is a **deductive argument**: the **explanandum** (the phenomenon to be explained) is deduced from an **explanans** of **general laws** plus **particular facts and initial or boundary conditions**.
> 2. **Conditions of adequacy.** The explanandum is a logical consequence of the explanans; the explanans contains a general law used **essentially**; it has **empirical content**; it is **true**.
> 3. **Explanation and prediction are symmetric:** same logical structure. To explain is to show the phenomenon was to be expected given the laws.
> 4. **Objections** (the key lists all of these; one argued well is enough):
>    - **Asymmetry:** flagpole height, sun's position and optics yield the shadow's length (explains); shadow length and the same laws yield the pole's height (predicts, does not explain). Both satisfy every D-N condition, so the model is not sufficient.
>    - **Irrelevance:** salt hexed by a magician dissolves in water. Valid, lawlike, true, and the hex explains nothing.
>    - **Apparently missing causation** (correlation or expectation $\neq$ explanation): a falling barometer predicts a storm but does not explain it, since both have a common cause (the drop in atmospheric pressure).
>    - **Explanations do not always involve laws:** citing a causally relevant fact can be enough ("the window broke because the ball hit it").
>
> Lead with the asymmetry (the strongest), and have the barometer ready as the second, since it is named in the official answer.

> [!exam]- Causation or unification: which gives the better account of scientific explanation? Use the flagpole, Salmon, Kitcher and Godfrey-Smith.
> **Key points:** Salmon: explanation situates the explanandum in the causal nexus, solves the flagpole; causal costs: Hume's problem, too narrow; Friedman and Kitcher: unification reduces brute facts, covers non-causal cases; Godfrey-Smith: not competitors, contextualism about standards.
>
> - **Causal (Salmon 1984):** explanation situates the explanandum in the **causal nexus**, showing the mechanisms that produce it. Solves the flagpole at once: the pole causes the shadow. Costs: it inherits **Hume's problem** (telling causation from correlation), and it may be **too narrow**, since explanations in quantum theory or of a law by a more general law are hard to phrase causally.
> - **Unification (Friedman 1974, Kitcher 1981, 1989):** explanation fits phenomena into a broader pattern, using the same patterns of derivation again and again and reducing the number of brute facts. Handles **non-causal explanation**. On the flagpole, Kitcher says causal talk summarises deeper asymmetries in unification: deriving shadows from poles belongs to a far wider pattern.
> - **Godfrey-Smith:** treating them as competitors is the mistake. Salmon came to accept unification, Kitcher came to accept causation. He goes past this pluralism to **contextualism**: there is no single special explanatory relation (or fixed short list), and standards of good explanation partly depend on the scientific field and period (following Kuhn). Not "anything goes": a standard can embed a factual error.

> [!exam]- Is explanation internal to science or external to it? Contrast van Fraassen with Godfrey-Smith.
> **Key points:** both treat explanation as context-dependent; van Fraassen: three-term relation, external, not an epistemic aim; Godfrey-Smith: internal, standards vary by field and period; stake: inference to the best explanation and unobservables.
>
> - **Shared ground:** both say explanation varies with context.
> - **Van Fraassen** (*The Scientific Image*, 1980): explanation is **pragmatic** and **external**. It is a **three-term relation between theory, fact and context**; an explanation is an answer to a why-question. It counts among reasons to **accept** a theory for some purpose but adds nothing to our **beliefs** about how theory relates to world. So it is **not an epistemic aim of science**: "scientific explanation is not (pure) science but an application of science", "a use of science to satisfy certain of our desires".
> - **Godfrey-Smith:** explanation is **thoroughly internal**. Assessing explanatory power is an important part of scientific reasoning, but each field (and period) sets its own standards.
> - **Why it matters:** van Fraassen's constructive empiricism needs explanation to be non-epistemic, or inference to the best explanation would push him to believe in unobservables (Musgrave's reply targets exactly this).

> [!exam]- Is scientific understanding anything more than having an explanation? Answer with Hempel, the reductivists, and De Regt and Dieks.
> **Key points:** Hempel: understanding epistemically irrelevant; reductivists: Khalifa, Lipton, Trout; De Regt and Dieks: understanding as skill (useability); two senses of pragmatic, meso-level standards, CUP and CIT; intelligibility standards vary historically.
>
> - **Hempel (eliminativist):** understanding belongs to the psychological, "pragmatic" side of explanation, with pragmatic meaning subjective. It is **epistemically irrelevant**, a topic for psychology.
> - **Reductivists:** explanation is understanding enough. **Khalifa:** understanding is a form of knowledge reducible to explanation (still philosophically interesting). **Lipton:** "simply more knowledge: knowledge of causes". **Trout:** the feeling of understanding is a product of cognitive biases.
> - **De Regt and Dieks:** understanding adds a third requirement, **useability**: the **skill to use a theory to build models** ($T \sim M \sim P$) that explain phenomena. This is an ability, not knowledge.
> - **Against the subjectivity charge:** separate "pragmatic" as **subjective** from "pragmatic" as **useable for aims**; the second is objective (inter-subjective). Standards of intelligibility are set at the **meso-level** of scientific communities. **CUP** and **CIT** make it testable: can scientists recognise qualitatively characteristic consequences of $T$ without exact calculation?
> - **Evidence:** standards of intelligibility vary historically (Huygens on Newton's action at a distance) and between contemporaries (Schrödinger against Heisenberg and Pauli, 1926).
>
> Losing marks: using "pragmatic" without saying which sense.

## Flashcards

> [!card]- What is explanatory inference?
> Inference from data to a hypothesis about a **hidden structure or process** that would explain the data.

> [!card]- Why did empiricists distrust the idea that science explains?
> Explaining seemed to claim **metaphysical insight** into hidden causes and essences behind appearances.

> [!card]- Why did Duhem hold that explanation is not an aim of physics?
> Explaining "gives hostages to fortune": it makes physics **depend on metaphysics**, like "atoms exist".

> [!card]- In Hempel's deductive-nomological model, what is an explanation?
> A **deductive argument** in which the explanandum is deduced from general laws plus particular conditions.

> [!card]- In Hempel's model, what is the explanandum?
> The phenomenon (or law) to be explained: the $X$ in "why $X$?".

> [!card]- What are Hempel's four conditions of adequacy for a D-N explanation?
> Explanandum is a **logical consequence** of the explanans; a **general law** used essentially; **empirical content**; explanans **true**.

> [!card]- What does Hempel claim about the relation between explanation and prediction?
> They are **symmetric**: same logical structure, differing only in whether the conclusion is already known.

> [!card]- What is inductive-statistical (I-S) explanation?
> The covering law variant with a **probabilistic** law: the explanans shows the explanandum was **highly probable**.

> [!card]- What does syphilis explaining symptoms that only some patients develop show against inductive-statistical explanation?
> **High probability** of the explanandum is not necessary for explanation.

> [!card]- Must the laws in a Hempelian D-N explanation be causal?
> No. **Functional laws** count: $PV = nRT$ explains without saying which quantity causes which.

> [!card]- Is Hempel's D-N model meant to describe how scientists actually explain?
> No, it is **normative**: the ideal of a good explanation, like the logician's concept of **proof**.

> [!card]- What does the asymmetry objection to Hempel's D-N model say?
> Deductions can run **both ways**, explanations cannot, so the D-N model admits backwards explanations.

> [!card]- In the flagpole case against Hempel, which direction of deduction explains?
> Pole height to shadow length **explains**; shadow length to pole height only **predicts**.

> [!card]- Why does the flagpole case show the D-N conditions are not sufficient for explanation?
> The shadow-to-pole deduction meets **all four conditions** of adequacy yet explains nothing.

> [!card]- Why can a symptom predict a disease but not explain it?
> Inference runs both ways, but **explanation runs only from disease to symptom**.

> [!card]- How did Hempel reply to the asymmetry objection?
> He **bit the bullet**: if his theory allows both directions, both are fine. Hopeless for the flagpole.

> [!card]- What does the hexed-salt example show against Hempel's D-N model?
> **Irrelevance**: true, used premises (the hex) can satisfy the D-N conditions yet explain nothing.

> [!card]- Why does a falling barometer predict a storm without explaining it?
> Both are effects of a **common cause**, the drop in atmospheric pressure. Correlation is not explanation.

> [!card]- What does "the window broke because the ball hit it" show against Hempel's D-N model?
> Explanations **need not involve laws**: citing a causally relevant fact can be enough.

> [!card]- What do the flagpole, hexed-salt and barometer counterexamples to Hempel have in common?
> All are missing **causation**: the pole causes the shadow, the hex causes nothing, barometer and storm share a cause.

> [!card]- What is explanation according to Salmon's causal theory?
> **Uncovering causes**: situating the explanandum in a **nexus of causal relations**, the mechanisms that produce it.

> [!card]- How does a causal theory of explanation solve the flagpole problem?
> Sunlight hitting the **pole causes the shadow**, not the reverse, so explanation runs pole to shadow.

> [!card]- What price does a causal theory of explanation pay?
> It must say **what causality is**, which empiricists since Hume have regarded as suspect.

> [!card]- What is causation on the regularity theory?
> **Constant conjunction**: $C$ regularly followed by $E$, producing an **expectation** in us, not a necessary connection.

> [!card]- What is causation on the counterfactual theory?
> If cause $C$ **hadn't occurred**, effect $E$ wouldn't have occurred (Woodward).

> [!card]- Which theory of causation, held by Salmon, says we can identify causal processes and interactions in nature?
> The **process** theory (picture a chain of falling dominoes).

> [!card]- Why does the regularity theory of causation bring back the barometer problem?
> Barometer drops are **constantly conjoined** with storms, so they would count as causes.

> [!card]- What is Hume's problem as it applies to causal theories of explanation?
> How to **distinguish causation from correlation**.

> [!card]- Why might causal theories of explanation be too narrow?
> Explanations in **quantum theory**, or of a law by a more general law, are hard to phrase causally.

> [!card]- On the unificationist theory, how are phenomena explained?
> By **fitting them into a broader pattern**, reducing the number of facts accepted as brute.

> [!card]- Which two philosophers developed the unificationist theory of explanation?
> **Michael Friedman** and **Philip Kitcher**.

> [!card]- What is the main advantage of unification over causal theories of explanation?
> It accounts for **non-causal explanations**, such as those in quantum theory.

> [!card]- Why were Darwin's evolution and Newton's later work compelling before making many new predictions, for Kitcher?
> **Explanatory promise**: they unified a great range of phenomena with a few principles.

> [!card]- How does Kitcher handle the flagpole asymmetry?
> Causal talk summarises deeper **unification asymmetries**: deriving shadows from poles fits a far wider pattern.

> [!card]- What is Godfrey-Smith's contextualism about explanation?
> Standards of good explanation **partially depend on the scientific context**: each field and period sets its own.

> [!card]- What is the difference between pluralism and Godfrey-Smith's contextualism about explanation?
> Pluralism allows a **fixed short list** of relations; contextualism lets each **field and period** set its standards.

> [!card]- What is the main objection to Godfrey-Smith's contextualism about explanation?
> If explanation differs by field, it is unclear **what makes them all explanation**.

> [!card]- What is Kuhn's Newtonian gravity example of changing standards of explanation?
> Newton gave a law and no mechanism; it became accepted that the right **mathematical law counts** as explanation.

> [!card]- Why is Godfrey-Smith's contextualism about explanation not "anything goes"?
> A conception of explanation can embed a **factual error**, e.g. requiring citation of God's will when there is no God.

> [!card]- For van Fraassen, does explanation add to our beliefs about how theory relates to the world?
> No. It is a reason to **accept** a theory for purposes, so **not an epistemic aim** of science.

> [!card]- What kind of relation is explanation for van Fraassen?
> A **three-term relation** between **theory, fact and context**; an explanation is an answer.

> [!card]- What is the difference between van Fraassen and Godfrey-Smith on explanation?
> Both make it contextual: van Fraassen puts it **external** to science, Godfrey-Smith **internal** to scientific reasoning.

> [!card]- Why does van Fraassen's constructive empiricism (his anti-realism) need explanation to be non-epistemic?
> Otherwise **inference to the best explanation** would push him to believe in unobservables.

> [!card]- What is Hempel's eliminativism about understanding?
> Understanding is **epistemically irrelevant**: a topic for **psychology, not philosophy**.

> [!card]- What is reductivism about understanding?
> **Explanation is understanding enough**: no separate theory of understanding is needed.

> [!card]- What separates Khalifa's reductivism about understanding from Hempel's eliminativism?
> Khalifa reduces understanding to knowledge and explanation but still finds it **philosophically interesting**.

> [!card]- What is Lipton's view of understanding?
> Not super-knowledge but **simply more knowledge: knowledge of causes**.

> [!card]- Why does Trout think the feeling of understanding is no epistemic guide?
> The *Eureka!* feeling results from **cognitive biases** such as overconfidence and hindsight.

> [!card]- What are Carnap's three senses of "understanding"?
> **Pragmatic** (capability of use), **subjective** (intuitive feeling), **epistemic** (knowing semantic rules).

> [!card]- What two meanings of "pragmatic" do Friedman and Woodward and Ross distinguish?
> **Subjective** (varies with individual psychology) versus **useable for certain aims** (compatible with objectivity).

> [!card]- Why does Hempel's step from "pragmatic" to "epistemically irrelevant" fail?
> It only works if pragmatic means **subjective**; read as **useable for aims**, understanding can be epistemic.

> [!card]- What three requirements for understanding do pragmatic theories share?
> **Explanation**, **requirements of adequacy** (consistency, empirical confirmation), **useability**.

> [!card]- What is the difference between reductivists and "more than explanation" theorists of understanding?
> Reductivists accept only explanation and adequacy; the others add **useability**.

> [!card]- What is the useability requirement for understanding about, if not knowledge?
> An **ability to act**: it involves **skills and judgement**.

> [!card]- How does understanding relate to explanation, for De Regt and Dieks?
> Understanding needs an explanation **plus the skill** to construct and use explanations.

> [!card]- Why can understanding unify a pluralist view of explanation, for De Regt and Dieks?
> Causal, unificatory and law-based explanations are all **routes to understanding** for a skilled scientist.

> [!card]- What are De Regt and Dieks's three levels of analysis of science?
> **Macro** (science as a whole), **meso** (scientific communities), **micro** (individual scientists).

> [!card]- At which of De Regt and Dieks's levels are standards of intelligibility set?
> The **meso-level** of scientific communities: objective, but relative to a discipline's level of progress.

> [!card]- What is De Regt's basic form of a claim of understanding?
> Scientist $S$ (in context $C$) understands phenomenon $P$ on the basis of theory $T$.

> [!card]- What is the difference between Hempel's explanation and De Regt's understanding in number of places?
> Hempel's explanation is **two-place** ($T$, $P$); De Regt's understanding is at least **four-place** ($S$, $C$, $P$, $T$).

> [!card]- In De Regt's $T \sim M \sim P$, what does the model $M$ do?
> It **represents $P$ so that $T$ can be applied** to it.

> [!card]- Why does model-based explanation require skills, according to De Regt?
> There are **no algorithms or strict rules** for building models: approximation, idealisation, pragmatic decisions.

> [!card]- What is understanding, according to De Regt?
> The **skill to use a theory for building models** to explain phenomena.

> [!card]- What do De Regt and Dieks call the value scientists attribute to qualities of a theory that facilitate its use?
> **Intelligibility**.

> [!card]- What is intelligibility, according to De Regt?
> The **value** scientists attribute to qualities of a theory that **facilitate its use**.

> [!card]- What does CUP, De Regt and Dieks's Criterion for Understanding Phenomena, say?
> $P$ is understood iff explained by an **intelligible theory** conforming to **empirical adequacy** and **internal consistency**.

> [!card]- What does CIT, De Regt and Dieks's Criterion for the Intelligibility of Theories, say?
> $T$ is intelligible if scientists can **recognise qualitatively characteristic consequences** without **exact calculations**.

> [!card]- What is the difference between De Regt and Dieks's CUP and CIT?
> **CUP** says understanding phenomena needs an intelligible theory; **CIT** tests when a theory is intelligible.

> [!card]- How does the kinetic theory of gases illustrate passing De Regt and Dieks's CIT?
> Without calculating, a physicist sees heating a closed gas **raises pressure**: molecules hit walls harder and more often.

> [!card]- Why does CIT not make understanding subjective?
> Context means the **meso-level community**; recognising consequences without calculating is a **public, testable** competence.

> [!card]- At which level do conceptual tools like diagrams, analogies and toy models operate?
> The **meso-level**: taught and shared within a discipline.

> [!card]- What is the main objection to understanding as an additional epistemic aim of science?
> A skill may be only a **means** to the real epistemic aims, truth or empirical adequacy.

> [!card]- What is the difference between the gravitation and 1926 quantum cases of varying intelligibility standards?
> Gravitation varies **diachronically** (over time); quantum mechanics **synchronically** (between contemporaries).

> [!card]- Why did Huygens find Newton's theory of gravitation unintelligible?
> As a Cartesian mechanist his standard was **contact action**; attributing falling to gravitational attraction "says nothing".

> [!card]- What happened to the intelligibility of action at a distance around 1800?
> It became the **ideal** of understanding: Coulomb's law copied Newton's form.

> [!card]- From Newton to Einstein, what shifted which theories of gravitation counted as intelligible?
> The **community's standards** of intelligibility changed; the phenomenon stayed the same.

> [!card]- In the 1926 quantum debate, which theory was the visualizable one?
> **Wave mechanics** (Schrödinger); matrix mechanics (Heisenberg, Pauli) was abstract.

> [!card]- Which German term for visualizability was at stake in the 1926 dispute between Schrödinger and Heisenberg and Pauli?
> ***Anschaulichkeit***.

> [!card]- Why had atoms become non-visualizable by 1925?
> **Wave-particle duality** left no particle trajectories, and **spin** had no picture in the Bohr model.

> [!card]- What was Pauli's reply to the demand for visualizability?
> Once new conceptual systems are settled, **they too become *anschaulich***: visualizability is not a fixed standard.

> [!card]- What does the 1926 quantum debate show about visualizability and understanding?
> It was valued for **increasing intelligibility** but is **not necessary** for understanding: one tool, valued in some contexts.

## Links

- **Course:** [[PhilSci - Overview|Course overview]] · [[PhilSci - Tutorials|Tutorials]] (tutorial 4 is on this lecture's two readings)
- **Previous:** [[PhilSci-L01 - Introduction and Logical Empiricism]] (Carnap, the logical empiricist background) · [[PhilSci-L02 - Kuhn on Scientific Practice]] (Kuhn's paradigm-relative standards, which Godfrey-Smith's contextualism builds on) · [[PhilSci-L03 - Under-determination]] (Duhem)
- **Next:** [[PhilSci-L05 - Scientific Realism and its Critiques]] (van Fraassen's constructive empiricism, inference to the best explanation, mock Q8)
- **Source:** `4 Scientific Explanation and Scientific Understanding.pdf` (Enrico Cinti, 22 September 2026). Readings: Godfrey-Smith (2003), *Theory and Reality*, ch. 13 "Explanation", pp. 190-201; De Regt and Dieks (2005), "A Contextual Approach to Scientific Understanding", *Synthese* 144 (not read for this note: its content here comes from the lecture and tutorial slides); optional: De Jong and De Haro, "Technological Understanding".
- **Further reading, from the deck:** Woodward and Ross, "Scientific Explanation", *Stanford Encyclopedia of Philosophy* · Reutlinger et al. (2018), "Understanding (with) Toy Models", *BJPS* · De Regt (2017), *Understanding Scientific Understanding*, Oxford University Press
