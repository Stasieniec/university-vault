---
type: lecture
course: 5354PHSC6Y
week: 7
lecture: 7
date: 2026-10-13
lecturer: Enrico Cinti
book_sections: ["Chen, Laws of Physics, sections 1-2.4, 3.1-3.2, 4.1-4.3"]
status: complete
topics:
  - Where the idea of laws of nature came from
  - Laws versus accidental generalisations
  - Counterfactuals, possible worlds and causation
  - Ontology, state space and nomological possibility and necessity
  - Determinism, strong determinism and chance
  - Maudlin's fundamental laws of temporal evolution and the objections
  - Humean versus non-Humean accounts
  - Chen's minimal primitivism
  - The Dretske-Tooley-Armstrong account (universals)
  - Dispositionalism
  - Lewis's Best Systems Account and the Humean mosaic
---

# PhilSci-L07: Laws of Nature

> [!abstract] In short
> What is a law of nature? "All Fs are Gs", true everywhere and always, is not enough: "all gold spheres are less than a mile across" is true and is no law. A law seems to *force* something to be true, where an accident just happens to be true. That force shows up in what laws do: they back counterfactuals ("if this match had been struck, it would have lit"), fix what is physically possible, and decide whether the world is deterministic.
>
> The debate is between two camps. **Non-Humeans:** laws govern the world, and the necessity they carry is real and cannot be reduced to anything else. Three versions: laws are primitive and govern by constraining (Chen), laws are necessitation relations between universals (Dretske, Tooley, Armstrong), laws come from dispositions. **Humeans** (Lewis): laws are the axioms of the best summary of everything that actually happens. The world fixes the laws, where non-Humeans have the laws fix the world.
>
> No mock exam question is on laws, but "Laws of nature" is topic 6 of the syllabus, so it is examinable. The practice questions below follow the mock's style: explain a view, then assess it in half a page.

## Where the idea of a law came from

Before modern science, God was thought to direct the world directly. Then Galileo and Newton produced **equations of motion** that predict the future from the present, **deterministically**: give the positions and velocities now, and the equations fix everything after. At first sight this leaves God nothing to do.

The response was to call the equations **laws of nature** and to say that laws **govern** how the world evolves. God fixes the laws, the laws run the world, and God is still in control even if the evolution is deterministic. Laws were the bridge between a directed world and a mathematical one.

That raised a new question: what *are* laws? Empiricists (who hold that all knowledge of the world comes from experience, see [[PhilSci-L01 - Introduction and Logical Empiricism|L01]]) were sceptical. David Hume's slogan is "no necessary connections between distinct entities": we see one event followed by another, never a necessity binding them. Against Hume and his followers, the **Humeans**, stand those who defend the reality of laws: laws govern, encode necessary connections, have a modal strength of their own ("modal": about what is possible or necessary). That split is still the shape of the debate.

**Scope.** The lecture is about the **fundamental laws of physics**, not laws of the special sciences (biology, economics, psychology). We have no candidate theory that supplies them (string theory is a hope at best), and we do not know there is one. So the approach is **suppositional**: given current science and the concept of a law, what might fundamental laws look like, knowing future science may overturn it.

## The puzzle: laws versus accidents

The obvious first answer is that a law is a **true universal generalisation**: a true statement of the form "all Fs are Gs". It fails, because of **accidental generalisations**: universal statements that are true by accident.

> [!example] The gold spheres
> "All gold spheres are less than one mile in diameter" is true: there is not that much gold gathered anywhere. But nothing, and in particular no law of nature, forces it to be true. It just happens to be the case.
>
> Compare (from the optional Carroll reading, outside the lecture): "all uranium spheres are less than one mile in diameter". Same form, also true, but plausibly a law: uranium's critical mass rules out such a sphere. Nothing in the form of the two sentences tells them apart.

So every account of laws works the same way: it adds conditions to "true universal generalisation" until only laws satisfy them. In other words, it tries to capture the sense in which laws **make** something true, where the gold-sphere statement just happens to be true.

(A link the lecture does not draw: Hempel's covering law model, [[PhilSci-L04 - Scientific Explanation and Understanding|L04]], needs a general law in every explanation, so it needs this distinction too.)

## What laws do

Laws sit at the centre of a web of concepts: counterfactuals, causation, the structure of theories, determinism, chance, time. Adopting an account of laws commits you on all of them, and each also shows what a law adds to an accident.

### Laws back counterfactuals

A **counterfactual** is a statement "if A were to happen, B would happen", usually about an A that did not happen. Counterfactuals are **modal** statements, about possibility and necessity. Equivalently, they are claims about **possible worlds**: complete ways the world could have been.

How would you check whether one is true? A first try: in **every** possible world where A holds, B holds. Too strong: there are possible worlds where the match is struck and does not light (it is wet, there is no oxygen, chemistry works differently), yet "if this match had been struck, it would have lit" still seems true. A better try, from David Lewis and Robert Stalnaker: the worlds where A and B both hold are **more similar to ours** than the worlds where A holds without B. Only the closest A-worlds count.

> [!definition] Laws and counterfactuals
> Laws **support counterfactuals**, because laws fix a **crucial respect of similarity** between worlds: a world that keeps our laws is much closer to ours than one that breaks them. So "if it were A then B" is true when it is an instance of a law.

> [!example] The match and the coin
> "If this match had been struck, it would have lit": true, because it instantiates a law (struck matches in normal conditions light).
>
> Suppose every coin in my pocket happens to be silver. "If this copper coin were in my pocket, it would be silver" is false. "All coins in my pocket are silver" is accidental, so it supports no counterfactual.

That gives a test for the gold spheres too: "if someone gathered enough gold, the sphere would still be under a mile" is false, so the generalisation is no law.

**Causation.** Hume, and later Lewis, used this link to give an account of causation: two events are causally related when there is **counterfactual dependence** between them. My striking the match caused it to light because, had I not struck it, it would not have lit.

### Laws pick out what is physically possible

If you are a scientific realist ([[PhilSci-L05 - Scientific Realism and its Critiques|L05]]), a theory specifies two things:

- an **ontology**: the entities there are, which correspond to the theory's variables;
- **dynamics** for those variables, saying how they evolve. **These are the laws.**

A convenient picture is the **state space**: a space where every point is one possible state of all the theory's entities. For $N$ particles, a point lists all $N$ positions (and momenta). As the world changes, its state moves, so a **trajectory** through state space is a **history** of the entities.

Without laws, any curve through state space is a possible history. **Laws are rules that select only some histories as allowed.** Newton's $F = ma$ (force equals mass times acceleration) plus his law of gravitation, for instance, allow exactly the curves that solve their equations. Treat each history as a possible world, and laws select the allowed worlds.

> [!definition] Nomological possibility and necessity
> Something true in **some** of the law-allowed worlds is **nomologically possible**; something true in **all** of them is **nomologically necessary**. This is **nomological modality**: the possibility and necessity that come with the laws of nature ("nomological" from Greek *nomos*, law).

### Laws decide determinism

> [!definition] Determinism
> Roughly: if you know the state of the universe at one time and you know the laws, you know its state at any other time.
>
> In possible-worlds language: given a segment of a nomologically possible world, there is **only one** full nomologically possible world with that segment as part.

Picture it in state space: under deterministic laws the allowed trajectories never cross or branch, so a single point fixes the whole curve. Whether this holds is a fact about the laws, which is why determinism is "highly tangled up" with them.

- **Newtonian mechanics** is the classic deterministic theory.
- **General relativity** is the interesting case. Only its **globally hyperbolic** models are deterministic: those where the state at one time plus the Einstein field equations (EFE: the equations relating spacetime's curvature to the matter in it) fix the state at all times. Not every model of the theory is globally hyperbolic. So whether general relativity is deterministic depends on whether the other models count as **physically significant**.
- **Quantum mechanics**, on collapse interpretations (on measurement the quantum state jumps at random to one outcome), is **not deterministic**.

A theory that fails to be deterministic involves **chance in an ineliminable way**: probabilities are part of what the laws say, and there is no way to get rid of them.

**Strong determinism.** Eddy Keming Chen, following the physicist Roger Penrose, discusses a stronger notion: **only one world** is compatible with the laws. Ordinary determinism fixes the world once you give a segment; strong determinism fixes it with no input at all. So the theory itself says there is only one possible world. There are no real examples (string theory was hoped to have the property, but does not on any reasonable reading). It still matters as a distinction, since we do not know what the fundamental laws will look like.

### Must laws push the world forward in time?

Tim Maudlin argues that laws necessarily involve **evolution in time**, a **direction** of time, and **dynamic evolution**: laws **produce later states of the world from earlier ones**. Laws, for him, are **Fundamental Laws of Temporal Evolution (FLOTEs)**. $F = ma$ and the Schrödinger equation (quantum mechanics' equation for how the state changes) are FLOTEs.

Two objections:

1. **Most laws of physics are time-symmetric**: run any allowed history backwards and you get another allowed history. The direction of time is **emergent**, for example from thermodynamics (entropy, roughly disorder, increases in one direction). In Chen's reading this asymmetry comes from the **Past Hypothesis**: a boundary-condition law saying that at one end of time the universe was in a low-entropy state.
2. **Not all laws describe evolution.** Some, like the Einstein field equations, are **constraints**: they say which whole spacetimes are allowed, without producing one moment from the last. Chen adds Gauss's law (a purely spatial law linking electric charge to the field around it) and principles of least action (which constrain a whole history between two times).

Both objections come back in the non-Humean accounts below.

## The great divide: Humeans and non-Humeans

Today the debate is mostly **Humean versus non-Humean**.

> [!definition] The two camps
> - **Non-Humeans:** laws **govern** the world, and there is an **irreducible** form of nomological modality associated with the laws themselves.
> - **Humeans** deny both: laws **do not govern**, and nomological modality is **reducible to non-modal notions** (facts about what actually happens).

Non-Humean views come in more varieties. The lecture takes three, using Chen's labels as well:

- **Primitivism**: Maudlin's view, and Chen's **minimal primitivism**, a variant of it;
- **Universals**: the Dretske-Tooley-Armstrong account, which Chen calls **Platonic reductionism**;
- **Dispositionalism**, which Chen calls **Aristotelian reductionism**.

## Non-Humean 1: minimal primitivism (Chen)

"Primitive" means we do **not analyse** a notion in terms of anything else, possibly more fundamental. A primitive notion is conceptual rock-bottom. Saying laws are primitive gives up any chance of analysing them, in exchange for avoiding the complicated metaphysics any analysis brings. Whether that trade-off is worth it is debatable.

**How primitive laws govern: by constraining.** Laws restrict the space of possibilities admissible for the theory's entities. The analogy is **mathematical axioms**: axioms govern mathematics by fixing what is allowed and what is not, without directing any evolution. $F = ma$ governs by limiting the physical possibilities to those compatible with $F = ma$ and requiring the actual world to be one of them.

> [!definition] Minimal primitivism (Chen, with Sheldon Goldstein)
> Fundamental laws of nature are certain **primitive facts** about the world. There is **no restriction on the form** of the fundamental laws. They **govern** the behaviour of material objects **by constraining the physical possibilities**.

**Why it is non-Humean:** the restriction comes from the laws and is irreducible, so laws really govern, and they govern by constraining.

**Why "minimal":** compared with heavier primitivism like Maudlin's, it commits to the least irreducible nomological modality you can have and still be non-Humean. There are only the entities and a constraint on how they may combine. It sets no substantial conditions on what laws look like, unlike Maudlin's demand that they be FLOTEs. So time-symmetric laws and constraint laws like the Einstein equations fit without trouble.

From Chen's reading: simplicity and informativeness are only **epistemic guides** for discovering laws, where Humeans make them part of what a law is (below). And to the **mystery objection** (who imposes the laws?) Chen answers with the axiom analogy: $2 + 3 = 5$ constrains every possible world, and no agent imposes it.

## Non-Humean 2: universals (Dretske, Tooley, Armstrong)

A **universal** is, roughly, the abstract entity that encodes a property in full generality. There is a universal of redness, and every red thing we see is an instance of it. So universals are **multilocated**: wholly present wherever they are instantiated.

On the **DTA account** (after Fred Dretske, Michael Tooley and David Armstrong), laws are **relations of necessitation between universals**.

> [!definition] The DTA account
> A law has the form "**necessarily, if F then G**", where F and G are universals. Written $N(F, G)$: the universal F-ness **necessitates** the universal G-ness, and this makes it the case that all Fs are Gs.

$F = ma$ becomes: necessarily, if the universal force is such and such, the universals mass and acceleration are such and such. Insofar as the modality coming from the laws is irreducible, the account is non-Humean.

It also answers the gold-sphere puzzle: "all uranium spheres are under a mile" holds because of a relation between universals, while nothing links being a gold sphere to being small.

**Advantages.** It fits some simple examples of laws naturally. And it is **reductive**: it analyses laws in terms of something more fundamental (universals). So it is **more explanatory than primitivism**: it says what laws are, where primitivism declines to.

**Problems.**

- It is unclear whether **all laws fit the schema**. What about laws that behave like constraints, such as the Einstein equations, or **path integrals** (a formulation of quantum mechanics that sums over all possible histories between two points)?
- From Chen's reading: the necessitation relation is **put in by hand**, and it is unclear why $N(F,G)$ should guarantee that all Fs are Gs.

## Non-Humean 3: dispositions

**Dispositionalism** (Chen's Aristotelian reductionism) analyses laws in terms of **causation** instead of universals, using a specific notion of causation built on **dispositions**.

> [!definition] Disposition
> A property of the kind "**glass breaks when it falls**": a property that comes with a specific contextual clause determining when it is realised (manifested). Fragility, solubility and electric charge are standard examples.

**Causation from dispositions:** certain events have a disposition to happen when certain other events happen, and the latter are their causes. The stone caused the window to break because windows have a disposition to break when hit by stones.

**Laws from causation:** exploit the link between causation and laws. For instance, Heather Demarest: laws are the **best possible summary of certain patterns of causal relations**.

**Is it non-Humean?** Yes, insofar as dispositions are **irreducible and modal**.

- **Modal:** a disposition encodes a **tendency**: under certain conditions something happens. So it is about something possible that is actualised under certain conditions. And the disposition is there **before** it is actualised (unstruck glass is still fragile), so it must be **intrinsically modal**.
- **Irreducible:** less obvious. It is the claim that dispositions must **not be analysed further**: for example, that glass's disposition to break cannot be explained by some non-dispositional property of the glass (Cinti's example: being fragile, treated as a non-dispositional property). Friends of dispositions deny that such explanations are available at the fundamental level.

Keep one nuance apart: Chen says that on this view **laws themselves do not govern**; the necessity and "oomph" live in the dispositions, and laws are derived from them. It counts as non-Humean because of the irreducible modality in the dispositions.

**Assessment.** It is **reductionist about laws** (laws analysed in terms of dispositions), so it has the **same advantage over primitivism** as DTA. It also has **similar problems**: irreducible dispositions are as mysterious as universals and primitive laws. And it is not obvious that **modern physical theories fit the dispositionalist mould**. Chen adds that conservation laws do not look like manifestations of dispositions, and that a manifestation must follow its stimulus, which builds in the fundamental direction of time the objections to Maudlin questioned.

## Humeanism: the Best Systems Account

The main alternative to all three is **Humeanism**, today chiefly David Lewis's **Best Systems Account (BSA)**, also called the **Mill-Ramsey-Lewis** account (after John Stuart Mill and Frank Ramsey).

> [!definition] Best Systems Account
> Laws of nature are **the axioms of a system that provides the best balance of strength and simplicity** when accounting for the **mosaic of spatiotemporally related local matters of fact** of our world.

Unpacking it:

- **The Humean mosaic** is the collection of local matters of fact, arranged in space and time: things like "a particle of mass $m$ is here". It is the **bedrock**: the ontological structure that determines the laws. Nothing in it is modal.
- **A system** is a set of true sentences about the mosaic, organised as axioms and their consequences.
- **Strength** is how much the system tells you about the mosaic; **simplicity** is how short and elegant it is. They pull against each other.

> [!example] Choosing the best system (Chen's illustration)
> Take a world of charged particles and an electromagnetic field.
> - **System 1** lists every fact, point by point. Maximally strong, hopelessly complicated.
> - **System 2** says "Things exist." Maximally simple, says almost nothing.
> - **System 3** is Maxwell's equations plus the Lorentz force law (the law for the force a field exerts on a charge). Fairly simple and very informative: the best balance.
>
> So, on the BSA, Maxwell's equations and the Lorentz force law are this world's laws.

This handles the puzzle: laws are distinguished from mere universal generalisations by belonging to the best balance of strength and simplicity. "All gold spheres are under a mile" would add a little strength at a real cost in simplicity, so it stays out. (The uranium case gets in because it follows from physics already in the system: Carroll reading, outside the lecture.)

It also **reduces** laws, and with them nomological modality, to something **non-modal**: the mosaic. Laws are no longer an irreducible feature of reality.

> [!definition] The hallmark of Humeanism
> The relation between laws and world is **reversed**. For non-Humeans the laws govern, constrain or determine the world. For Humeans **the world determines the laws**.

**Advantages.**

1. **Nothing mysterious**: no mysterious entity (universal, disposition, primitive law) and no mysterious notion of governance. Laws are **effective summaries** of things we are familiar with, local matters of fact.
2. **A natural link with science.** The BSA's balancing of competing summaries is an idealised version of what science actually does. Non-Humean accounts have no such link: nothing stops the laws that govern from **differing from the equations** our theories discover.

**Problems.**

- Not all physical theories fit the template of laws fixed by a mosaic of **local** facts. **Quantum entanglement** is the famous case: entangled particles have joint properties that are not fixed by the properties of each particle at its own location.
- From Chen's reading: the patterns in the mosaic are **ultimately unexplained**, since the laws only summarise them. And what counts as "best" may depend on us, making laws mind-dependent (Lewis's "ratbag idealism").

## The accounts side by side

| | **Laws** | **Gain** | **Cost** |
|---|---|---|---|
| **Primitivism** | Govern by constraint | Fits any law | Unanalysed |
| **DTA** | Link universals | Says what laws are | Constraint laws |
| **Dispositions** | Come from powers | Says what laws are | Mysterious |
| **Humean BSA** | Summarise the mosaic | No mystery | Entanglement |

The core of the debate is metaphysical: what laws themselves are. A topic the lecture leaves open is laws in the **special sciences**. (Background from the Carroll reading: such laws usually hold only **ceteris paribus**, "other things being equal". The law of supply and demand fails when prices are regulated, so it seems to need that clause.)

## What to be able to write

- Why a true universal generalisation is not enough: gold spheres, and the coin counterfactual.
- How laws support counterfactuals (similarity between worlds, Lewis and Stalnaker), and the counterfactual account of causation.
- Ontology plus dynamics; laws select allowed histories; nomological possibility and necessity.
- Determinism in plain and possible-worlds terms; Newton, general relativity, collapse; strong determinism.
- Maudlin's FLOTEs, and the two objections (time symmetry, constraint laws).
- The Humean/non-Humean definitions.
- Each non-Humean account: its core claim, why it is non-Humean, its advantage and problem.
- The BSA definition, the mosaic, the reversed relation, its two advantages and the entanglement problem.

## Exam questions

> [!exam]- Practice question: Why is it not enough to say that a law of nature is a true universal generalisation? Use an example, and explain how counterfactuals help draw the line.
> **Key points:** Accidental generalisations are true universal statements; gold spheres under a mile; laws force truth, accidents happen to be true; laws support counterfactuals by fixing a crucial respect of similarity between worlds; accidents do not (coin in pocket).
>
> - **The proposal:** a law is a true statement "all Fs are Gs".
> - **Counterexample:** **accidental generalisations**, true universal statements whose truth is accidental. "All gold spheres are less than one mile in diameter" is true, but no law of nature forces it; it just happens to be the case. (Contrast "all uranium spheres are less than a mile": plausibly a law, because of critical mass.)
> - So accounts of laws add conditions to universal truth until only laws satisfy them, trying to capture how laws **make** things true.
> - **Counterfactuals** ("if A were to happen, B would happen") are modal: claims about possible worlds. On the Lewis-Stalnaker account, one is true when worlds where A and B hold are more similar to ours than worlds where A holds without B.
> - **Laws support counterfactuals** because they fix a crucial respect of similarity: "if this match had been struck, it would have lit" is true as an instance of a law. If every coin in my pocket happens to be silver, "if this copper coin were in my pocket, it would be silver" is still false: the generalisation is accidental.
> - So a generalisation that supports counterfactuals behaves like a law; one that does not is an accident.

> [!exam]- Practice question: Explain the contrast between Humean and non-Humean accounts of laws of nature, illustrating it with Lewis's Best Systems Account and one non-Humean account of your choice. Which do you find more convincing?
> **Key points:** Non-Humeans: laws govern, irreducible nomological modality; Humeans: laws do not govern, modality reducible; BSA: axioms of the system with best balance of strength and simplicity for the mosaic of local matters of fact; world determines laws; a non-Humean account with its advantage and problem; argued verdict.
>
> - **Definitions:** non-Humeans hold that laws **govern** the world and carry an **irreducible** nomological modality; Humeans deny both and reduce nomological modality to **non-modal** facts. Background: Hume's "no necessary connections between distinct entities".
> - **BSA (Lewis; Mill-Ramsey-Lewis):** laws are the axioms of the system with the **best balance of strength and simplicity** in accounting for the **Humean mosaic**, the spatiotemporal arrangement of local matters of fact ("a particle of mass $m$ is here"). Example: a full list of facts is too complex, "things exist" too weak, Maxwell's equations plus the Lorentz force law strike the balance. Hallmark: **the world determines the laws**.
> - **Non-Humean example, minimal primitivism (Chen):** laws are primitive facts with no restriction on their form; they govern by **constraining** the possibilities, as axioms govern mathematics. Minimal: the least irreducible modality needed to be non-Humean, no requirement of temporal evolution.
> - **For Humeanism:** no mysterious entity or governance; mirrors the balancing science actually does; non-Humean laws could differ from the equations science finds.
> - **Against:** entanglement does not fit a mosaic of local facts; patterns in the mosaic are left unexplained. Against primitivism: it explains nothing about what laws are.
> - **Verdict (one option):** primitivism, since physics' laws include non-local and constraint laws that the mosaic template struggles with, at the price of leaving "governing" unanalysed.

> [!exam]- Practice question: Explain the Dretske-Tooley-Armstrong account of laws and the dispositionalist account. In what sense are both reductive, and what advantage and problems do they share compared with primitivism?
> **Key points:** DTA: laws are necessitation relations between universals, "necessarily, if F then G"; universals multilocated (redness); dispositionalism: laws derived from dispositions via causation (window, Demarest); both non-Humean through irreducible modality; both reductive, so more explanatory than primitivism; shared problems: mysterious entities, laws of modern physics that do not fit.
>
> - **Universal:** the abstract entity encoding a property in full generality; every red thing instantiates redness, so universals are multilocated.
> - **DTA:** laws are **relations of necessitation between universals**: "necessarily, if F then G", $N(F,G)$. $F = ma$: necessarily, if force is such and such, mass and acceleration are such and such. Non-Humean insofar as this modality is irreducible.
> - **Dispositionalism (Chen: Aristotelian reductionism):** a disposition is a property like "glass breaks when it falls", realised under a contextual clause. Causation via dispositions: the stone broke the window because windows are disposed to break when hit. Laws recovered from causation, e.g. as the best summary of patterns of causal relations (Demarest). Non-Humean insofar as dispositions are **modal** (a tendency present before it is actualised) and **irreducible** (not explained by a non-dispositional property at the fundamental level).
> - **Shared advantage:** both are **reductive**, analysing laws in terms of something more fundamental (universals, dispositions), so both say what laws are, which primitivism refuses to do.
> - **Shared problems:** the posited entities are as mysterious as primitive laws; modern physics may not fit (constraint laws like the Einstein equations and path integrals for DTA; conservation laws and least-action principles for dispositions).

> [!exam]- Practice question: How are laws of nature connected to physical possibility and to determinism? Explain nomological necessity, determinism and strong determinism, with examples from physics.
> **Key points:** Theory gives ontology plus dynamics (laws); state space and histories; laws select allowed histories (worlds); nomological possibility and necessity; determinism: segment fixes the whole world; Newton deterministic, GR only globally hyperbolic models, collapse QM not, chance ineliminable; strong determinism: only one world compatible with the laws, no real examples.
>
> - A theory specifies an **ontology** (entities matching its variables) and **dynamics** (the laws). In **state space** each point is a possible state, and each trajectory a history.
> - **Laws select which histories are allowed.** Treating histories as possible worlds: true in some allowed worlds is **nomologically possible**, true in all is **nomologically necessary** (nomological modality).
> - **Determinism:** the state at one time plus the laws fixes the state at all times; in modal terms, a segment of a nomologically possible world belongs to only one such world.
> - **Examples:** Newtonian mechanics is deterministic. In general relativity only **globally hyperbolic** models are (state at one time plus the Einstein equations fix all times), so its determinism depends on whether other models are physically significant. Quantum mechanics on collapse interpretations is not deterministic: **chance** enters the laws ineliminably.
> - **Strong determinism** (Chen, after Penrose): only **one** world is compatible with the laws, so the theory says there is only one possible world. No real examples (string theory does not qualify), but a live conceptual option since we do not know the fundamental laws.

> [!exam]- Practice question: Maudlin holds that laws of nature are Fundamental Laws of Temporal Evolution. Explain this view and the objections to it, and show how Chen's minimal primitivism avoids them.
> **Key points:** FLOTEs: laws produce later states from earlier ones, with a direction of time; objection: laws mostly time-symmetric, asymmetry emergent (thermodynamics); objection: some laws are constraints (Einstein equations); minimal primitivism: laws govern by constraining possibilities, like axioms, with no restriction on form.
>
> - **Maudlin:** laws necessarily involve evolution in time, temporal asymmetry and dynamic evolution; they **produce later states of the world from earlier ones**. Laws are **FLOTEs** ($F = ma$, Schrödinger's equation). This is a heavyweight primitivism.
> - **Objection 1:** most laws of physics are **time-symmetric** (a reversed history is also allowed). Temporal asymmetry is **emergent**, e.g. from thermodynamics (for Chen, from the low-entropy Past Hypothesis).
> - **Objection 2:** not all laws involve dynamical evolution. The **Einstein field equations** are **constraints** on whole spacetimes (also Gauss's law, least-action principles).
> - **Chen's minimal primitivism:** laws are primitive facts that govern **by constraining** the space of possibilities, as axioms govern mathematics by fixing what is allowed. No restriction on their form, no direction of time presupposed. So time-symmetric and constraint laws fit.
> - It stays **non-Humean**: the constraint comes from the laws and is irreducible. And it is **minimal**: the least irreducible modality needed to be non-Humean, with no FLOTE requirement.

## Flashcards

> [!card]- Why did early modern thinkers say laws of nature "govern" the world?
> To keep **God in control**: God fixes the laws, the laws run a deterministic world.

> [!card]- What is Hume's slogan against the reality of laws?
> "**No necessary connections between distinct entities.**"

> [!card]- Which laws does the debate on laws of nature mainly concern?
> The **fundamental laws of physics**, not special-science laws.

> [!card]- Why must the study of fundamental laws be suppositional?
> We have **no candidate final theory** and do not know one exists.

> [!card]- Why is "a law is a true universal generalisation" not enough?
> Some true universal generalisations are **accidental**.

> [!card]- What is an accidental generalisation?
> A true universal generalisation whose truth is **completely accidental**: nothing forces it.

> [!card]- Give the standard example of an accidental generalisation.
> "All **gold spheres** are less than one mile in diameter."

> [!card]- What distinguishes "all uranium spheres are under a mile" from the gold version?
> It is plausibly a **law**: critical mass prevents such spheres; nothing prevents a big gold sphere.

> [!card]- What do accounts of laws try to capture, beyond universal truth?
> The sense in which laws **force** something to be true, rather than it just happening to be true.

> [!card]- What is a counterfactual?
> A statement "**if A were to happen, B would happen**".

> [!card]- Why are counterfactuals called modal statements?
> They concern **possibility and necessity**: equivalently, claims about **possible worlds**.

> [!card]- Why is "all worlds where A holds also have B" too strong for counterfactuals?
> Some distant A-worlds lack B (wet match, no oxygen), yet the counterfactual is still true.

> [!card]- When is "if A were to happen, B would happen" true, for Lewis and Stalnaker?
> When **A-and-B worlds** are **more similar to ours** than A-without-B worlds.

> [!card]- Why do laws support counterfactuals?
> They fix a **crucial respect of similarity** between worlds.

> [!card]- Why is "if this coin were in my pocket, it would be silver" false, even if all my pocket's coins are silver?
> "All coins in my pocket are silver" is **accidental**, so supports no counterfactual.

> [!card]- What makes "if this match had been struck, it would have lit" true?
> It is an **instance of a law**.

> [!card]- What is the counterfactual account of causation?
> Two events are causally related if there is **counterfactual dependence** between them.

> [!card]- Who used the link between counterfactuals and laws to analyse causation?
> **Hume**, and later **Lewis**.

> [!card]- What two things does a scientific theory specify, on a realist reading?
> An **ontology** (entities matching its variables) and **dynamics** for them: the **laws**.

> [!card]- What is a state space?
> A space where **every point is a state** of the theory's entities.

> [!card]- What does a trajectory in state space represent?
> A **history** of the theory's entities.

> [!card]- What do laws do to the histories in state space?
> **Select only some histories as allowed.**

> [!card]- When is something nomologically possible?
> When it is true in **some** law-allowed possible world.

> [!card]- When is something nomologically necessary?
> When it is true in **all** law-allowed possible worlds.

> [!card]- What is nomological modality?
> The **possibility and necessity associated with laws of nature**.

> [!card]- What is determinism, roughly?
> Knowing the **state at one time** and **the laws**, you know the state at **any other time**.

> [!card]- How is determinism stated in possible-worlds language?
> Given a segment of a nomologically possible world, **only one** full nomologically possible world contains it.

> [!card]- Is Newtonian mechanics deterministic?
> **Yes**: it is the classic deterministic theory.

> [!card]- Which models of general relativity are deterministic?
> Only **globally hyperbolic** ones: state at one time plus the Einstein equations fix all times.

> [!card]- Why is it open whether general relativity is deterministic?
> It depends on whether **non-globally hyperbolic models** count as **physically significant**.

> [!card]- Is quantum mechanics deterministic?
> **Not on collapse interpretations.**

> [!card]- What follows if a theory fails to be deterministic?
> It involves **chance ineliminably**: probabilities are part of the laws' content.

> [!card]- What is strong determinism (Chen, after Penrose)?
> **Only one world** is compatible with the laws.

> [!card]- How is strong determinism stronger than ordinary determinism?
> Determinism needs a segment as input; strong determinism fixes the world with **no input**.

> [!card]- Is there a real example of a strongly deterministic theory?
> **No.** String theory was hoped to be, but is not on any reasonable reading.

> [!card]- What does Maudlin take laws of nature to be?
> **Fundamental Laws of Temporal Evolution (FLOTEs)**: they produce later states from earlier ones.

> [!card]- Why does time symmetry count against Maudlin's view of laws?
> Most physical laws are **time-symmetric**; temporal asymmetry is **emergent**, e.g. from thermodynamics.

> [!card]- Why do the Einstein field equations count against Maudlin's view of laws?
> They are **constraints**, not laws of dynamical evolution.

> [!card]- What do non-Humeans hold about laws?
> Laws **govern**, with an **irreducible nomological modality**.

> [!card]- What do Humeans hold about laws?
> Laws **do not govern**; nomological modality **reduces to non-modal** facts.

> [!card]- Name the three non-Humean accounts of laws.
> **Primitivism**, **universals** (DTA), **dispositionalism**.

> [!card]- What does it mean to call laws primitive?
> They are **not analysed** in terms of anything else: conceptual **rock-bottom**.

> [!card]- What is the trade-off in calling laws primitive?
> Give up **analysing** laws to avoid the **complicated metaphysics** of any analysis.

> [!card]- How do laws govern on Chen's minimal primitivism?
> By **constraining** the possibilities admissible for the theory's entities.

> [!card]- What analogy explains governing-by-constraint?
> **Mathematical axioms**: they fix what is allowed, without directing any evolution.

> [!card]- Why is minimal primitivism genuinely non-Humean?
> The restriction comes **from the laws** and is **irreducible**.

> [!card]- Why is Chen's primitivism called minimal?
> The **least irreducible modality** needed to be non-Humean; **no restriction on the form** of laws.

> [!card]- What is the difference between Chen's and Maudlin's primitivism?
> Maudlin requires laws to be **FLOTEs**; Chen sets **no condition** on their form.

> [!card]- What role does simplicity play for laws on minimal primitivism?
> An **epistemic guide** to finding laws, not part of what makes something a law.

> [!card]- What is a universal?
> The **abstract entity** encoding a property **in full generality**, e.g. redness.

> [!card]- Why are universals said to be multilocated?
> Every instance of the property, e.g. every red thing, is **an instance of the one universal**.

> [!card]- What is a law on the Dretske-Tooley-Armstrong account?
> A **relation of necessitation between universals**: "necessarily, if F then G".

> [!card]- Why is the DTA account more explanatory than primitivism?
> It is **reductive**: it analyses laws in terms of **universals**, so says what laws are.

> [!card]- Which laws are hard to fit into the DTA schema?
> **Constraint** laws like the **Einstein equations**, and **path integrals**.

> [!card]- What is a disposition?
> A property like "**glass breaks when it falls**", realised under a specific contextual clause.

> [!card]- How does dispositionalism explain why a stone broke a window?
> **Windows have a disposition to break when hit by stones.**

> [!card]- How does Demarest recover laws from dispositions?
> Laws are the **best summary of patterns of causal relations**.

> [!card]- Why are dispositions intrinsically modal?
> A disposition is present **before** it is actualised: it encodes a tendency.

> [!card]- What does it mean that dispositions are irreducible?
> They are **not analysed further** in terms of non-dispositional properties, at the fundamental level.

> [!card]- What advantage do dispositionalism and DTA share over primitivism?
> Both are **reductionist about laws**, so they explain what laws are.

> [!card]- What problem do dispositions share with universals and primitive laws?
> They are **mysterious**.

> [!card]- What is Lewis's Best Systems Account of laws?
> Laws are the **axioms** of the system with the **best balance of strength and simplicity** for the Humean mosaic.

> [!card]- What is the Humean mosaic?
> The **spatiotemporally related local matters of fact**: the bedrock that determines the laws.

> [!card]- What is the Best Systems Account also called?
> The **Mill-Ramsey-Lewis** account.

> [!card]- Why would "all gold spheres are under a mile" not be a law on the Best Systems Account?
> It adds **little strength** at a **cost in simplicity**, so it is not in the best system.

> [!card]- What is the hallmark of the Humean account of laws?
> The relation is **reversed**: the **world determines the laws**, not the laws the world.

> [!card]- What is the Humean account's first advantage over non-Humean accounts?
> **No mysterious entity** or mysterious notion of governance.

> [!card]- What is the Humean account's link to empirical science?
> Balancing competing summaries is an **idealised version of what science does**.

> [!card]- Why do non-Humean accounts lack the Humean link to empirical science?
> Nothing stops governing laws from **differing from the equations science discovers**.

> [!card]- What is the most famous problem for Humeanism?
> **Quantum entanglement**: not all theories fit laws fixed by a mosaic of **local** facts.

> [!card]- True or false: on dispositionalism, the laws themselves govern the world.
> **False.** The modal force lives in the **dispositions**; laws are derived from them (Chen).

## Links

- **Course:** [[PhilSci - Overview|Course overview]] · [[PhilSci - Tutorials|Tutorials]]
- **Earlier lectures:** [[PhilSci-L01 - Introduction and Logical Empiricism]] (empiricism, the tradition Hume belongs to) · [[PhilSci-L01b - Popper and Lakatos]] (Hume's problem of induction, and why no observations can verify a universal law) · [[PhilSci-L04 - Scientific Explanation and Understanding]] (Hempel's covering law model, which needs laws in every explanation) · [[PhilSci-L05 - Scientific Realism and its Critiques]] (scientific realism, behind reading a theory as ontology plus laws) · [[PhilSci-L06 - Scientific Realism and the Pessimistic Meta-Induction]] (the previous lecture)
- **Source:** "Lecture 7: Laws of Nature", Enrico Cinti, 13 October 2026 (Canvas file `7 Laws of Nature`, `Assets/7 Laws of Nature.pdf`).
- **Reading:** Eddy Keming Chen, *Laws of Physics* (Cambridge Elements in Philosophy of Physics, 2024), only sections 1-2.4, 3.1-3.2, 4.1-4.3. Not on Canvas (library); used here from the arXiv preprint 2309.03484 (September 2023), whose section numbers match: introduction and the Great Divide section 1, ontology, nomology and possibility 2.1, counterfactuals and causation 2.2, dynamic production and the direction of time (FLOTEs, Past Hypothesis) 2.3, determinism, chance and strong determinism 2.4, minimal primitivism and the mystery objection 3.1, explanation by simple constraint (Mandelbrot world) 3.2, Humean reductionism (the three systems, natural properties, ratbag idealism) 4.1, Platonic reductionism (DTA) 4.2, Aristotelian reductionism (dispositions, Demarest) 4.3.
- **Optional reading:** John W. Carroll, "Laws of Nature", *Stanford Encyclopedia of Philosophy* (revised 2020): gold and uranium spheres section 1, systems section 2, universals and Lewis's objection section 3, special-science and ceteris paribus laws section 10.
