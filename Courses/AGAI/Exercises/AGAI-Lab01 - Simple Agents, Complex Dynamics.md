---
type: exercise
course: AGAI
week: 1
source: "Lab 1 notebook (werkcollege, 28 Oct 2026)"
concepts:
  - "[[Nash Equilibrium]]"
  - "[[Pareto Optimality]]"
  - "[[Prisoner's Dilemma]]"
  - "[[Stag Hunt]]"
  - "[[Repeated Game]]"
  - "[[Replicator Dynamics]]"
  - "[[Reward Hacking]]"
  - "[[Value Iteration]]"
  - "[[Markov Decision Process]]"
  - "[[Agent Program]]"
status: complete
---

# AGAI-Lab01: Simple Agents, Complex Dynamics

> [!abstract] In short
> The lab uses agents with almost no memory and two or three actions to show three things. In one-shot games, self-interest gives the [[Prisoner's Dilemma]] mutual defection and leaves the [[Stag Hunt]] stuck between two equilibria, and in repeated games, memory lets cooperation pay. When agents adapt by copying what works instead of calculating, the [[Replicator Dynamics|replicator equation]] predicts where a population ends up, and the same tools explain when signals between agents stay honest. Finally, a perfect optimiser in a two-square vacuum world shows how a badly chosen reward turns a harmless new action into [[Reward Hacking]].
>
> The lab is ungraded practice that the lecturer calls the best preparation for the midterm. All fifteen questions are worked below with derivations. Numbers from random simulations come from a re-run of the same models in my own code, and may differ in the second decimal from the classroom notebook. Analytic results (equilibria, thresholds, rest points, stability, policies) are exact.

**Concepts tested:** [[Nash Equilibrium]], [[Pareto Optimality]], [[Prisoner's Dilemma]], [[Stag Hunt]], [[Repeated Game]], [[Replicator Dynamics]], [[Reward Hacking]], [[Value Iteration]], [[Markov Decision Process]], [[Agent Program|simple reflex agents]]

---

## Part I: one-shot games, beliefs and repeated games

### The model

Every game in Part I is a **symmetric two-player, two-action game**. Each player picks C (cooperate) or D (defect) at the same time. Four numbers give the row player's payoff, and the column player is paid the same way:

| | partner plays C | partner plays D |
|---|---|---|
| **play C** | $R$ (reward) | $S$ (sucker) |
| **play D** | $T$ (temptation) | $P$ (punishment) |

Three ways to judge an outcome (a pair of actions):

- A **best response** is the action that maximises my payoff given what my partner does.
- A **pure [[Nash Equilibrium]]** is an outcome where each player's action is a best response to the other's, so no player gains by changing its action alone. It describes where self-interested players can end up.
- An outcome is **[[Pareto Optimality|Pareto optimal]]** if no other outcome makes one player better off without making the other worse off. It means "not wasteful".
- The **social optimum** maximises the sum of the two payoffs (total welfare). Every social optimum is Pareto optimal; the converse fails.

With only four outcomes, all of these can be found by checking each cell. The two games used throughout:

- **[[Prisoner's Dilemma]] (PD):** $T > R > P > S$, here $R=3, S=0, T=5, P=1$.
- **[[Stag Hunt]] (SH):** $R > T > P > S$, here $R=4, S=0, T=3, P=2$. A stag (C) is caught only if both hunt it; a hare (D) can be caught alone. It stands for any coordination problem where a risky joint action pays most and a safe solo action tempts, such as two coding agents that must coordinate edits to shared files to ship a big feature.

**Beliefs.** An agent unsure about its partner holds a belief $p$, the probability that the partner cooperates. Its expected payoffs are

$$\pi_C(p) = pR + (1-p)S, \qquad \pi_D(p) = pT + (1-p)P.$$

Setting $\pi_C(p) > \pi_D(p)$ and solving for $p$:

$$p(R - T) > (1-p)(P - S) \;\Longleftrightarrow\; p > p^* = \frac{P - S}{(R - T) + (P - S)}$$

where $P - S$ is what you lose by cooperating when the partner does not (fear) and $R - T$ is what you gain by cooperating when the partner also does. The division keeps the direction of the inequality only when $(R-T)+(P-S) > 0$, which holds in a coordination game like the SH. The larger $p^*$, the more trust hunting the stag needs.

**Repeated games.** In a [[Repeated Game]] the same two agents play the PD for several rounds, and a strategy maps the history to an action. Memory-0 strategies ignore the history; memory-1 strategies look at the last round. A memory-one strategy is five probabilities of cooperating: in the first round, and after each previous outcome (my last action, partner's last action):

| strategy | first | after CC | after CD | after DC | after DD |
|---|---|---|---|---|---|
| AllC | 1 | 1 | 1 | 1 | 1 |
| AllD | 0 | 0 | 0 | 0 | 0 |
| Random | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| TFT (Tit-for-Tat) | 1 | 1 | 0 | 1 | 0 |
| WSLS (Win-Stay Lose-Shift) | 1 | 1 | 0 | 0 | 1 |

TFT copies the partner's last action. WSLS repeats its last action after a "win" (3 or 5 points, which happens exactly when the partner cooperated) and switches after a "loss" (0 or 1). Memory-0 strategies are the special case with the last four numbers equal.

The tournament is a round robin: every strategy plays every strategy, including a copy of itself, each pairing repeated 50 times, and a strategy's score is its average payoff per round over all opponents. The notebook frames it after Axelrod's tournaments: in 1980 Robert Axelrod, a political scientist at the University of Michigan, asked game theorists to submit strategies, paired each with each for 200 rounds of the PD, and Tit-for-Tat, submitted by Anatol Rapoport, won. A `noise` parameter flips an agent's intended action with some probability (an execution error); every question below uses noise 0.

### Q1: predictions before the analysis

> [!question] Statement
> (a) You play the following game **once**, against an anonymous agent you will never meet again. Both choose simultaneously. Your payoff (row player): you cooperate and the other cooperates, 3; you cooperate and the other defects, 0; you defect and the other cooperates, 5; you defect and the other defects, 1. The other agent receives points the same way (a normal-form game). Which action would *you* choose? Which action would a purely self-interested agent choose?
>
> (b) The same game is played **50 times** against the same partner, and each agent sees what happened in the previous round. Five types enter a round-robin tournament (everyone plays everyone, including a copy of itself): AllC (always cooperate), AllD (always defect), Random (fair coin each round), TFT (cooperate first, then copy the partner's previous action), WSLS (cooperate first, then repeat the previous action if it earned 3 or 5 points, switch otherwise). Which type do you expect to obtain the **highest** average score per round, and which the **lowest**?

**(a)** Many people pick C, out of fairness or hope. A purely self-interested agent picks D, because D pays more whatever the other does: $5 > 3$ if the other cooperates, $1 > 0$ if it defects. D is a dominant action.

**(b)** The tempting prediction, from the Axelrod story, is that TFT wins and AllD loses. A careful prediction looks at *this* field: two of the five entrants (AllC and Random) can be exploited every round, and WSLS can be exploited every other round. AllD collects 5 per round from AllC and about 3 from Random and from WSLS, so it may well stay on top, while AllC, which never punishes, should come last. The run (Q6) confirms this: AllD 2.62, TFT 2.45, WSLS 2.34, Random 2.24, AllC 2.11 points per round.

### Q2: Nash, Pareto and social optimum in the PD

> [!question] Statement
> The code finds, for the PD with $R=3, S=0, T=5, P=1$: pure Nash equilibrium (D,D); Pareto-optimal outcomes (C,C), (C,D), (D,C); social optimum (C,C).
>
> (a) Is the Nash equilibrium of the Prisoner's Dilemma Pareto optimal? Is it the social optimum? Explain why.
>
> (b) Using the idea of a best response, explain in one or two sentences why two self-interested agents end up at the Nash equilibrium even though both prefer (C,C). Hint: does the best response of the row player depend on what the column player does?
>
> (c) (C,D) is Pareto optimal. Does that make it a "good" outcome? What does this tell you about the difference between Pareto optimality and the social optimum?

The game as (row, column) payoffs, with total welfare:

| | col C | col D |
|---|---|---|
| **row C** | (3, 3), welfare 6 | (0, 5), welfare 5 |
| **row D** | (5, 0), welfare 5 | (1, 1), welfare 2 |

**(a)** No to both. (C,C) gives each player 3 instead of 1, so it makes both better off and (D,D) is Pareto dominated. Its welfare is 2, the lowest of the four cells, against 6 at the social optimum (C,C).

**(b)** The row player's best response is D whether the column player plays C ($5 > 3$) or D ($1 > 0$): it does not depend on the partner at all. The same holds for the column player, so each plays D, and (D,D) is the only cell where both are best-responding. Preferring (C,C) jointly does not help, because at (C,C) each player gains 2 by switching alone.

**(c)** (C,D) is Pareto optimal only because any move away from it hurts the column player (who gets 5, the most anyone can get). It is unequal (0 against 5) and wastes welfare (5 against 6). Pareto optimality rules out outcomes where someone can gain for free; it says nothing about total welfare or fairness. The social optimum picks the single outcome with the largest sum, so it is a much stronger requirement: of the three Pareto-optimal outcomes, only (C,C) is socially optimal.

### Q3: two games, two motives

> [!question] Statement
> The code finds, for the SH with $R=4, S=0, T=3, P=2$: pure Nash equilibria (C,C) and (D,D); Pareto optimal (C,C); social optimum (C,C).
>
> (a) Compare the Stag Hunt with the Prisoner's Dilemma: how many pure Nash equilibria does each game have? In which game is mutual cooperation an equilibrium?
>
> (b) Two motives can push a player towards D: **greed** (defecting pays more than cooperating when the partner cooperates, $T > R$) and **fear** (defecting pays more than cooperating when the partner defects, $P > S$). Which of these motives is present in the PD? Which in the SH? Use this to explain why the SH has a problem of *trust/coordination* while the PD has a problem of *incentives*.

**(a)** The PD has one pure Nash equilibrium, (D,D). The SH has two, (C,C) and (D,D). Mutual cooperation is an equilibrium only in the SH: at (C,C) a stag hunter who switches to hare drops from 4 to 3, so nobody wants to deviate. In the PD a cooperator facing a cooperator gains by switching ($5 > 3$).

**(b)**

| motive | condition | PD (3, 0, 5, 1) | SH (4, 0, 3, 2) |
|---|---|---|---|
| greed | $T > R$ | yes, $5 > 3$ | no, $3 < 4$ |
| fear | $P > S$ | yes, $1 > 0$ | yes, $2 > 0$ |

In the PD both motives point to D, so D is dominant: even an agent *certain* that its partner will cooperate defects. No amount of trust fixes that; the payoffs themselves would have to change (punishment, contracts, repetition). That is an incentive problem. In the SH there is no greed, so an agent sure that its partner hunts stag hunts stag too. Only fear, doubt about the partner, pushes it to D. Fix the beliefs and cooperation follows, so the problem is trust and coordination.

### Q4: which Stag Hunt equilibrium? (prediction)

> [!question] Statement
> Two agents meet once to play the Stag Hunt ($R=4, S=0, T=3, P=2$). They cannot communicate and know nothing about each other.
>
> (a) Which outcome do you expect to be most frequent: (C,C), (D,D), or a miscoordinated outcome?
>
> (b) Suppose the hare becomes more attractive: $T$ increases from 3 to 4, everything else fixed. Will (C,C) become more or less frequent? Why?

**Careful prediction.** With no information, the natural model is the one the lab then uses: each agent's belief $p$ is uniform on $[0, 1]$, and each plays its best response. Here

$$p^* = \frac{2 - 0}{(4-3) + (2-0)} = \frac{2}{3}.$$

An agent hunts stag only if $p > 2/3$, which happens with probability $1/3$. Independently for the two agents:

$$\Pr(C,C) = \left(\tfrac13\right)^2 = \tfrac19 \approx 0.111, \quad \Pr(D,D) = \left(\tfrac23\right)^2 = \tfrac49 \approx 0.444, \quad \Pr(\text{miscoordinated}) = 2 \cdot \tfrac13 \cdot \tfrac23 = \tfrac49 \approx 0.444.$$

**(a)** (C,C) is the least frequent outcome by far. (D,D) and miscoordination are equally likely in expectation, each $4/9$. The simulation of 10,000 encounters gives (C,C) 0.110, (D,D) 0.446, miscoordinated 0.445.

**(b)** Less frequent, in fact never. With $T = 4$: $p^* = 2 / (0 + 2) = 1$, and no belief exceeds 1, so nobody hunts stag: (C,C) 0.000, (D,D) 1.000. A larger $T$ shrinks the gain from cooperating with a cooperator ($R - T$ drops from 1 to 0) while the fear term stays at 2. At $T = R = 4$ the game is no longer a strict Stag Hunt: hare pays the same as stag against a stag hunter and more against a hare hunter, so D weakly dominates C.

### Q5: a rule for which equilibrium wins

> [!question] Statement
> (a) Find a simple rule, written only in terms of $R, S, T, P$, that predicts which equilibrium is played more often.
>
> (b) Consider this sentence: "Rational agents will always coordinate on (C,C), because it is better for both of them." Which evidence from this section contradicts this, and what is the flaw in the argument?
>
> (c) The results depend on how we modelled what agents believe before playing. Describe one situation (for example, related to communication or memory) in which you would expect a different outcome.

**(a)** Under uniform beliefs, $\Pr(C,C) = (1 - p^*)^2$ and $\Pr(D,D) = (p^*)^2$. So (C,C) is played more often exactly when

$$1 - p^* > p^* \iff p^* < \tfrac12 \iff P - S < (R - T)\ \text{ i.e. }\ R - T > P - S.$$

**Rule:** (C,C) is more frequent when the gain from cooperating with a cooperator, $R - T$, exceeds the loss from cooperating with a defector, $P - S$; (D,D) is more frequent when fear outweighs gain. (In game theory, outside the lab, the equilibrium that wins this comparison is called *risk dominant*, a term from Harsanyi and Selten.)

My re-run checks the rule on four games:

| game $(R,S,T,P)$ | $R-T$ vs $P-S$ | $p^*$ | (C,C) | (D,D) | predicted winner |
|---|---|---|---|---|---|
| (4, 0, 3, 2), the lab's SH | 1 < 2 | 0.667 | 0.110 | 0.446 | (D,D) |
| (5, 0, 2, 1) | 3 > 1 | 0.250 | 0.573 | 0.065 | (C,C) |
| (4, 0, 3, 1) | 1 = 1 | 0.500 | 0.252 | 0.251 | tie |
| (4, 0, 4, 2) | 0 < 2 | 1.000 | 0.000 | 1.000 | (D,D) |

**(b)** The baseline run: rational best-responders reach (C,C) in only 11% of encounters and (D,D) in 45%. The argument confuses "better for both if both do it" with "best for me given what I expect". Each agent chooses alone; (C,C) being Pareto superior does not make stag the best response unless the agent believes its partner hunts stag with probability above $p^* = 2/3$. With no reason for that belief, the safe hare wins.

**(c)** Any change that makes beliefs high and correlated changes the result. Examples:
- **Communication:** if agents can say "I will hunt stag" before playing, the message is self-enforcing in the SH: if the partner believes it and hunts stag, hunting stag is then the sender's own best response, so keeping the promise pays. Beliefs jump above $p^*$ and (C,C) becomes common.
- **Memory and repetition:** agents who meet repeatedly and remember past rounds update $p$ from experience. After a few successful stag hunts, beliefs settle near 1 and (C,C) locks in; after a few failures they settle near 0. Part II models exactly this adaptive version.
- **Reputation or shared conventions:** a partner known to hunt stag, or agents built by the same company, start with $p$ near 1.

### Q6: one-shot against repeated tournament

> [!question] Statement
> The tournament of Q1(b) is run twice, with 1 round and with 50 rounds.
>
> (a) Compare the two rankings. How did the scores of AllD, TFT and WSLS change when going from 1 to 50 rounds? Was your prediction in Q1(b) correct? In the one-shot game AllC, TFT and WSLS take exactly the same action, yet their scores differ slightly: why, and what does this tell you about how to read small differences in these rankings?
>
> (b) Look at the **cooperation-rate** matrix. Which strategies cooperate at *different* rates depending on who they play against, and which do not? Relate this to the memory of each strategy. Why is this ability essential for cooperation to pay off against a defector?
>
> (c) AllD may still score well in this strategy space. Using the payoff matrix, identify *whose* payoff AllD's score mostly comes from. Quote the numbers.

Rankings (points per round), with the exact expected values I derived by hand alongside:

| rank | one-shot (1 round) | exact | repeated (50 rounds) | exact |
|---|---|---|---|---|
| 1 | AllD 3.82 | 3.80 | AllD 2.62 | 2.616 |
| 2 | Random 2.96 | 2.95 | TFT 2.45 | 2.443 |
| 3 | AllC 2.12 | 2.10 | WSLS 2.34 | 2.347 |
| 4 | WSLS 2.05 | 2.10 | Random 2.24 | 2.264 |
| 5 | TFT 2.02 | 2.10 | AllC 2.11 | 2.100 |

Score matrix for 50 rounds (row strategy's average payoff per round against the column strategy):

| | AllC | AllD | Random | TFT | WSLS |
|---|---|---|---|---|---|
| **AllC** | 3.00 | 0.00 | 1.56 | 3.00 | 3.00 |
| **AllD** | 5.00 | 1.00 | 3.01 | 1.08 | 3.00 |
| **Random** | 3.99 | 0.50 | 2.22 | 2.23 | 2.26 |
| **TFT** | 3.00 | 0.98 | 2.25 | 3.00 | 3.00 |
| **WSLS** | 3.00 | 0.50 | 2.19 | 3.00 | 3.00 |

**(a)** Going from 1 to 50 rounds, AllD falls from 3.82 to 2.62, TFT rises from 2.02 to 2.45 and WSLS from 2.05 to 2.34. In a single round TFT and WSLS are just cooperators and get exploited; with 50 rounds they stop feeding AllD after the first round (TFT) or every other round (WSLS), and they earn 3 per round from each other. Whether the Q1(b) prediction was right depends on what you wrote: AllD still wins this field and AllC is last, so "TFT wins" was wrong here.

In the one-shot game AllC, TFT and WSLS all play C, so their expected score is identical: $(3 + 0 + 1.5 + 3 + 3)/5 = 2.10$. The spread (2.12, 2.05, 2.02) comes only from the Random opponent's coin: each strategy met Random in a separate set of 50 games, and a payoff of 3 or 0 with equal chance has a standard deviation of 1.5, so the mean of 50 games has a standard error of about 0.21, or about 0.04 after averaging over the five opponents. Differences of a few hundredths in these rankings are noise; trust only gaps well above about 0.1, or derive the expected values.

**(b)** Cooperation rates in the 50-round run:

| | vs AllC | vs AllD | vs Random | vs TFT | vs WSLS |
|---|---|---|---|---|---|
| **AllC** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **AllD** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **Random** | 0.50 | 0.50 | 0.52 | 0.47 | 0.50 |
| **TFT** | 1.00 | 0.02 | 0.52 | 1.00 | 1.00 |
| **WSLS** | 1.00 | 0.50 | 0.53 | 1.00 | 1.00 |

AllC, AllD and Random cooperate at the same rate whoever they face (Random's small wiggles are sampling noise). They are memory-0: their action ignores the history, so they cannot react to the partner. TFT and WSLS are memory-1 and their rates depend on the partner: full cooperation with cooperators, 0.02 (only the first round) for TFT against AllD, 0.50 for WSLS against AllD.

This conditioning is what makes cooperation pay against a defector. An unconditional cooperator is exploited every round (AllC earns 0 against AllD). A strategy that withdraws cooperation after being defected on caps its losses at one round (TFT earns 0.98 against AllD) while still earning 3 per round with fellow cooperators. WSLS is weaker here: after mutual defection (1 point, a "loss") it switches back to C, so against AllD it alternates C, D, C, D and is exploited every other round (0.50).

**(c)** AllD's 2.62 is the average of 5.00 (against AllC), 1.00 (itself), 3.01 (Random), 1.08 (TFT) and 3.00 (WSLS). Of its 13.09 total, 11 points come from AllC, Random and WSLS: AllC hands it the temptation payoff $T = 5$ every round while getting $S = 0$; Random and WSLS each give it 5 about half the time (earning 0.50 themselves). Against the reciprocator TFT it gets 1.08, barely above mutual defection. AllD's score is paid for by exploitable partners, so it depends on how many of them are in the field.

### Q7: when does AllD stop being best?

> [!question] Statement
> In a strategy space with only AllD, TFT and WSLS: from how many rounds onwards do you expect AllD to stop being the best strategy? Give a number and a one-line reason.

**Answer: from 4 rounds onwards.** One-line reason: AllD's only profit is the first-round temptation against TFT and the alternating exploitation of WSLS, while TFT and WSLS earn 3 per round from each other; from 4 rounds the one-off gain is spread too thin.

**Derivation.** No noise, so everything is deterministic. With $n$ rounds, per-round payoffs in each pairing:

- AllD vs AllD: 1. AllD vs TFT: $5$ in round 1, then $1$, so $(n + 4)/n$. AllD vs WSLS: WSLS alternates C, D, C, ... (it switches after every loss), so AllD gets 5, 1, 5, 1, ..., a total of $5\lceil n/2 \rceil + \lfloor n/2 \rfloor$.
- TFT vs AllD: $0$ then $1$, so $(n-1)/n$. TFT vs TFT and TFT vs WSLS: 3 each (both cooperate forever).
- WSLS vs AllD: 0, 1, 0, 1, ..., so $\lfloor n/2 \rfloor / n$. WSLS vs TFT and vs WSLS: 3.

Tournament scores (average over the three opponents), multiplied by 3:

$$3\,\text{AllD} = \begin{cases} 5 + 4/n & n \text{ even} \\ 5 + 6/n & n \text{ odd} \end{cases} \qquad 3\,\text{TFT} = 7 - \frac1n \qquad 3\,\text{WSLS} = 6 + \frac{\lfloor n/2 \rfloor}{n}.$$

TFT is always at least WSLS, so AllD is best exactly when it beats TFT:
- $n$ even: $5 + 4/n > 7 - 1/n \iff 5/n > 2 \iff n < 2.5$;
- $n$ odd: $5 + 6/n > 7 - 1/n \iff 7/n > 2 \iff n < 3.5$.

So AllD wins for $n = 1, 2, 3$ and TFT wins from $n = 4$. The run agrees exactly:

| rounds | 1 | 2 | 3 | 4 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|---|
| AllD | 3.67 | 2.33 | 2.33 | 2.00 | 2.07 | 1.80 | 1.73 | 1.69 |
| TFT | 2.00 | 2.17 | 2.22 | 2.25 | 2.27 | 2.30 | 2.32 | 2.33 |
| WSLS | 2.00 | 2.17 | 2.11 | 2.17 | 2.13 | 2.17 | 2.17 | 2.17 |

Removing AllC and Random is what flips the 50-round result of Q6: without exploitable partners AllD falls to 1.69.

> [!tip] Beyond the questions: execution noise
> The notebook's `noise` parameter is 0 in every run. With 5% noise (my own extra run), two TFT players lose a lot: one accidental D starts an echo of alternating retaliation, and TFT's self-play payoff over 200 rounds drops to 2.26. WSLS recovers in two rounds: after an accidental (D,C), the defector won (stays D) and the victim lost (switches to D); at (D,D) both lose and both switch back to C. WSLS self-play stays at 2.77, and in the five-strategy field with noise WSLS (2.38) overtakes TFT (2.22).

> [!warning] Common pitfalls in Part I
> - **Pareto optimal is not the same as socially optimal.** (C,D) in the PD is Pareto optimal and is a bad outcome. Pareto only rules out waste that hurts nobody.
> - **"(C,C) is better for both, so rational agents play it."** Each agent best-responds to its own belief; in the SH stag is a best response only when the belief exceeds $p^*$, and in the PD never.
> - **Greed and fear.** The PD has both (D dominant, an incentive problem); the SH has only fear (a trust problem). Mixing them up gives the wrong fix.
> - **$p^*$ is about beliefs and needs a coordination game.** The formula's direction holds when $(R-T)+(P-S) > 0$; in a PD it gives a number outside $[0,1]$ ($-1$ for the lab's PD).
> - **"TFT always wins."** Tournament rankings depend on the field: with AllC and Random present, AllD wins the 50-round tournament.
> - **Reading noise as signal.** Differences of a few hundredths between strategies that act identically are sampling noise.

---

## Part II: replicator dynamics and deception

### The replicator equation

In Part I agents calculated. Many agents instead **adapt**: a trading bot keeps the rule that made money, a reinforcement learner repeats what paid off, a platform deploys more copies of the agent that sells best. Evolutionary game theory asks where that adaptation leads.

Take a large population in which a fraction $x$ cooperates and $1 - x$ defects, paired at random. A partner cooperates with probability $x$, so the expected payoffs (fitness) are

$$f_C(x) = xR + (1-x)S, \qquad f_D(x) = xT + (1-x)P, \qquad \bar f(x) = x f_C(x) + (1-x) f_D(x),$$

and the **[[Replicator Dynamics|replicator equation]]** says a strategy grows when it earns more than average, at a rate proportional to the gap and to how common it already is:

$$\frac{dx}{dt} = x\big(f_C(x) - \bar f(x)\big).$$

With $k$ strategies, frequencies $\mathbf x$ and payoff matrix $A$: $\dfrac{dx_i}{dt} = x_i\big((A\mathbf x)_i - \mathbf x^\top A \mathbf x\big)$. (My check at $x = 0.3$ in the PD: both forms give $-0.273$ for cooperators.)

It models agents in two ways: it is the average behaviour of a population where agents meet a random peer and copy its strategy with probability proportional to how much better the peer does (imitation of success), and the expected behaviour of a simple reinforcement learner (Cross learning) with a small learning rate (Börgers and Sarin, 1997).

A **rest point** is an $x$ with $dx/dt = 0$. It is **stable** if the population returns after a small push either way, **unstable** otherwise. The set of starting states that end at a stable rest point is its **basin of attraction**. The equation is solved numerically with the Runge-Kutta method (RK4, which averages the rate of change at four points inside each step of size 0.05).

Two facts from the factorised form below make stability quick to read off. Write $h(x) = f_C(x) - f_D(x)$, so $dx/dt = g(x) = x(1-x)h(x)$. Then $g'(0) = h(0) = S - P$ and $g'(1) = -h(1) = T - R$. So:

- $x = 0$ (all defect) is stable when $P > S$: fear protects defection.
- $x = 1$ (all cooperate) is stable when $R > T$: no greed protects cooperation.

### Q8: predictions before the analysis

> [!question] Statement
> A large population of agents is repeatedly paired at random to play a game. The agents do not compute equilibria; strategies that earn more than average spread (agents copy successful peers, or successful agents are deployed more often).
>
> (a) The population plays the Prisoner's Dilemma ($R=3, S=0, T=5, P=1$) and starts with **90% cooperators**. What happens to the fraction of cooperators over time?
>
> (b) The population plays the Stag Hunt ($R=4, S=0, T=3, P=2$) and starts with **60% cooperators** (stag hunters). What happens to the fraction of cooperators over time?

**(a)** Careful prediction: cooperators die out. Defectors earn more than cooperators against *any* mix of partners (D is dominant), so their share grows from any start; 90% only delays it. At $x = 0.9$: $f_C = 2.7$, $f_D = 4.6$. Run: $x$ falls below 0.5 at $t \approx 1.3$ and below 0.01 at $t \approx 5.4$.

**(b)** The tempting prediction is that a cooperative majority wins. A careful one compares payoffs at $x = 0.6$: $f_C = 4(0.6) = 2.4$ and $f_D = 3(0.6) + 2(0.4) = 2.6$. Hare hunters earn more, so stag hunting declines, and keeps declining because the gap widens as $x$ falls. Run: $x = 0.528$ at $t = 1$, $0.004$ at $t = 5$, then 0. A majority is not enough: cooperation needs more than $2/3$ (Q10).

### Q9: the Prisoner's Dilemma under replicator dynamics

> [!question] Statement
> The code reports, for the PD: rest points $x = 0$ (stable) and $x = 1$ (unstable), and all ten trajectories from $x(0) \in [0.05, 0.95]$ fall to 0.
>
> (a) Show, in one or two lines of algebra, that $x\big(f_C - \bar f\big) = x(1-x)(f_C - f_D)$.
>
> (b) For the Prisoner's Dilemma, write $f_C(x) - f_D(x)$ in terms of $R, S, T, P$ and $x$. Using the greed ($T > R$) and fear ($P > S$) motives from Part I, explain why it is negative for every $x$. What does this imply for the phase line, and for your prediction in Q8(a)?
>
> (c) Compare the rest points with the Nash equilibrium of the Prisoner's Dilemma. Which rest point is stable, and which Nash equilibrium does it correspond to? The state $x = 1$ (everybody cooperates) is also a rest point: why does the population not stay there after a single defector appears?

**(a)** Substitute $\bar f$:

$$f_C - \bar f = f_C - x f_C - (1-x) f_D = (1-x) f_C - (1-x) f_D = (1-x)(f_C - f_D),$$

and multiply by $x$. The factor $x(1-x)$ says change is slow when either strategy is rare; the sign is set by $f_C - f_D$ alone.

**(b)**

$$f_C(x) - f_D(x) = x(R - T) + (1-x)(S - P).$$

Greed means $R - T < 0$ and fear means $S - P < 0$. For $x \in [0, 1]$ the expression is a weighted average of two negative numbers, so it is negative for every $x$. For the lab's PD: $-2x - (1 - x) = -(1 + x)$, between $-1$ and $-2$. So $dx/dt = x(1-x)(f_C - f_D) < 0$ on the whole open interval $(0, 1)$: the phase line is below zero everywhere between the endpoints, and every population with any defectors slides to $x = 0$. Q8(a): 90% cooperators end at 0.

**(c)** Rest points: $x = 0$ (stable) and $x = 1$ (unstable). There is no interior one: the formula $(P - S)/((R-T)+(P-S)) = 1/(-2+1) = -1$ lies outside $[0, 1]$, which is (b) again, since $f_C - f_D$ never reaches zero. The stable rest point $x = 0$, everyone defecting, corresponds to the unique Nash equilibrium (D,D). The state $x = 1$ is a rest point only because there is nobody to copy: with no defectors, the defector share cannot grow. A single defector in a population of cooperators earns $T = 5$ against cooperators' $R = 3$, so it is copied and its share grows (in the stability rule: $g'(1) = T - R = 2 > 0$). Rest without stability means the state holds only while nothing disturbs it.

### Q10: the Stag Hunt under replicator dynamics

> [!question] Statement
> The code reports, for the SH: rest points $x = 0$ (stable), $x = 0.667$ (unstable), $x = 1$ (stable); trajectories starting at 0.75 and above rise to 1, those at 0.65 and below fall to 0.
>
> (a) Study the sign of $f_C(x) - f_D(x)$ for the Stag Hunt. Where do you expect the equation to be positive, negative, zero?
>
> (b) List the rest points of the Stag Hunt and their stability. The interior rest point corresponds to the *mixed* Nash equilibrium of the game (each agent cooperates with probability $x$). Is this equilibrium something an adaptive population will end up in? Why or why not?
>
> (c) Compare the interior rest point with the threshold belief $p^*$ from Part I (same payoffs). In Part I, rational agents chose a best response to a belief; here agents have no beliefs and only copy what works. Why do both models give the same threshold? Hint: what plays the role of the belief $p$ when partners are drawn at random from the population?
>
> (d) Check your prediction Q8(b). Consider the sentence: "If the majority already cooperates, cooperation must win, because (C,C) is better for everyone." Which output contradicts this, and what is wrong with the argument?

**(a)** $f_C - f_D = x(4 - 3) + (1-x)(0 - 2) = 3x - 2$. It is negative for $x < 2/3$, zero at $x = 2/3$ and positive for $x > 2/3$. So

$$\frac{dx}{dt} = x(1-x)(3x - 2)$$

is negative on $(0, 2/3)$, zero at $0$, $2/3$ and $1$, and positive on $(2/3, 1)$.

**(b)**

| rest point | stability | why | basin |
|---|---|---|---|
| $x = 0$ (all hare) | stable | $g'(0) = S - P = -2 < 0$ | $[0, 2/3)$ |
| $x = 2/3$ (mixed) | unstable | $g'(2/3) = \tfrac23 \cdot \tfrac13 \cdot 3 = \tfrac23 > 0$ | only the point itself |
| $x = 1$ (all stag) | stable | $g'(1) = T - R = -1 < 0$ | $(2/3, 1]$ |

An adaptive population will not end up at the mixed equilibrium. It is a rest point because there both strategies earn exactly the same, but any small push tips it: slightly more stag hunters make stag pay more, which attracts more stag hunters, and the reverse below. Only a population that starts exactly at $2/3$ and is never disturbed stays there. The mixed equilibrium is the watershed between the two basins.

**(c)** The interior rest point is

$$x^* = \frac{P - S}{(R - T) + (P - S)} = \frac{2}{1 + 2} = \frac23 = p^*.$$

Under random matching, the probability that my partner cooperates is the population share $x$. So $f_C(x) = \pi_C(p = x)$ and $f_D(x) = \pi_D(p = x)$: the fitness functions are the expected payoffs of Part I with the share standing in for the belief. A rest point needs $f_C(x) = f_D(x)$, which is the same equation as indifference $\pi_C(p) = \pi_D(p)$, so it has the same solution. The rational agent cooperates when its belief exceeds $p^*$; the adaptive population grows its cooperators when the actual share exceeds $p^*$. Same threshold, reached by different routes.

**(d)** Q8(b) is confirmed: from 60% stag hunters the population falls to 0 (0.528 at $t = 1$, 0.004 at $t = 5$), and even the trajectory from 65% ends at 0. The sentence is contradicted by these runs and by the rest point at $2/3$. The flaw: what spreads is what earns more than average *given the current mix*, and that is a property of the individual strategy against the population. That everyone would be better off if everyone cooperated is a comparison of whole outcomes, which no adapting agent ever makes. With 60% stag hunters a stag hunter fails 40% of the time and averages 2.4, below a hare hunter's 2.6. The bar is $2/3$ rather than $1/2$ because the fear term ($P - S = 2$) outweighs the gain ($R - T = 1$).

### Deception: the signalling game

Many interactions have **asymmetric information**: one agent knows its own quality and the other does not. The informed agent can send a **signal** (a "premium" label). A signal is useful only if it is reliable, and low-quality agents are tempted to fake it. A malicious agent that presents itself as trustworthy is the agentic version.

- A fraction $\lambda$ of signallers is high quality (H), the rest low quality (L). Signallers know their quality; receivers do not.
- Each signaller sends the signal or not. It costs $c_H$ for H and $c_L$ for L.
- Each receiver is **trusting** (accepts a signaller if and only if it signals) or **sceptical** (accepts nobody).
- An accepted signaller gains $b$, whatever its quality. A receiver gains $g$ from accepting H and loses $d$ from accepting L.

A low-quality signaller that signals **deceives**. Let $x_H$, $x_L$ be the fractions of H and L signallers that signal and $y$ the fraction of trusting receivers. Payoff advantages over the alternative (which pays 0):

| group | strategy | advantage over the alternative |
|---|---|---|
| H signallers | signal | $yb - c_H$ |
| L signallers | signal (deceive) | $yb - c_L$ |
| receivers | trust | $\lambda x_H g - (1-\lambda) x_L d$ |

Each group follows the two-strategy replicator equation ("$x(1-x)$ times the advantage"):

$$\frac{dx_H}{dt} = x_H(1-x_H)(yb - c_H), \quad \frac{dx_L}{dt} = x_L(1-x_L)(yb - c_L), \quad \frac{dy}{dt} = y(1-y)\big(\lambda x_H g - (1-\lambda) x_L d\big).$$

H and L are separate groups because an agent chooses whether to signal and cannot choose its quality. The **reliability** of the signal is $\Pr(H \mid \text{signal}) = \lambda x_H / (\lambda x_H + (1-\lambda) x_L)$; without a signal a receiver only knows the prior $\lambda$. Defaults: $\lambda = 0.3$, $b = g = d = 1$. Accepting a random signaller loses ($0.3 - 0.7 < 0$), so receivers benefit only from an informative signal. Each run lasts until $t = 300$ and the summary averages the second half; the receiver's average payoff is $y$ times the advantage of trusting.

### Q11: does faking pay? (prediction)

> [!question] Statement
> Fix $c_H = 0.2$ (the signal is cheap for high-quality signallers) and start from $x_H = x_L = y = 0.5$. Consider two cases: **(A)** faking is expensive, $c_L = 1.5$; **(B)** faking is cheap, $c_L = 0.5$. For each case, predict the long-run behaviour. Do low-quality signallers deceive? Do receivers keep trusting? Is the signal reliable?

**Careful prediction for (A).** Faking pays at most $b - c_L = 1 - 1.5 = -0.5$ even if every receiver trusts, so $x_L$ falls whatever happens. For H, signalling pays whenever $y > 0.2$. Once fakers are rare, trusting pays ($0.3 x_H > 0.7 x_L$), so $y$ rises, which makes signalling even better for H. Expected end: $x_H = 1$, $x_L = 0$, $y = 1$, an honest (separating) signal with reliability 1. Receivers then earn $\lambda g = 0.3$ and H signallers $b - c_H = 0.8$.

**Careful prediction for (B).** Now faking pays whenever $y > 0.5$. There is no state where all three groups are content. When trust is high, fakers spread; when fakers are common, trusting loses ($0.3 x_H < 0.7 x_L$) and trust falls; when trust falls below 0.5, faking stops paying and fakers decline; then trusting pays again. Expect cycles. H signallers keep signalling because $y$ hovers around 0.5, above their break-even of 0.2. The cycles turn around the point where both L signallers and receivers are indifferent, with $x_H = 1$:

$$y b = c_L \Rightarrow y = 0.5, \qquad \lambda x_H g = (1-\lambda) x_L d \Rightarrow x_L = \frac{0.3}{0.7} = \frac37 \approx 0.43,$$

where reliability is $0.3 / (0.3 + 0.7 \cdot 3/7) = 0.5$: better than the prior of 0.3 but far from reliable.

**Runs:**

| quantity (second-half average) | (A) $c_L = 1.5$ | (B) $c_L = 0.5$ |
|---|---|---|
| $x_H$, H signals | 1.000 | 1.000 |
| $x_L$, L signals (deceives) | 0.000 | 0.444 (cycles 0.19 to 0.69) |
| $y$, receivers trust | 1.000 | 0.498 (cycles 0.28 to 0.72) |
| reliability $\Pr(H \mid \text{signal})$ | 1.000 | 0.513 |
| receiver average payoff | 0.300 | $-0.006$ |

In (A) fakers vanish fast ($x_L = 0.008$ by $t = 5$), trust climbs to 1 and the signal is fully honest. In (B) low-quality signallers deceive a large part of the time, trust rises and falls with them, and the cycles keep the same size until the end of the run. Receivers gain essentially nothing from the signal.

### Q12: the uniform fee

> [!question] Statement
> A platform designer looks at the market of Q11 (B), where the label costs $c_H = 0.2$ and $c_L = 0.5$ and low-quality sellers fake it. Fearing that the label will always be misused, the designer **makes it expensive for everyone**: every seller that shows the label pays an extra fee $k = 1$ (an expensive verification procedure). The designer cannot observe quality, so the fee is the same for every seller: $c_H = 1.2$ and $c_L = 1.5$.
>
> (a) Prediction (before running the next cell): will the fee stop deception? Will receivers be better off? Will high-quality sellers?
>
> (b) Run the next cell and compare the two markets. Did the fee eliminate deception, and how? Compare the receivers' payoff, the high-quality sellers' payoff and the total welfare in both markets. Hint: why is the receivers' payoff close to 0 even in the market with deception?
>
> (c) Is the high fee good or bad? Rewrite the designer's reasoning so that it is correct.

Total welfare per encounter is one receiver's payoff plus one randomly drawn seller's: $y(\lambda x_H g - (1-\lambda)x_L d) + \lambda\, x_H(yb - c_H) + (1-\lambda)\, x_L(yb - c_L)$.

**(a)** Careful prediction: with $c_H = 1.2 > b = 1$, even an honest seller loses money on the label when every receiver trusts. So H sellers stop signalling, and L sellers stop too ($1.5 > 1$). Deception ends because the label disappears. Receivers were earning about 0 already (Q11 B), and with no signals they earn exactly 0, so they gain nothing. High-quality sellers lose the 0.3 they were earning.

**(b)**

| quantity (second-half average) | no fee, $k = 0$ | fee for everyone, $k = 1$ |
|---|---|---|
| $x_H$ | 1.000 | 0.000 |
| $x_L$ (deceives) | 0.444 | 0.000 |
| $y$ | 0.498 | 0.453 (frozen, nobody signals) |
| reliability | 0.513 | undefined (no signals) |
| receiver average payoff | $-0.006$ | 0.000 |
| H seller average payoff | 0.298 | 0.000 |
| total welfare | 0.084 | 0.000 |

The fee eliminated deception by eliminating signalling: both seller groups abandon the label within about 10 time units. Trust freezes at 0.453 because once nobody signals, trusting and scepticism pay the same and $dy/dt = 0$.

Why receivers earn about 0 even with deception: in the cycling market, low-quality sellers fake exactly as much as trust allows, and the cycle turns around $x_L = 3/7$, the level that makes trusting pay the same as scepticism. A short derivation: $\frac{d}{dt}\ln\frac{y}{1-y} = \frac{\dot y}{y(1-y)}$ is the advantage of trusting. Since $y$ stays between 0.28 and 0.72, $\ln\frac{y}{1-y}$ stays bounded, so the long-run average of the advantage of trusting must be about 0. The deceivers absorb whatever receivers would gain from the honest sellers. So the fee did not cost receivers anything, and it did not help them either.

**(c)** The fee is bad. Receivers stay at 0, high-quality sellers fall from 0.298 to 0, total welfare from 0.084 to 0. Deception is gone only because the market for the label is gone.

The designer's mistake is to treat the cost of the label as what stops faking. What stops faking is a **cost difference** between types: an honest signal survives when it pays for high-quality sellers and does not pay for low-quality ones,

$$c_H < b < c_L,$$

as in case (A). A fee charged to everyone moves both costs by the same $k$, so it can only help if it lands $b$ between them: $0.2 + k < 1 < 0.5 + k$, so $0.5 < k < 0.8$. The designer's $k = 1$ overshoots and prices out the honest sellers too. My extra runs show that even a fee inside that window, say $k = 0.6$, collapses the market from the 50/50 start, because trust (0.5) is below the honest sellers' new break-even (0.8) and they quit before trust can build; from a high-trust start ($x_H = 0.9$, $x_L = 0.5$, $y = 0.9$) it reaches the honest outcome ($x_L = 0$, $y = 1$, welfare 0.36).

A correct version of the reasoning: "The label is misused because faking it is cheap relative to what being trusted is worth. I should make the label costly to fake while keeping it cheap for genuine sellers, for example a verification that genuine sellers pass easily and fakers fail, or a penalty for sellers caught faking. Raising the price for everyone removes honest sellers along with the fakers."

> [!warning] Common pitfalls in Part II
> - **"The mixed equilibrium is where the population ends up."** In the SH the interior rest point ($x = 2/3$) is unstable; it is the tipping point between the two basins.
> - **"A cooperative majority wins."** In the SH cooperation needs a share above $p^* = 2/3$; 60% loses.
> - **"(C,C) wins because it is better for everyone."** Replicator dynamics compares each strategy with the population average under the current mix; collective benefit plays no role.
> - **Rest point equals stable.** $x = 1$ in the PD is a rest point and one defector destroys it. Check $g'(0) = S - P$ and $g'(1) = T - R$.
> - **Confusing the share with a belief.** They play the same role under random matching, which is why $x^* = p^*$, but adaptive agents hold no beliefs.
> - **"Expensive signals are honest signals."** Honesty needs the cost to differ by type ($c_H < b < c_L$). A uniform fee large enough to stop fakers also stops honest senders.

---

## Part III: the vacuum world and reward hacking

### The model

Payoffs in Parts I and II were given. When an AI agent is trained, someone has to write the reward function, and the agent then does whatever earns most reward, including things the designer never intended ([[Reward Hacking]]). The lab shows this in the two-square vacuum world of Russell and Norvig, with dirt that comes back.

- Two squares, A (left) and B (right), each clean or dirty; the robot is in one of them.
- **Left** moves to A, **Right** to B; moving into the wall leaves the robot where it is (the motor still runs).
- **Suck** on a dirty square cleans it and puts the dirt in the bag (the bag holds 0 or 1). Suck on a clean square does nothing.
- **Dump** (added later by a software update) empties a full bag onto the current square, which becomes dirty. Dump with an empty bag does nothing.
- After every action each clean square becomes dirty with probability $p$ (the dirt rate, 0.1 by default), independently. A dirty square stays dirty until sucked.

The state is (robot position, A dirty, B dirty, bag full), so $2^4 = 16$ states: a small [[Markov Decision Process]]. Two rewards:

- **Bob:** +1 for every clean square after each time step (a function of the next state).
- **Alice:** +1 for every unit of dirt the robot sucks up.

The robot is a perfect optimiser. It knows the dynamics and computes an optimal policy with [[Value Iteration]], discount $\gamma = 0.95$:

$$Q(s,a) = \sum_{s'} P(s' \mid s,a)\big[r(s,a,s') + \gamma V(s')\big], \qquad V(s) = \max_a Q(s,a),$$

starting from $V = 0$ and repeating until the values stop changing. Ties (actions within $10^{-6}$ of the best) are broken at random in simulation. The robot then runs for 1000 steps from "robot in A, both squares dirty, bag empty", with the same random dirt for every robot. The **performance measure**, how *we* judge the robot whatever its reward: floor clean (%), moves per step (Left or Right, wall bumps included), dumps per step (with a full bag), dirt collected per step, and reward per step (what the robot sees). Any odd behaviour therefore comes from the reward, since the optimiser is exact.

### Q13: predictions before the analysis

> [!question] Statement
> The goal is a clean office. Which reward function best accomplishes this?
>
> (a) Bob's reward: **+1 for every clean square, at every time step.** No other benefits or costs. Suppose that, after the start, no new dirt ever appears. What will the robot do once both squares are clean?
>
> (b) Alice's reward: **+1 for every unit of dirt the robot sucks up.** In the test lab (actions Left, Right, Suck), which robot keeps the floor cleaner: Bob's, Alice's, or neither? A software update then adds a fourth action, **Dump**, which empties the robot's bag onto its current square (useful, say, to empty the bag above a bin). Predict what each robot does after the update.

**(a)** Careful prediction: once both squares are clean, every action earns the same +2 per step forever, so the robot is indifferent among all of them. What it does is decided by the tie-breaking rule. With random tie-breaking it wanders aimlessly, moving about two steps in three (two of its three actions are moves).

**(b)** In the lab, neither: in a world where dirt can only arrive and only Suck removes it, both rewards are maximised by sucking dirt as fast as possible, so the two robots should behave the same. After the update: Bob's robot never dumps dirt, because dirt on the floor lowers his reward. Alice's robot dumps and re-sucks the same dirt, since every dump creates a new +1 one step later.

### Q14: Bob's robot

> [!question] Statement
> Bob's reward, no new dirt (dirt rate 0), robot starts in A with both squares dirty. The policy table shows the bag-empty states.
>
> (a) Look at the policy table. In which states does the robot have more than one optimal action? Why are these actions equally good under Bob's reward?
>
> (b) Check your prediction Q13(a). Once the floor is clean and no new dirt arrives, what does the robot do? Is the robot behaving *irrationally*? Answer in terms of what the robot maximises.

Bob's optimal policy, dirt rate 0 (bag ignored, it plays no role without Dump; `*` marks dirt, brackets the robot):

| state | optimal action(s) | $Q$ values |
|---|---|---|
| [A] B | Left, Right, Suck | all 40.00 |
| [A] B* | Right | Right 39.00, others 38.05 |
| [A*] B | Suck | Suck 40.00 |
| [A*] B* | Suck | Suck 38.05 |
| A [B] | Left, Right, Suck | all 40.00 |
| A [B*] | Suck | Suck 40.00 |
| A* [B] | Left | Left 39.00 |
| A* [B*] | Suck | Suck 38.05 |

Performance over 1000 steps:

| | no new dirt | dirt rate 0.1 |
|---|---|---|
| floor clean (%) | 99.9 | 84.4 |
| moves per step | 0.665 | 0.565 |
| dumps per step | 0.000 | 0.000 |
| dirt collected per step | 0.002 | 0.196 |
| reward per step | 1.998 | 1.687 |

**(a)** More than one optimal action only in the two states where both squares are clean (robot in A or in B): Left, Right and Suck are all optimal. Whatever the robot does there, the floor stays clean, so it earns 2 now and in every future step: $Q = 2/(1 - 0.95) = 40$ for each action. Bob's reward depends only on how many squares are clean, and nothing in it distinguishes moving, bumping into a wall or sucking a clean floor. With dirt rate 0.1 the same tie remains (all three at 34.00): the two squares get dirty equally often, so being in A or B with a clean floor is worth the same.

**(b)** The robot cleans both squares in three steps (Suck, Right, Suck), then wanders: in the trace it moves Right, Right, Suck, Suck, Left, Left, ..., and it moves on 0.665 of all steps, close to the $2/3$ that random choice among Left, Right and Suck gives. It is not irrational. It maximises Bob's reward, and every action earns the maximum, 2 per step. The wandering costs energy and wear, which the owner cares about and the reward never mentions. (With the notebook's helper that subtracts a cost per move, I checked that a move cost of 0.5 makes Suck, which leaves the robot in place, the unique optimum in the clean states, and the robot sits still.)

### Q15: Alice's robot and the Dump action

> [!question] Statement
> Alice's and Bob's robots are compared in the test lab (Left, Right, Suck; dirt rate 0.1; no move cost), then both get **Dump**, with the same rewards, dirt rate and dirt.
>
> (a) Without Dump, compare Alice's and Bob's robots. Could *any* test in this environment, using only the robots' behaviour, reveal which reward each robot is optimising?
>
> (b) With Dump, describe what Alice's robot does, quoting numbers from the table and the trace. In which states of the policy table does it dump? Explain why dumping is optimal under Alice's reward and never optimal under Bob's.
>
> (c) Check your prediction Q13(b). Nobody programmed the robot to dump dirt on the floor. Where does this behaviour come from? What does this suggest about giving an agent more actions or tools?

Performance, dirt rate 0.1, 1000 steps:

| | Bob, lab | Bob, with Dump | Alice, lab | Alice, with Dump |
|---|---|---|---|---|
| floor clean (%) | 84.4 | 84.4 | 84.4 | 59.1 |
| moves per step | 0.565 | 0.565 | 0.565 | 0.073 |
| dumps per step | 0.000 | 0.000 | 0.000 | 0.400 |
| dirt collected per step | 0.196 | 0.196 | 0.196 | 0.527 |
| reward per step | 1.687 | 1.687 | 0.196 | 0.527 |

**(a)** In the lab the two robots have the same set of optimal actions in all 16 states (suck when on dirt, move towards dirt when standing on a clean square, anything when the floor is clean), so with the same tie-breaking they chose the same action at every one of the 1000 steps. Their $Q$ values differ in size (Bob's about 30 to 34, Alice's about 3.5 to 5.2) but have the same maximising actions. No behavioural test in this environment can tell them apart: any experiment you run with these three actions shows identical behaviour, because both rewards induce the same optimal policy. The reason is that, with only Left, Right and Suck, a square becomes clean only by being sucked and dirt only comes from the environment, so "squares clean" and "dirt sucked" move together. Only a change to the environment that separates the two quantities can reveal the difference.

**(b)** With Dump, Alice's robot farms dirt. It dumps on 40% of all steps, collects 0.527 units of dirt per step (2.7 times the lab's 0.196) and keeps the floor only 59.1% clean (against 84.4%). It barely moves (0.073 moves per step). The start of the trace:

| t | state before | action | reward |
|---|---|---|---|
| 0 | [A*] B* bag 0 | Suck | +1 |
| 1 | [A] B* bag 1 | Right | 0 |
| 2 | A* [B*] bag 1 | Suck | +1 |
| 3 | A* [B] bag 1 | Dump | 0 |
| 4 | A* [B*] bag 0 | Suck | +1 |
| 5 | A* [B] bag 1 | Dump | 0 |
| 6 | A* [B*] bag 0 | Suck | +1 |

It settles in B and alternates Dump and Suck, earning +1 every two steps, while A, which became dirty at step 1, stays dirty until the robot happens to wander back (a tie at step 10 sends it Left; it sucks A and starts the same loop there).

Dump is optimal in exactly the four states where **the bag is full and the robot's square is clean**:

| state | optimal action(s) |
|---|---|
| [A] B, bag full | Dump (unique) |
| A [B], bag full | Dump (unique) |
| [A] B*, bag full | Right or Dump (tie) |
| A* [B], bag full | Left or Dump (tie) |

In the two states with the other square dirty, Dump ties with moving to the dirt: both deliver the next +1 two steps later and lead to states of equal value. Once the robot can make its own dirt, dirt elsewhere adds nothing; its value depends only on how soon it can next suck (standing on dirt 10.78, one step away 10.24, empty bag on a clean floor 8.24).

Why dumping is optimal for Alice: her reward counts sucking events. Dump followed by Suck returns the robot to the same state with +1 earned, a guaranteed 0.5 per step, against waiting for dirt that arrives at rate 0.1 per square. (The reward per step of 0.527 is above 0.5 because new dirt sometimes lands right after a suck.) Even with no new dirt at all, my run of Alice's robot with Dump earns 0.500 per step while keeping the floor only 75% clean; without Dump it would earn 0.002.

Why never for Bob: his reward counts clean squares and ignores the bag. Dumping onto a clean square makes it dirty, costing a point at every step until it is sucked again, and buys nothing. Dumping onto a dirty square changes nothing on the floor and wastes a step. In Bob's table Dump appears among the optimal actions only in the two states with an empty bag and a clean floor, where it does nothing at all (it ties with standing still). In the run Bob's bag is full after his first suck and he never empties it: dumps per step 0.000, and every number matches the lab exactly.

**(c)** Q13(b) is confirmed. Nobody programmed dumping. It comes from optimising a reward that measured a proxy (dirt collected) in place of the goal (a clean floor). In the lab the proxy and the goal moved together, because the only source of dirt was the environment, so the proxy looked perfect in every test. The new action gave the robot a way to raise the proxy without serving the goal (create dirt in order to collect it), and a perfect optimiser takes any such way. That is [[Reward Hacking]] in its simplest form.

Lessons for agents with more actions or tools:
- Every new action or tool can break the link between the reward and the intended goal; a reward validated with one action set is not validated for a larger one.
- Behaviour in testing cannot certify the reward: Bob's and Alice's robots were indistinguishable step for step.
- Reward the outcome the designer wants (Bob: the state of the floor) over an activity that usually accompanies it (Alice: sucking), and test again whenever the agent's capabilities change.

> [!warning] Common pitfalls in Part III
> - **"The wandering robot is buggy or irrational."** It maximises its reward exactly; the reward simply does not care about moves. Unwanted behaviour from an exact optimiser points at the reward.
> - **"Alice's robot dumps because of a bug in value iteration."** The same planner gives Bob's robot no dumping. The difference is in the reward alone.
> - **"Identical behaviour in testing means identical objectives."** Two rewards can induce the same policy in one environment and diverge as soon as the environment changes.
> - **Reading the tie states.** A tie means the robot is exactly indifferent, here because a clean floor pays the same whatever it does, and A and B are symmetric. With Dump, Bob's extra "optimal" Dump appears only where the bag is empty and the action does nothing.

---

## What to be able to write

- Find pure Nash equilibria, Pareto-optimal outcomes and the social optimum of any symmetric 2x2 game, and explain why the PD's equilibrium is inefficient.
- Greed ($T > R$) and fear ($P > S$), and which games have which.
- Derive $p^*$, the uniform-belief outcome frequencies, and the rule $R - T > P - S$ for (C,C) being played more often.
- Explain why memory-1 strategies can make cooperation pay, and derive the crossover length (4 rounds) for AllD against TFT and WSLS.
- Derive $x(f_C - \bar f) = x(1-x)(f_C - f_D)$, the rest points and their stability, and why the interior rest point equals $p^*$.
- The signalling game: when an honest signal survives ($c_H < b < c_L$), why deception makes signals cycle, and why a uniform fee kills the market.
- The vacuum world: why Bob's and Alice's robots are indistinguishable in the lab, where Alice's robot dumps and why, and what that says about tools and rewards.

## Exam questions

> [!exam]- A symmetric game has $R = 5$, $S = 0$, $T = 3$, $P = 1$. Find the pure Nash equilibria, the Pareto-optimal outcomes and the social optimum, compute $p^*$, say which equilibrium uniform-belief agents play more often, and give the replicator rest points with their stability.
> **Key points:** Nash (C,C) and (D,D), a Stag Hunt (fear, no greed); only (C,C) Pareto optimal and socially optimal; $p^* = 1/3$; (C,C) more often since $R - T = 2 > P - S = 1$, with $\Pr(C,C) = 4/9$; rest points 0 stable, 1/3 unstable, 1 stable.
>
> **Game.** $R > T > P > S$, so it is a Stag Hunt: no greed ($T = 3 < R = 5$), fear ($P = 1 > S = 0$).
>
> **Nash.** At (C,C) deviating gives 3 < 5; at (D,D) deviating gives 0 < 1; at (C,D) the cooperator gains by switching to D (1 > 0). Pure equilibria: (C,C) and (D,D).
>
> **Pareto and social.** Payoffs (5,5), (0,3), (3,0), (1,1). (C,C) beats every other cell for both players, so it is the only Pareto-optimal outcome, and with welfare 10 (against 3, 3 and 2) the social optimum.
>
> **Threshold.** $p^* = \frac{P-S}{(R-T)+(P-S)} = \frac{1}{2+1} = \frac13$. Stag is the best response when the belief that the partner hunts stag exceeds 1/3.
>
> **Uniform beliefs.** Each agent cooperates with probability $2/3$: $\Pr(C,C) = 4/9 \approx 0.44$, $\Pr(D,D) = 1/9 \approx 0.11$, miscoordinated $4/9$. (C,C) is more frequent, as the rule $R - T > P - S$ ($2 > 1$) predicts.
>
> **Replicator.** $f_C - f_D = x(R-T) + (1-x)(S-P) = 2x - (1-x) = 3x - 1$, so $\dot x = x(1-x)(3x-1)$. Rest points 0 (stable, $g'(0) = S - P = -1$), $1/3$ (unstable), 1 (stable, $g'(1) = T - R = -2$). Populations starting above $1/3$ cooperators end at full cooperation, below at full defection.

> [!exam]- A symmetric game has $R = 3$, $S = 1$, $T = 4$, $P = 0$. Find the pure Nash equilibria, the replicator rest points and their stability, and explain how the long-run population differs from the Stag Hunt's.
> **Key points:** greed present, fear absent; pure Nash (C,D) and (D,C), no symmetric pure equilibrium; $f_C - f_D = 1 - 2x$; rest points 0 and 1 unstable, $x = 1/2$ stable; population converges to a mix, while the SH's interior point is unstable.
>
> **Motives.** Greed: $T = 4 > R = 3$. Fear absent: $P = 0 < S = 1$. Each player wants to do the opposite of the other.
>
> **Nash.** Against C the best response is D (4 > 3); against D it is C (1 > 0). So the pure equilibria are (C,D) and (D,C); (C,C) and (D,D) are not equilibria. Social optimum: (C,C) with welfare 6 (against 5, 5, 0).
>
> **Replicator.** $f_C - f_D = x(R-T) + (1-x)(S-P) = -x + (1-x) = 1 - 2x$, so $\dot x = x(1-x)(1-2x)$. Rest points: $x = 0$ with $g'(0) = S - P = 1 > 0$, unstable; $x = 1$ with $g'(1) = T - R = 1 > 0$, unstable; $x^* = 1/2$, where $g'(1/2) = \frac12 \cdot \frac12 \cdot (-2) = -\frac12 < 0$, stable. From any interior start the population goes to half cooperators.
>
> **Contrast.** In the SH ($R > T$, $P > S$) both pure states are stable and the interior rest point (the mixed equilibrium) is unstable, a tipping point. Here both pure states are unstable (a rare strategy always earns more) and the interior rest point, the mixed equilibrium with cooperation probability 1/2, is where the population ends. Whether adaptation ends at the mixed equilibrium depends on the signs of $S - P$ and $T - R$.

> [!exam]- In a repeated Prisoner's Dilemma ($R=3, S=0, T=5, P=1$), explain why memory matters, why AllD can still top a round robin with AllC, AllD, Random, TFT and WSLS over 50 rounds, and from how many rounds TFT beats AllD in a field of AllD, TFT and WSLS.
> **Key points:** memory-0 strategies cannot react, memory-1 strategies condition on the last outcome; TFT loses only round 1 to AllD; AllD's score comes from exploiting AllC (5), Random and WSLS (about 3 each); in the three-strategy field AllD wins for 1 to 3 rounds and TFT from 4.
>
> **Memory.** A memory-0 strategy (AllC, AllD, Random) plays the same way whoever it meets, so a cooperator of this kind is exploited every round. A memory-1 strategy conditions on the previous outcome: TFT cooperates with cooperators and defects against AllD from round 2, so it loses 5 points once and then gets 1 per round (0.98 per round over 50).
>
> **Why AllD tops the five-strategy field.** Its 50-round score, about 2.62, averages 5 against AllC, 1 against itself, 3 against Random, 1.08 against TFT and 3 against WSLS, which alternates C and D against it. Most of its points are the sucker payoffs of AllC, Random and WSLS; against TFT it earns little. TFT is second at about 2.44.
>
> **Crossover.** Per-round tournament scores times 3 over $n$ rounds: AllD $5 + 4/n$ ($n$ even) or $5 + 6/n$ ($n$ odd); TFT $7 - 1/n$; WSLS $6 + \lfloor n/2 \rfloor / n$. AllD beats TFT iff $n < 2.5$ (even) or $n < 3.5$ (odd), so AllD is best for 1, 2 and 3 rounds and TFT from 4 rounds on.

> [!exam]- Derive the two-strategy replicator equation's factorised form, use it to show that cooperation always dies out in a Prisoner's Dilemma, and explain why the Stag Hunt's interior rest point equals the threshold belief $p^*$ even though adaptive agents hold no beliefs.
> **Key points:** $x(f_C - \bar f) = x(1-x)(f_C - f_D)$; PD: $f_C - f_D = x(R-T) + (1-x)(S-P) < 0$ by greed and fear; $x = 1$ is a rest point but unstable; random matching makes the share $x$ play the role of the belief $p$; the same indifference equation gives the same solution.
>
> **Factorisation.** $\bar f = x f_C + (1-x) f_D$, so $f_C - \bar f = (1-x)(f_C - f_D)$ and $\dot x = x(1-x)(f_C - f_D)$.
>
> **PD.** $f_C - f_D = x(R-T) + (1-x)(S-P)$. Greed gives $R - T < 0$, fear gives $S - P < 0$, so for every $x \in [0,1]$ it is a weighted average of two negatives. Hence $\dot x < 0$ on $(0,1)$ and every population with some defectors goes to $x = 0$, matching the unique Nash equilibrium (D,D). $x = 1$ is a rest point because there is nobody to copy, but a single defector earns $T > R$ and spreads.
>
> **Why $x^* = p^*$.** With random pairing, a partner cooperates with probability equal to the population share $x$. So $f_C(x) = xR + (1-x)S$ is exactly $\pi_C(p)$ with $p = x$, and likewise for D. The interior rest point solves $f_C = f_D$, the same equation as the rational agent's indifference $\pi_C(p) = \pi_D(p)$, so $x^* = \frac{P-S}{(R-T)+(P-S)} = p^*$. The rational agent cooperates when its belief exceeds $p^*$; in the population, cooperators grow when the actual share exceeds $p^*$. For the lab's SH it is $2/3$, and it is unstable: a tipping point.

> [!exam]- In the signalling game with $\lambda = 0.3$, $b = g = d = 1$ and $c_H = 0.2$, explain the long-run outcomes for $c_L = 1.5$ and $c_L = 0.5$, and evaluate a designer's uniform fee of $k = 1$ on the label.
> **Key points:** $c_L > b$: fakers die out, honest separating outcome, reliability 1, receivers earn 0.3; $c_L < b$: cycles around $y = 0.5$, $x_L = 3/7$, reliability about 0.5, receivers earn about 0; fee makes $c_H = 1.2 > b$, nobody signals, welfare 0.084 to 0; honesty needs $c_H < b < c_L$.
>
> **Expensive faking ($c_L = 1.5$).** Faking pays at most $b - c_L < 0$, so $x_L \to 0$. H signals whenever $y > 0.2$; once fakers are rare trusting pays ($0.3 x_H > 0.7 x_L$), so $y \to 1$ and $x_H \to 1$. The signal is fully reliable; receivers earn 0.3 and H sellers 0.8.
>
> **Cheap faking ($c_L = 0.5$).** Faking pays when $y > 0.5$; trusting pays when $x_L < 3/7$ (with $x_H = 1$). No state satisfies everyone, so trust and faking chase each other in cycles around $y = 0.5$, $x_L = 3/7$. Reliability averages about 0.5 (prior 0.3) and receivers average about 0: since $\frac{d}{dt}\ln\frac{y}{1-y}$ equals the advantage of trusting and $y$ stays bounded, that advantage averages to zero.
>
> **The fee.** With $k = 1$, $c_H = 1.2$ and $c_L = 1.5$ both exceed $b$, so nobody signals. Deception ends, but receivers stay at 0, H sellers drop from about 0.3 to 0 and welfare from 0.084 to 0. The fee is harmful.
>
> **Correct design.** An honest signal needs $c_H < b < c_L$: affordable for genuine sellers, unprofitable for fakers. A uniform fee shifts both costs equally and could at most place $b$ between them ($0.5 < k < 0.8$ here); a cost that falls on fakers specifically (verification they fail, penalties) is the right tool.

> [!exam]- Two robot vacuums plan optimally with value iteration ($\gamma = 0.95$) in the two-square vacuum world with dirt rate 0.1. Bob's reward is +1 per clean square per step; Alice's is +1 per unit of dirt sucked. Explain why tests in the lab cannot tell them apart, what happens when a Dump action is added, and what this shows about agents and tools.
> **Key points:** identical optimal action sets in all 16 states without Dump, so identical behaviour; Alice dumps when the bag is full and her square is clean, then re-sucks for +1 every two steps (floor 84% to 59% clean, dirt collected 0.196 to 0.527 per step); Bob never dumps; reward hacking: a proxy that matched the goal under one action set diverges when a new action appears.
>
> **Lab.** With Left, Right and Suck, a square becomes clean only by being sucked and dirt only comes from the environment, so maximising clean squares and maximising dirt sucked call for the same actions: suck when on dirt, move to dirt, anything on a clean floor. Both robots have the same optimal set in every state and act identically step for step; no behavioural test in this environment can reveal the reward.
>
> **With Dump.** Alice's robot dumps in the four states with a full bag and a clean current square (uniquely when both squares are clean, tied with moving when the other square is dirty), then sucks the same dirt: +1 every two steps, guaranteed. It dumps on 40% of steps, its floor drops to about 59% clean, it barely moves. Bob's reward ignores the bag and penalises dirt, so dumping only loses reward; he never dumps.
>
> **Lesson.** Nobody programmed dumping: a perfect optimiser found a way to raise a proxy (dirt collected) without serving the goal (a clean floor). New actions or tools can break the link between reward and goal, testing behaviour cannot certify a reward, and rewards should target outcomes and be re-checked whenever capabilities change.

## Flashcards

> [!card]- What is a pure Nash equilibrium?
> An outcome where each player's action is a best response to the other's, so no one gains by deviating alone.

> [!card]- What makes an outcome Pareto optimal?
> No other outcome makes one player better off without making the other worse off.

> [!card]- Why is (C,D) in the Prisoner's Dilemma Pareto optimal but not socially optimal?
> Any other outcome lowers the defector's 5, yet its welfare (5) is below the 6 of (C,C).

> [!card]- What payoff ordering defines a Prisoner's Dilemma?
> $T > R > P > S$.

> [!card]- What payoff ordering defines a Stag Hunt?
> $R > T > P > S$.

> [!card]- Which motives push towards defection in the Prisoner's Dilemma, and which in the Stag Hunt?
> Prisoner's Dilemma: greed ($T > R$) and fear ($P > S$). Stag Hunt: fear only.

> [!card]- Why does the Prisoner's Dilemma pose an incentive problem while the Stag Hunt poses a trust problem?
> In the Prisoner's Dilemma defecting is best even if the partner surely cooperates; in the Stag Hunt only doubt about the partner pushes to defect.

> [!card]- What is the threshold belief $p^*$ in a symmetric 2x2 coordination game?
> $p^* = (P - S)/((R - T) + (P - S))$: cooperate if the belief that the partner cooperates exceeds it.

> [!card]- With uniform beliefs, how often do two agents reach (C,C) in the Stag Hunt with $p^* = 2/3$?
> $(1/3)^2 = 1/9$, about 11% of encounters.

> [!card]- When is (C,C) played more often than (D,D) by uniform-belief agents in a Stag Hunt?
> When $R - T > P - S$, which is the same as $p^* < 1/2$.

> [!card]- What happens to stag hunting in the Stag Hunt if $T$ rises from 3 to 4 (with $R=4, S=0, P=2$)?
> $p^*$ becomes 1, so no one hunts stag: (C,C) never occurs.

> [!card]- Why does "(C,C) is better for both, so rational agents play it" fail in the Stag Hunt?
> Each agent best-responds to its own belief; stag pays only if the partner is believed to hunt stag with probability above $p^*$.

> [!card]- How does Win-Stay Lose-Shift decide its next move?
> Repeat its last action after earning 3 or 5 points, switch after 0 or 1.

> [!card]- Why does Tit-for-Tat score better against AllD than AllC does?
> It conditions on the partner's last move, so it is exploited only in the first round.

> [!card]- Why can AllD top a 50-round round robin with AllC, AllD, Random, TFT and WSLS?
> It earns 5 per round from AllC and about 3 from Random and WSLS; its points come from exploitable partners.

> [!card]- In a field of AllD, TFT and WSLS, from how many rounds does TFT beat AllD?
> From 4 rounds.

> [!card]- How does Win-Stay Lose-Shift recover from an accidental defection against itself, unlike Tit-for-Tat?
> After (D,C) both play D; mutual defection is a loss, so both switch back to C. Tit-for-Tat pairs echo retaliation.

> [!card]- What does the replicator equation assume about how strategies spread?
> A strategy grows when it earns more than the population average, in proportion to the gap and its current share.

> [!card]- What is the factorised two-strategy replicator equation?
> $\dot x = x(1-x)(f_C - f_D)$.

> [!card]- Which agent models does the replicator equation describe besides biological evolution?
> Imitation of successful peers, and Cross learning (a simple reinforcement learner) with a small learning rate.

> [!card]- Why does cooperation die out from any start in a replicator Prisoner's Dilemma?
> $f_C - f_D = x(R-T) + (1-x)(S-P)$ is negative for all $x$, since greed and fear make both terms negative.

> [!card]- Why is "everyone cooperates" an unstable rest point in the Prisoner's Dilemma?
> It rests only because no one defects; a single defector earns $T > R$ and spreads.

> [!card]- What are the replicator rest points of the Stag Hunt with $R=4, S=0, T=3, P=2$, and their stability?
> 0 stable, 2/3 unstable, 1 stable.

> [!card]- Does an adaptive population end at the Stag Hunt's mixed equilibrium?
> No: it is an unstable rest point, the tipping point between the two basins.

> [!card]- Why does the Stag Hunt's interior rest point equal the threshold belief $p^*$?
> Under random matching the cooperator share is the probability a partner cooperates, so indifference gives the same equation.

> [!card]- What happens to a Stag Hunt population ($p^* = 2/3$) starting at 60% stag hunters?
> Stag hunting dies out: hare hunters earn more (2.6 against 2.4).

> [!card]- What condition on signalling costs makes an honest separating signal stable?
> $c_H < b < c_L$: signalling pays for high-quality senders and not for low-quality ones.

> [!card]- What happens in the signalling game when faking is cheap ($c_L = 0.5 < b$)?
> Faking and trust cycle; the signal is only about 50% reliable and receivers earn about 0.

> [!card]- Why does a uniform fee of 1 on the label eliminate deception in the signalling game?
> It pushes both signalling costs above the benefit $b$, so nobody signals at all.

> [!card]- True or false: the uniform fee in the signalling game made receivers better off.
> **False.** Receivers earned about 0 before and exactly 0 after; high-quality sellers and welfare lost.

> [!card]- Why can no behavioural test distinguish Bob's and Alice's robots without Dump?
> Both rewards give the same optimal actions in all 16 states, so behaviour is identical.

> [!card]- In which states does Alice's robot dump?
> When the bag is full and its own square is clean.

> [!card]- Why is dumping optimal under Alice's reward?
> Dump then Suck returns to the same state with +1, a guaranteed reward every two steps.

> [!card]- Why does Bob's robot never dump dirt?
> His reward counts clean squares and ignores the bag, so dirtying a square only loses reward.

> [!card]- Is Bob's wandering robot on a clean floor irrational?
> No: every action earns the maximum reward, so it is indifferent; the reward has no move cost.

## Links

- **Course:** [[AGAI - Overview|Agentic AI overview]]
- **Concepts:** [[Nash Equilibrium]] · [[Pareto Optimality]] · [[Prisoner's Dilemma]] · [[Stag Hunt]] · [[Repeated Game]] · [[Replicator Dynamics]] · [[Reward Hacking]] · [[Value Iteration]] · [[Markov Decision Process]] · [[Bellman Optimality Equation]] · [[Reward Signal]] · [[Agent Program]] · [[Multi-Agent System]]
- **Source:** `Lab1-AgenticAI-2026.ipynb`, Agentic AI 2026/2027 (UvA MSc AI), Lab 1, "Simple agents, Complex dynamics", werkcollege 28 Oct 2026. The notebook ships without outputs; all numbers above come from my own re-implementation of its models (same seeds and call order where applicable).
- **References named in the lab:** Axelrod's PD tournaments (1980) and *The Evolution of Cooperation* · Börgers and Sarin (1997), replicator dynamics as the limit of Cross learning · Russell and Norvig, *Artificial Intelligence: A Modern Approach*, Chapter 2 (vacuum world)
