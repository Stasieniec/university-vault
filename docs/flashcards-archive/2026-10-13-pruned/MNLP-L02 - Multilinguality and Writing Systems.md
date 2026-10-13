Cards removed on 2026-10-13 from the Flashcards section of Courses/MNLP/Lectures/MNLP-L02 - Multilinguality and Writing Systems.md, pruned to what the MNLP multiple-choice exam can ask.

## Not examinable

Numbers for the cheat sheet or history, with no statement about how a method works resting on them.

> [!card]- How many codes did ASCII define?
> **128**: enough for English letters, digits and punctuation.

> [!card]- How many codepoints does each Unicode plane contain?
> **65,536** ($2^{16}$), in 17 planes.

## Duplicate

Same fact as kept cards: coverage follows commercial and military interest (recall plus true/false kept); the code-page disadvantages Unicode removes (recall plus true/false kept); UTF-8 is the only ASCII-compatible encoding (recall plus the UTF-32 true/false kept).

> [!card]- Is speaker count a good predictor of how well a language is resourced for NLP?
> **No.** Coverage follows commercial and military interest; on Monz's tiers most languages with over ten million speakers are low resource.

> [!card]- Which two code-page problems does Unicode remove?
> No **table to guess** when decoding, and one document can **mix any scripts**.

> [!card]- True or false: UTF-8 is ASCII compatible.
> **True.** `U+0000` to `U+007F` are encoded as single bytes identical to ASCII.
