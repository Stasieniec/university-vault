Cards removed on 2026-10-19 from the Flashcards section of Courses/MNLP/Lectures/MNLP-L02 - Multilinguality and Writing Systems.md, when the set was pruned to what the exam (multiple choice and multiple selection) can ask.

> [!card]- According to Monz, what determines how well a language is covered by NLP resources?
> **Commercial interest** and **military relevance**.
Reason: duplicate of "True or false: How well a language is covered by NLP resources depends mainly on its number of speakers"

> [!card]- Which languages make up Monz's high-resource tier?
> English and Mandarin, then after a **big gap** French, Spanish, Arabic, Russian.
Reason: obscure list of languages per resource tier

> [!card]- How does Chinese mark past tense in 我不知道她去哪儿了 ("I don't know where she went")?
> With the **sentence-final particle** 了; the verb 去 itself is unchanged.
Reason: single-language example (Chinese sentence-final particle)

> [!card]- What does Russian *knígi* (of the book, the books, books as object) show for NLP?
> One surface form fills several cells, so the **case cannot be read off the string**.
Reason: not on slides (note commentary); the slide lists the Russian paradigm without this point

> [!card]- Why can more training data not fix a fixed word-level vocabulary for Finnish?
> Agglutination yields **combinatorially many** word forms, so unseen ones always remain. This motivates **subword segmentation**.
Reason: not on slides (note commentary); subword segmentation is motivated in later lectures

> [!card]- Which languages break the assumption that words are separated by whitespace?
> Chinese, Japanese, Thai, Lao, Khmer, Burmese.
Reason: duplicate of the multiple-selection card on whitespace

> [!card]- How does writing-system type affect a character-level vocabulary?
> Alphabets need **tens** of symbols; logographic scripts need **thousands**, which the vocabulary must budget for.
Reason: not on slides (note commentary); "thousands of symbols" is an inference

> [!card]- What is Devanagari called, where each consonant letter carries an inherent vowel?
> **Alphasyllabic**.
Reason: single-script side example

> [!card]- What does Arabic script leave out of a word such as كتب (ktb)?
> The **vowels**: only the consonant root is written.
Reason: single-script side example

> [!card]- Why is Arabic's vowel-less writing a problem no tokenizer can fix?
> The vowels were **never written**, so one written form can stand for several words; context must disambiguate.
Reason: not on slides (note commentary)

> [!card]- What do the letter shapes of Korean Hangul encode, beyond the sound itself?
> **Articulatory features**: how the sound is produced.
Reason: single-script side example

> [!card]- Why can a single "script of this text" label fail for a Japanese sentence?
> One Japanese sentence mixes **Katakana, Kanji and Hiragana**.
Reason: not on slides (note commentary); the slides do not make the mixed-script point

> [!card]- Does detecting the script of a text identify its language?
> **No.** Latin serves many languages; Arabic script covers Arabic, Persian, Urdu, Pashto, Sorani.
Reason: duplicate of the multiple-selection card on script and language (option B)

> [!card]- Does knowing a text's language tell you its script?
> **No.** Serbian is written in Cyrillic and Latin, sometimes in the same document.
Reason: duplicate of the multiple-selection card on script and language (option A)

> [!card]- What happens to a model trained on Cyrillic Serbian and tested on Latin Serbian?
> It shares **almost no letters** with the test data, although the language is the same.
Reason: not on slides (note commentary)

> [!card]- What are the two disadvantages of code pages?
> Gibberish when decoded with the **wrong table**; **one script per document**.
Reason: duplicate of "True or false: Code pages allow only one script per document"; the gibberish half is self-evident

> [!card]- Why does decoding text with the wrong code page usually raise no error?
> The bytes are often **legal in both tables**, so you silently get wrong characters (mojibake).
Reason: not on slides (note commentary); the slide only says decoding with the wrong code page gives gibberish

> [!card]- Does Unicode define how characters are stored as bytes?
> **No.** Unicode defines codepoints; storing them as bytes is a separate **encoding** question (UTF-8, UTF-16, UTF-32).
Reason: duplicate of "True or false: Unicode defines both each character's codepoint and how it is stored as bytes"

> [!card]- What follows from a codepoint lying outside plane 0, the Basic Multilingual Plane?
> It needs **more than 16 bits**, so it takes four bytes in UTF-8 and UTF-16, as emoji do.
Reason: not on slides: the link from planes to byte counts is note inference

> [!card]- Is every CJK character in the Basic Multilingual Plane (plane 0)?
> **No.** Planes 2 and 3 hold more CJK ideographs, mostly rare or historical.
Reason: obscure detail of the Unicode planes table

> [!card]- Why does a naive comparison fail on two identical-looking "café" strings?
> One is NFC (é **one precomposed codepoint**), the other NFD (e **plus combining accent**): different sequences, lengths, hashes.
Reason: duplicate of "True or false: NFC and NFD versions of "café" render identically but are unequal as strings"

> [!card]- How does the Unicode stack run from what a reader sees down to what a byte-level tokenizer eats?
> A **grapheme**; one or more **codepoints**; 1 to 4 **bytes** per codepoint.
Reason: duplicate of the codepoint, character and glyph card and "True or false: In Unicode, every character a reader perceives has exactly one codepoint"

> [!card]- Which Unicode normalization forms are lossy?
> The **compatibility** forms NFKC and NFKD; the canonical forms NFC and NFD are lossless.
Reason: duplicate of "True or false: NFKD normalization is lossless"

> [!card]- What does the K in NFKC and NFKD stand for?
> **Compatibility**: these forms also merge stylistic variants such as ligatures.
Reason: duplicate of "True or false: NFKD normalization makes the ligature ﬁ and the pair fi identical"; naming detail

> [!card]- What does NFKD do that NFD does not?
> Breaks characters into their **most basic parts**, e.g. splitting ligatures, normalizing visual styling away.
Reason: duplicate of "True or false: NFKD normalization makes the ligature ﬁ and the pair fi identical"

> [!card]- Which normalization form should you use to compare rendered text?
> **NFD** (non-lossy, preserves semantic and visual equivalence).
Reason: duplicate of "True or false: The form to use for comparing rendered text is NFD"

> [!card]- Why is NFKD's lossiness useful for search indexing?
> It unifies variants: the ligature **ﬁ matches fi**, ① becomes 1.
Reason: duplicate of "True or false: NFKD normalization makes the ligature ﬁ and the pair fi identical"; the ① example is not on slides

> [!card]- Why is NFKD not necessarily suitable for generation targets?
> A model trained on them **never sees, so cannot produce**, the characters NFKD folded away.
Reason: duplicate of the multiple-selection card on Unicode normalization (option B); the reason given is note commentary

> [!card]- What happens to the long s ſ under NFD, and under NFKD?
> NFD **keeps** ſ; NFKD replaces it with plain **s**, irreversibly.
Reason: duplicate of the multiple-selection card on Unicode normalization (option C)

> [!card]- How many bytes per codepoint does UTF-8 use?
> **1 to 4**, variable.
Reason: duplicate of the multiple-selection card on UTF-8 (option A)

> [!card]- How many UTF-8 bytes does a common CJK character such as 中 take?
> **3**.
Reason: number from a slide example; duplicate of the byte inflation card on "中文"

> [!card]- In UTF-8, what does a byte starting with `11` signal?
> The **first byte** of a multi-byte sequence; its number of leading 1s gives the **whole sequence's** length.
Reason: duplicate of the multiple-selection card on UTF-8 (option B)

> [!card]- Why does 中 (U+4E2D, 15 significant bits) need the 3-byte UTF-8 form?
> Its bits exceed the **11 payload bits** of the two-byte form; the three-byte form carries sixteen.
Reason: not on slides (bit arithmetic)

> [!card]- Why can a reader find character boundaries from any offset in a UTF-8 stream?
> No byte is ambiguous about its role: skip `10xxxxxx` continuation bytes to reach a character start.
Reason: duplicate of the multiple-selection card on UTF-8 (option D); note commentary

> [!card]- How does higher byte inflation affect byte-level tokenization?
> Text tends to be split **more aggressively**: more pieces per character.
Reason: duplicate of "True or false: Byte-level tokenizers tend to split text more aggressively in scripts that need more UTF-8 bytes per character"

> [!card]- What does `iconv -f GB2312 -t UTF-8` do if the source encoding is not correct?
> It **fails**, which beats silent mojibake.
Reason: obscure tool detail

> [!card]- What goes wrong when breaking a string at the byte level?
> Multi-byte codepoints get split into fragments that are **not characters** at all.
Reason: duplicate of the multiple-selection card on string breaking (option A)

> [!card]- What goes wrong when breaking a string at the codepoint level?
> A base letter can be **separated from its combining diacritic** (e + ◌́).
Reason: duplicate of the multiple-selection card on string breaking (option B)

> [!card]- Why is grapheme-level breaking hard?
> Grapheme boundaries are **very language-specific**.
Reason: duplicate of the multiple-selection card on string breaking (option C)

> [!card]- Where does mixing string-breaking strategies cause misalignment bugs?
> **Span annotation** (SQuAD-style offsets shift) and **counting across corpora**.
Reason: duplicate of the multiple-selection card on string breaking (option D)

> [!card]- How should a multilingual tokenizer match punctuation?
> By **Unicode general category P** (`\p{P}`), not a hand-written class like `[.!?]`, which misses 。 and ।.
Reason: not on slides (note commentary); the slide shows the \p{P} regex but not the comparison with [.!?] or these examples

> [!card]- True or false: UTF-32 is ASCII compatible.
> **False.** It uses exactly 4 bytes for every codepoint, ASCII included.
Reason: duplicate of "Which of UTF-8, UTF-16 and UTF-32 is ASCII compatible?"

> [!card]- True or false: UTF-16 encodes codepoints outside plane 0 in 4 bytes.
> **True.** They need more than 16 bits; plane-0 codepoints take 2 bytes.
Reason: not on slides: the link from planes to byte counts is note inference

> [!card]- True or false: Detecting a text's script is enough to identify its language.
> **False.** Arabic script alone covers Arabic, Persian, Urdu, Pashto and Sorani.
Reason: duplicate of the multiple-selection card on script and language (option B)

> [!card]- True or false: Deduplicating a crawled corpus that mixes NFC and NFD raises an error on the mismatched copies.
> **False.** It fails silently: both copies are kept and the numbers are just wrong.
Reason: duplicate of "Which preprocessing steps go wrong when crawled text mixes NFC and NFD?"; "fails silently" is note commentary

> [!card]- True or false: Grapheme boundaries are language-specific.
> **True.** They are very language-specific, which makes the ideal grapheme-level breaking hard.
Reason: duplicate of the multiple-selection card on string breaking (option C)

> [!card]- True or false: In UTF-8, the number of leading 1s in the first byte gives the length of the whole sequence.
> **True.** The first byte is included: `1110xxxx` starts a 3-byte sequence, so two bytes follow.
Reason: duplicate of the multiple-selection card on UTF-8 (option B)
