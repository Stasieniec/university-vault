---
type: lecture
course: 5204MNLP6Y
week: 1
lecture: 2
status: complete
topics:
  - English-centric NLP and the resource tiers
  - Typological diversity across languages
  - Morphology, from English to Russian to Finnish
  - The four English-first assumptions
  - Types of writing system
  - Script versus language
  - Code pages and mojibake
  - Unicode, codepoints, planes
  - Normalization forms NFC, NFD, NFKC, NFKD
  - UTF-8, UTF-16, UTF-32
  - Byte inflation and tokenization
  - Grapheme breaking and punctuation handling
  - Lowercasing and sorting pitfalls
---

# MNLP-L02: Multilinguality and Writing Systems

> [!abstract] In short
> Standard NLP pipelines split on whitespace, rely on word order, treat a word as one unit of meaning and assume text runs left to right. Those are properties of English, and each fails for a large group of languages. The lecture then goes down to where text enters a model: how writing systems become numbers and bytes.
>
> What a reader sees as one character (a grapheme) can be several Unicode codepoints, each stored as 1 to 4 bytes in UTF-8. The same visible text can have different codepoint sequences, so pipelines must normalize consistently, and the normalization form (NFC, NFD, NFKC, NFKD) decides what information survives. Byte-level tokenizers see bytes, so scripts with more bytes per character get split into more tokens.
>
> No sample exam question covers this lecture. Expect statements on exact conditions: which normalization forms are lossy, which encodings are ASCII compatible, which languages break which assumption.

## NLP mostly means English NLP

Most NLP resources are in English: **annotated (labelled) training data, unannotated training data, toolkit coverage** (dictionaries, analyzers) and **benchmarks and competitions**. Most publications also focus on English.

Non-English resources arise mostly:

- **as a naturally occurring by-product**: translations, product reviews, which nobody made for NLP;
- **as translations of existing (annotated) English resources**, so the annotation scheme and content stay English-shaped;
- **less so as a dedicated effort**, the route that would actually close the gap.

### Resource tiers follow money

There are **over 7,000 spoken languages**: a long tail, though more than a hundred have very large speaker populations. How well a language is covered by NLP resources does not follow how many people speak it. It depends on **commercial interest** and **military relevance**. Christof Monz (the lecturer) sorts languages into three tiers:

**High resource:** large labelled and unlabelled data, many tools. English and Mandarin Chinese, then a **big gap**, then French, Spanish, Arabic and Russian. So even the top tier is two very different groups.

**Medium resource:** limited labelled data, reasonable unlabelled data, some tools. German, Italian, Dutch, Korean, Japanese (European and Asian languages).

**Low resource:** no labelled data, limited unlabelled data, no tools. All of the roughly 7,000 remaining languages.

## Languages differ in ways English hides

### One sentence, two languages

> [!example] How many words is this?
> **English:** I don't know where she went.
>
> **Chinese:** 我不知道她去哪儿了。 Chinese has (almost) no spaces, so splitting on whitespace returns the whole sentence as one token.
>
> | 我 | 不 | 知道 | 她 | 去 | 哪儿 | 了 | 。 |
> |---|---|---|---|---|---|---|---|
> | wǒ | bù | zhīdào | tā | qù | nǎr | le | |
> | I | NEG | know | she | go | where | SFP | PUNC |

(NEG: negation; SFP: sentence-final particle.) The gloss shows:

- **The verb 去 ("go") carries no tense.** English changes the verb ("go" against "went"); Chinese leaves it alone.
- **Past tense comes from the sentence-final particle 了**, at the end of the clause, away from the verb.
- **Both clauses are SVO** (subject, verb, object): the main clause "I NEG know" and the embedded clause "she go where". Word order matches English here; word boundaries and the place of tense do not.

### Morphology: what is inside a word

> [!definition] Morphology
> **Morphology** is the study of the internal structure of words and how they are formed from smaller units of meaning.
> - **Morpheme:** the smallest unit of meaning or grammatical function. *dogs* has two: *dog* (the base meaning) and *-s* (plural).
> - **Root / base:** the core part of a word that carries the primary meaning (*friend* in *unfriendly*).
> - **Affix:** a prefix (added to the front) or suffix (added to the back) that modifies a root's meaning or grammatical role (*un-* and *-ly* in *unfriendly*).

The lecture runs three languages of increasing morphological complexity.

**English has limited morphology.** From *happy* you get *happier, happiest, unhappy, unhappier, unhappiest, happiness, unhappiness*, and little more. That is why closed (fixed-size) word vocabularies seemed a reasonable design.

**Russian has more complex morphology.** The noun *kniga* ("book") changes its ending for six cases (grammatical roles such as subject, possessor, instrument) in singular and plural: *kníga, knígi, knígye, knígu, knígoy, knig, knígam, knígami, knígakh*. Some forms fill several cells: *knígi* is "of the book", "the books" and "books" as direct object. So the case cannot always be read off the string.

**Finnish has very complex morphology.** It is **agglutinative**: morphemes are glued on one after another, each adding one piece of meaning, so a single word can carry what English spreads over a phrase.

> [!example] Building "even at my happiest" in Finnish
> *onni* ("happiness") + *-llinen* (adjective-forming suffix, "having the quality of") gives *onnellinen*, "happy". Before further suffixes the stem becomes *onnellis-*, so *onnellis-* + *-empi* (comparative) gives *onnellisempi*, "happier". Then:
>
> ```
> onnellis  -imm-  -i-   -lla   -ni     -kin
> │         │      │     │      │       │
> │         │      │     │      │       └─ "even"
> │         │      │     │      └───────── 1st person singular possessive, "my"
> │         │      │     └──────────────── adessive case, "at/on/in"
> │         │      └────────────────────── plural oblique-case marker
> │         └───────────────────────────── superlative marker
> └─────────────────────────────────────── stem, "happy"
> ```
>
> *onnellisimmillanikin* = "even at my happiest": one word for four English ones.

A fixed word-level vocabulary has almost certainly never seen *onnellisimmillanikin*, and Finnish produces combinatorially many words like it, so more data does not fix this. That motivates subword segmentation ([[MNLP-L04 - Subword Segmentation|L04]]); morphology itself is [[MNLP-L03 - Morphology and Word Formation|L03]].

## The four English-first assumptions

Each assumption is invisible in an English pipeline and false for a large group of languages.

**1. Words are separated by whitespace.** False for **Chinese, Japanese, Thai, Lao, Khmer and Burmese**, which have no word delimiter. Thai does use spaces, but to end **clauses or sentences** (it has no dedicated full stop), so a Thai space does not mark a word boundary.

**2. Grammatical relations are encoded by word order.** False for languages with **rich morphology, especially case marking**: **Russian, Finnish, Korean**. Who did what to whom is marked on the nouns themselves, so word order is free and a word's position says much less about its role.

**3. One word = one unit of meaning.** False for **agglutinative** languages (**Finnish, Turkish, Hungarian**) and for languages with **compounding** (**German, Dutch, Swedish**). German *Bundesverfassungsgericht* is one token holding three concepts: federal, constitutional, court.

**4. Text flows from left to right.** False for **Semitic languages and languages in Arabic script**: **Arabic, Hebrew, Urdu, Persian**, which are read right to left. Unicode text in these languages is **written (stored) left to right but rendered right to left**: codepoints sit in the file in reading order and are displayed from the right, so the first character of the string is the rightmost on screen.

### Why the assumptions are built in

- The overwhelming majority of NLP research and resources target **a small handful of languages**.
- Standard pipelines encode assumptions that are **properties of English, not of language in general**: whitespace tokenization, fixed word order, closed vocabularies.
- **Claims of language independence are usually untested** outside English (or a very small set of languages).
- This is **understandable**: creating resources costs time and money.

## Writing systems: what a symbol stands for

> [!definition] Types of writing system
> - **Alphabetic:** each symbol represents a **consonant or vowel sound**; typically a **small inventory** of symbols. Latin, Cyrillic, Hangul.
> - **Syllabic:** each symbol represents a **whole syllable**, a combination of consonant and vowel. Japanese Hiragana and Katakana.
> - **Logographic:** each symbol represents **words or morphemes directly**, with typically **little sound information**. Chinese Hanzi, Japanese Kanji.

For NLP the practical difference is inventory size: an alphabet has tens of symbols, a logographic script thousands, which a character-level vocabulary must budget for.

| Script | Word | Reading | What the symbols encode |
|---|---|---|---|
| Latin | water | water | alphabetic: sounds spelled with letters |
| Devanagari | पानी | pani | alphasyllabic: consonant + inherent vowel |
| Han | 水 | shui | logographic: one symbol, one morpheme |
| Arabic | كتب | ktb | root only, vowels omitted |
| Korean | 한글 | Hangul | letter shapes encode articulation |

**Devanagari** (Hindi) is **alphasyllabic**: each consonant letter carries a built-in vowel, which an attached vowel sign can replace; पानी is two units, *pā* + *nī*. (Outside the course material such scripts are called abugidas.)

**Arabic** writes the consonant root and omits the vowels: كتب (k-t-b, "write") can be read *kataba* ("he wrote"), *kutiba* ("it was written") or *kutub* ("books"). (Outside the course material: an abjad.) The writing is lossier than the language, so the model must resolve the ambiguity from context; no tokenizer can recover vowels that were never written.

**Korean Hangul** is alphabetic (each letter a consonant or vowel), and its letter shapes depict how the sound is articulated. Outside the course material: the letters are grouped into square syllable blocks, so it looks syllabic while working alphabetically.

A single Japanese sentence can mix three systems: in アムステルダムはオランダの首都です ("Amsterdam is the capital of the Netherlands"), the names are Katakana, 首都 ("capital") is Kanji and は, の, です are Hiragana. One "script of this text" label cannot describe it.

## Script is not language

**One writing system, many languages:**
- **Latin:** English, German, French, most European languages (except some Eastern European ones), and also several African, Turkic and Asian languages
- **Arabic:** Arabic, Persian, Urdu, Pashto, Sorani
- **Cyrillic:** Russian, Belarusian, Serbian
- **Devanagari:** Hindi, Marathi, Nepali
- **Ethiopic (Ge'ez):** Amharic, Tigrinya

**Language-specific writing systems:** Armenian, Georgian, Greek, Sinhala, Khmer.

**One language, more than one writing system:** Serbian (Cyrillic and Latin), Kurdish (Latin and Arabic), Uzbek (Latin and Cyrillic), Punjabi (Gurmukhi and Arabic).

> [!tip] What each direction breaks
> **Detecting the script does not identify the language.** Latin script narrows almost nothing; Arabic script still leaves Arabic, Persian, Urdu, Pashto and Sorani.
>
> **Knowing the language does not fix the script.** A Serbian corpus can contain Cyrillic and Latin text, even in the same document. A model trained on one script and tested on the other sees no shared characters, though the language is the same.

## Before Unicode: ASCII and code pages

**ASCII** (1963) assigned **128 codes**: enough for English letters, digits and punctuation.

**Code pages** extended this by reinterpretation. They use the same 128 (or 256) codes that fit in one byte, and **interpret them differently depending on the language or script**: Latin-1 for Western Europe, Shift-JIS for Japanese, GBK for Chinese, KOI8-R for Russian, and many more.

Two disadvantages:

- **Gibberish when text encoded for one script is decoded as another.** The bytes are legal in both tables, so usually no error is raised (the result is called **mojibake**).
- **No mixing of scripts in one document**, since a document is read with one table.

> [!example] Same bytes, two readings
> The bytes `D0 9F D1 80 D0 B8 D0 B2 D0 B5 D1 82` decoded as UTF-8 give Привет (Russian "hello"). Decoded with the Western European code page Windows-1252, they give ÐŸÑ€Ð¸Ð²ÐµÑ‚, with no error.

## Unicode: one number for every character

> [!definition] Unicode and codepoints
> **Unicode** is a **single, universal character set**, maintained by the Unicode Consortium, in which **every character, in every supported script, gets exactly one unique number**. That number is the character's **codepoint**, written `U+number` in hexadecimal: `U+0041` is Latin "A", `U+4E2D` is the Han character 中.

- **Unicode defines meaning and leaves bytes to an encoding.** A codepoint is an abstract identifier. How to store it as bytes on disk or in memory is a separate encoding question, answered by UTF-8, UTF-16 or UTF-32.
- **Unicode is a superset of ASCII:** `U+0000` to `U+007F` are identical to ASCII. So English-only documents were Unicode compatible from the start.

**Scale.** The code space runs from `U+0000` to `U+10FFFF`, over 1.1 million possible codepoints, in **17 planes of $2^{16} = 65{,}536$ codepoints each**. Plane 0, the **Basic Multilingual Plane** (BMP), holds most modern languages, including the common CJK (Chinese, Japanese, Korean) characters. Plane 1 holds historic scripts, symbols and **emoji**. Anything beyond plane 0 needs more than 16 bits, which is why it takes 4 bytes in both UTF-8 and UTF-16.

## Codepoint, character, glyph, grapheme

> [!definition] Three levels that are easy to conflate
> - **Codepoint:** an abstract number, such as `U+00E9`; Unicode's unit of identity.
> - **Character:** what a reader perceives as one unit. It may be built from **several codepoints**.
> - **Glyph:** a stylistic rendering of a character (how a font draws it).
>
> What a reader perceives as one character is also called a **grapheme**.

> [!example] Two identical-looking "café"
> ```
> café    U+0063 U+0061 U+0066 U+00E9            4 codepoints
> café    U+0063 U+0061 U+0066 U+0065 U+0301     5 codepoints
> ```
> In the first, é is one **precomposed** codepoint; in the second, plain e plus a **combining** acute accent. They render identically but differ in equality, length and hash: **a naive string comparison will not match them.** In JavaScript, `'ã' === 'ã'` can return false, with lengths 1 and 2.

## Normalization: choosing one sequence for the same text

**Normalization** rewrites text into a standard codepoint sequence. Four forms come from two choices: **composed** (C) or **decomposed** (D: base character plus combining marks), and **canonical** (merges only sequences for the same character) or **compatibility** (K: also merges stylistic variants, such as a ligature and its letters).

| | Composed | Decomposed |
|---|---|---|
| **Canonical** (reversible) | NFC | NFD |
| **Compatibility** (lossy) | NFKC | NFKD |

The two "café" strings above are the NFC and NFD versions of one word.

> [!example] Where compatibility forms lose information
> The source character is long s with dot above plus a combining dot below (ẛ̣).
> - **NFC:** ẛ + combining dot below (2 codepoints)
> - **NFD:** long s ſ + combining dot below + combining dot above (3 codepoints)
> - **NFKC:** ṩ, a plain s with both dots (1 codepoint)
> - **NFKD:** plain s + dot below + dot above (3 codepoints)
>
> The canonical forms keep the long s ſ, because canonically ſ and s are different characters. The compatibility forms replace it with plain s, and nothing can bring it back.

> [!definition] The decomposition forms and when to use them
> - **NFD (canonical decomposition):** splits characters into **base characters and combining marks**; preserves semantic and visual equivalence. **Non-lossy**: you can decompose and compose again without losing styling or formatting details.
> - **NFKD (compatibility decomposition):** breaks characters down into their **most basic parts, including splitting ligatures**. **Lossy**: it normalizes visual styling to the base form, which cannot be reconstructed when composing.
> - **Use NFD to compare rendered text.**
> - **Use NFKD for search indexing or NLP analysis**, but not necessarily for generation.

Why: for analysis, merging variants helps (the ligature ﬁ should match "fi" in a search index, ① should count as 1). For generation it can hurt: a model trained on NFKD targets never sees, and so cannot produce, the characters that were folded away.

## Storing codepoints as bytes: UTF-8, UTF-16, UTF-32

| Encoding | Bytes per codepoint | ASCII compatible | Used by |
|---|---|---|---|
| **UTF-8** | 1 to 4 | yes | the web, Linux, most file formats, nearly all NLP tooling |
| **UTF-16** | 2 or 4 | no: at least 2 bytes each | Java, JavaScript strings, Windows, .NET |
| **UTF-32** | exactly 4 | no: exactly 4 bytes each | rare; simple to index, memory-hungry |

**UTF-8 is by far the most common**; it is what people usually mean by "Unicode".

### How UTF-8 works

UTF-8 uses more bytes for larger codepoints. The first byte says how many bytes follow, and every following byte carries 6 bits of the codepoint.

> [!definition] The three UTF-8 byte rules
> - A byte starting with **`0`** is a **single byte**, encoding one of 128 characters: ASCII.
> - A byte starting with **`11`** is the **first byte of a multi-byte sequence**; the **number of leading 1s** gives the sequence length.
> - A byte starting with **`10`** is a **continuation byte** inside a multi-byte sequence.

```pseudo
Algorithm: encode one codepoint c in UTF-8
───────────────────────────────────────────
if c ≤ U+007F:     pattern 0xxxxxxx                              // 7 payload bits
elif c ≤ U+07FF:   pattern 110xxxxx 10xxxxxx                     // 5+6 = 11 bits
elif c ≤ U+FFFF:   pattern 1110xxxx 10xxxxxx 10xxxxxx            // 4+6+6 = 16 bits
else (≤ U+10FFFF): pattern 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx   // 3+6+6+6 = 21 bits
write c in binary, left-padded with zeros to the pattern's payload width
fill the x positions left to right with those bits
```

> [!example] 中, U+4E2D
> 0x4E2D in binary is `100111000101101`, 15 bits. That is more than the 2-byte form's 11, so use the 3-byte form with 16 payload bits. Pad to `0100111000101101` and split 4 + 6 + 6: `0100 | 111000 | 101101`.
>
> Byte 1 = `1110` + `0100` = `11100100`
> Byte 2 = `10` + `111000` = `10111000`
> Byte 3 = `10` + `101101` = `10101101`, so 中 is `E4 B8 AD`.

By the same rules, A (`U+0041`) is one byte identical to ASCII, é (`U+00E9`) two bytes, most CJK characters three, and a plane-1 emoji such as `U+1F600` four.

Since no byte is ambiguous about its role, you can drop into a UTF-8 stream anywhere and find the next character start by skipping continuation bytes (called self-synchronisation, a term from outside the course material).

## What bytes do to tokenizers: byte inflation

**Byte-level tokenizers** (vocabulary built over UTF-8 bytes, see [[MNLP-L04 - Subword Segmentation|L04]]) are affected by how characters are represented as bytes, not as codepoints.

> [!formula] Byte inflation
> $$\text{inflation}(s) = \frac{\lvert \text{UTF-8 bytes}(s) \rvert}{\lvert \text{characters}(s) \rvert}$$
>
> where:
> - $\lvert \text{UTF-8 bytes}(s) \rvert$: the length of string $s$ once encoded as UTF-8
> - $\lvert \text{characters}(s) \rvert$: the number of characters a reader perceives

> [!example] Three strings
> "hello": 5 bytes for 5 characters, 1.0x. "café": 5 bytes for 4 characters (é takes two), 1.25x. "中文": 6 bytes for 2 characters, 3.0x.

Two consequences:

- **The more bytes are needed, the more aggressively text tends to be split into tokens.** The same meaning costs more tokens in a script with more bytes per character.
- **Byte-level splitting can produce non-existent characters.** Cut a 3-byte character after its second byte and neither piece is a valid character.

An implication beyond the lecture's own claims: CJK users pay more tokens for the same content, so a shorter effective context window and more compute.

## Normalizing crawled data

**Resources crawled from the internet can mix normalization forms even when they are all in Unicode**: some pages NFC, others NFD.

Many preprocessing steps **assume identical codepoint sequences**: **deduplication**, **search**, **string equality and prefix matching**. If the text is not normalized properly, these steps become **inaccurate**, silently: a deduplicated corpus keeps two copies of a document that differ only in normalization form, and no error is raised.

**The easy fix: always perform consistent normalization as part of the pipeline**, for example first decompose and then compose. In Python, the `unicodedata` module can **normalize** a string into a given form and **test** whether it is already in one:

```python
import unicodedata

a = "caf\u00e9"     # NFC: 4 codepoints
b = "cafe\u0301"    # NFD: 5 codepoints

a == b                                    # False
unicodedata.is_normalized("NFC", b)       # False
unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b)   # True
```

Text that is not in Unicode at all, but in a code page such as GB2312 or KOI8-R, is converted on the way in, for example with `iconv -f GB2312 -t UTF-8`. This **fails if the source encoding is wrong**, which is useful: a failure is better than silent mojibake.

## Where to break a string

Splitting a string can happen at three levels, each with a cost:

- **Byte-level breaking can split codepoints**, which can consist of several bytes, producing fragments that are not characters.
- **"Smarter" codepoint breaking is not ideal either**: it can break one character into pieces, for example separating a base letter from its diacritic.
- **Ideally, break at the grapheme level** (what the reader perceives as a character), but grapheme boundaries are **very language-specific**.

```
  "café" in NFD, broken at three levels

  bytes:       63 | 61 | 66 | 65 | CC | 81    the last two are not characters
  codepoints:  c  | a  | f  | e  | ◌́          the accent is stranded
  graphemes:   c  | a  | f  | é               what the reader sees
```

**Mixing breaking strategies leads to misalignments**:

- **Span annotation systems** (SQuAD-style question answering stores answers as character offsets): offsets counted one way and read another give shifted answers.
- **Any counting across corpora** that used different breaking strategies: lengths and counts are no longer comparable.

So breaking and length computation must be **consistent across documents**.

## Other pitfalls: case, sorting, punctuation

**Lowercasing is language-dependent.** In English, lowercase(I) = i. In Turkish, lowercase(I) = ı (dotless i), because Turkish has two pairs: dotted İ / i and dotless I / ı. Lowercasing Turkish with English rules changes the word.

**Codepoint order is not alphabetical order**: accented letters' codepoints are not necessarily near their base letter's.

**Languages use different alphabetical orders.** Swedish puts å, ä, ö **after z**, as separate letters. German sorts ä as ae, ö as oe, ü as ue, **interleaved** with the plain letters.

**Punctuation.** Defining a word is hard, but **punctuation adjacent to a word is not part of it**. Tokenizers split it off, and may also expand contractions, a further, lossy normalization decision:

```
He'd say "I promise you!", but then disappear.
→ He ' d say " I promise you ! " , but then disappear .
→ He would say " I promise you ! " , but then disappear .
```

Punctuation is matched as a **general category** rather than a hand-written list: the Unicode property `\p{P}` covers punctuation in every script, and `re.sub(r'(\p{P})', r' \1 ', string)` puts spaces around each mark. (Outside the course material: Python's built-in `re` lacks `\p{P}`; the third-party `regex` module has it.) A class such as `[.!?]` misses the Japanese 。, the Bengali ।, the Amharic ።.

**Punctuation normalization** (“...” against «...» against 「...」) has **no simple solution: the mappings must be encoded by hand**. It comes up in quotes, end-of-sentence markers, question marks and single quotes.

## The Unicode stack in one picture

> [!tip] Recap
> A reader sees a **grapheme**, which is represented by **one or more codepoints**, each encoded in **1 to 4 bytes** (UTF-8).
> ```
>   what the reader sees      é                1 grapheme
>   what Unicode identifies   U+0065 U+0301    2 codepoints (NFD)
>   what the tokenizer eats   65 CC 81         3 bytes (UTF-8)
> ```

## What to be able to answer

- The four English-first assumptions and the languages that break each (Thai spaces end clauses).
- What sets a language's resource tier; the three tiers; why "language independent" claims are suspect.
- Alphabetic, syllabic, logographic; the Devanagari, Arabic and Hangul cases.
- Script and language in both directions.
- The two disadvantages of code pages.
- Codepoint, character, glyph, grapheme; why two "café" strings fail to match.
- The four normalization forms: which are lossy, which for rendered text, which for search.
- UTF-8, UTF-16, UTF-32: bytes and ASCII compatibility; the UTF-8 byte rules; encoding by hand.
- Byte inflation and its two consequences.
- What breaks silently without normalization, and the fix.
- Byte, codepoint and grapheme breaking; where mixing them misaligns.
- Turkish lowercasing, Swedish and German sorting, punctuation handling.

Model answers are in `## Exam questions` below.

## Exam questions

> [!exam]- State the four English-first assumptions built into standard NLP pipelines. For each, give languages where it fails and what goes wrong concretely.
> **Key points:** whitespace word boundaries fail for Chinese, Japanese, Thai; word-order grammar fails for case-marking Russian, Finnish, Korean; one word one meaning fails for agglutinative and compounding languages; left-to-right fails for Arabic, Hebrew, Urdu, Persian.
>
> 1. **Words are separated by whitespace.** Fails for Chinese, Japanese, Thai, Lao, Khmer, Burmese: there is no word delimiter, so whitespace tokenization returns one token for a whole Chinese sentence. Thai uses spaces to end **clauses or sentences** and has no dedicated full stop, so a space does not mark a word boundary.
> 2. **Grammatical relations are encoded by word order.** Fails for languages with rich morphology, especially case marking (Russian, Finnish, Korean): who did what to whom is marked on the noun, word order is free, and position says much less about a word's role.
> 3. **One word = one unit of meaning.** Fails for agglutinative languages (Finnish, Turkish, Hungarian) and compounding languages (German, Dutch, Swedish): *Bundesverfassungsgericht* is one token holding federal + constitutional + court.
> 4. **Text flows left to right.** Fails for Semitic languages and languages in Arabic script (Arabic, Hebrew, Urdu, Persian): the text is stored left to right but rendered right to left, so the first character in the string is the rightmost on screen.

> [!exam]- Argue that "script is not language" holds in both directions, and explain what each direction breaks in an NLP pipeline.
> **Key points:** one script serves many languages (Latin, Arabic); script detection does not identify the language; one language uses several scripts (Serbian, Uzbek); language does not fix the script, so train and test scripts can share no characters.
>
> - **One script, many languages:** Latin (English, German, French, most European languages, several African, Turkic and Asian languages), Arabic (Arabic, Persian, Urdu, Pashto, Sorani), Cyrillic (Russian, Belarusian, Serbian), Devanagari (Hindi, Marathi, Nepali), Ethiopic (Amharic, Tigrinya). So **detecting the script does not identify the language**: Latin narrows almost nothing, and Arabic script still leaves five candidates.
> - **One language, many scripts:** Serbian (Cyrillic and Latin), Kurdish (Latin and Arabic), Uzbek (Latin and Cyrillic), Punjabi (Gurmukhi and Arabic). So **knowing the language does not tell you the script**: a Serbian corpus can mix both, even in one document, and a model trained on one script and tested on the other sees no shared characters for the same language.
> - Some scripts are language-specific (Armenian, Georgian, Greek, Sinhala, Khmer).
> - Also: one sentence can mix several writing systems (Japanese uses Katakana, Kanji and Hiragana together), so a single "script of this text" label can fail.

> [!exam]- Why is the English-centric bias of NLP structural, and why do claims of "language independence" deserve suspicion?
> **Key points:** resources exist mostly in English; non-English data are by-products or translations of English resources; coverage follows commercial and military interest, not speaker numbers; pipelines encode English properties; language independence is rarely tested.
>
> - Most resources are in English: annotated data, unannotated data, toolkit coverage (dictionaries, analyzers), benchmarks. Most publications also focus on English.
> - Non-English resources arise mostly as **by-products** (translations, product reviews) or as **translations of English annotated resources** (so the annotation scheme stays English-shaped), less so as dedicated efforts.
> - Coverage tracks **commercial interest and military relevance**. Over 7,000 languages exist; only English and Mandarin are strongly covered, then (after a big gap) French, Spanish, Arabic, Russian; German, Dutch, Japanese and Korean are only medium resource; the rest are low resource.
> - Standard pipelines encode properties of English (whitespace tokenization, fixed word order, closed vocabularies) as if they were properties of language.
> - "Language independent" claims are usually untested outside English or a very small set of languages.
> - The bias is understandable, since resource creation costs time and money.

> [!exam]- Compare alphabetic, syllabic and logographic writing systems and explain what each, plus the Devanagari, Arabic and Hangul cases, implies for NLP.
> **Key points:** alphabetic: sounds, small inventory; syllabic: whole syllables; logographic: words or morphemes, thousands of symbols; Devanagari alphasyllabic; Arabic omits vowels, so the model must disambiguate; Hangul letter shapes encode articulation.
>
> - **Alphabetic:** each symbol is a consonant or vowel sound, small inventory (Latin, Cyrillic, Hangul).
> - **Syllabic:** each symbol is a whole syllable (Hiragana, Katakana).
> - **Logographic:** each symbol is a word or morpheme directly, little sound information (Chinese Hanzi, Japanese Kanji).
> - **Inventory size** sets the cost of a character: tens of symbols for an alphabet, thousands for a logographic script, which a character-level vocabulary must budget for.
> - **Devanagari** is alphasyllabic: a consonant letter carries an inherent vowel that a vowel sign can replace.
> - **Arabic** writes the consonant root and omits vowels: كتب can be *kataba*, *kutiba* or *kutub*, so the writing is lossier than the language and the model must resolve the ambiguity from context. No tokenizer can recover vowels that were never written.
> - **Hangul** is alphabetic, and its letter shapes encode articulatory features.

> [!exam]- Explain the difference between a grapheme, a codepoint and a byte, and show how confusing them corrupts tokenization, deduplication and span annotation.
> **Key points:** grapheme is the perceived character; one or more codepoints per grapheme; 1 to 4 UTF-8 bytes per codepoint; NFC/NFD mismatch silently breaks dedup and search; mixed breaking conventions misalign spans.
>
> - **Grapheme:** what the reader sees as one character. **Codepoint:** Unicode's abstract number, one or more per grapheme. **Byte:** UTF-8 stores each codepoint in 1 to 4 bytes.
> - Example: NFD é is 1 grapheme, 2 codepoints (U+0065 U+0301), 3 bytes (65 CC 81).
> - **Tokenization:** byte-level tokenizers see the byte stream. Byte-level splits can produce non-existent characters; codepoint splits can strand a diacritic from its base letter.
> - **Deduplication, search, equality, prefix matching** assume identical codepoint sequences, so NFC and NFD copies of the same text silently fail to match.
> - **Span annotation** (SQuAD-style QA): offsets recorded under one breaking convention and read under another give shifted answers. Counts across corpora with different conventions are not comparable.
> - Fix: normalize consistently in the pipeline, and keep breaking and length computation consistent across documents.

> [!exam]- Crawled multilingual data mixes normalization forms. Which Unicode normalization form would you use for search indexing, for comparing rendered text, and for generation targets, and why?
> **Key points:** canonical reversible, compatibility lossy; NFD for comparing rendered text; NFKD for search indexing and analysis; NFKD not necessarily for generation; normalize consistently.
>
> - The four forms form a grid: canonical (reversible) NFC and NFD; compatibility (lossy) NFKC and NFKD; composed (C) against decomposed (D).
> - **Comparing rendered text:** NFD, which splits into base characters plus combining marks and preserves semantic and visual equivalence. It is non-lossy, so you can decompose and recompose.
> - **Search indexing or NLP analysis:** NFKD. Its lossiness helps: the ligature ﬁ and the pair fi become the same, ① becomes 1.
> - **Generation:** NFKD is not necessarily suitable. A model trained on NFKD targets never sees, so cannot produce, the folded-away characters (for example the long s ſ becomes plain s).
> - Whatever you choose, apply it consistently in the pipeline (for example decompose then compose), otherwise dedup, search and equality checks are silently inaccurate.
>
> Losing marks: getting the analysis against generation rule backwards, or claiming NFKD is reversible.

> [!exam]- Explain byte inflation for byte-level tokenizers and compute it for "hello", "café" and "中文". What are its consequences?
> **Key points:** UTF-8 bytes over perceived characters; 1.0x, 1.25x, 3.0x; more bytes means more aggressive splitting; byte splits can produce non-existent characters.
>
> - $\text{inflation}(s) = \lvert\text{UTF-8 bytes}(s)\rvert \,/\, \lvert\text{characters}(s)\rvert$, characters meaning what a reader perceives.
> - "hello": 5 bytes / 5 = **1.0x**. "café" (NFC, é is 2 bytes): 5 / 4 = **1.25x**. "中文" (3 bytes each): 6 / 2 = **3.0x**.
> - More bytes per character means more aggressive splitting, so the same meaning costs more tokens; byte splits can also produce fragments that are not characters at all.
> - An implication beyond the lecture's own claims: CJK users pay more tokens for the same sentence, so a shorter effective context window and more compute.

> [!exam]- Encode U+4E2D (中) in UTF-8 by hand, stating which byte rules you use.
> **Key points:** 15 bits exceed 11, so 3-byte form; 1110xxxx 10xxxxxx 10xxxxxx; result E4 B8 AD; lead and continuation byte rules.
>
> - 0x4E2D = `100111000101101`: 15 bits, more than the 11 payload bits of the 2-byte form, so use the 3-byte form `1110xxxx 10xxxxxx 10xxxxxx` with $4 + 6 + 6 = 16$ payload bits.
> - Pad to 16 bits and split 4 + 6 + 6: `0100 | 111000 | 101101`.
> - Bytes: `11100100 10111000 10101101` (hex E4 B8 AD).
> - Rules: a byte starting `0` is a single byte (ASCII); a byte starting `11` begins a multi-byte sequence, with the number of leading 1s giving its length; a byte starting `10` is a continuation byte.
> - Because no byte is ambiguous about its role, a reader can find the next character start from any offset by skipping `10xxxxxx` bytes.

## Flashcards

> [!card]- According to Monz, what determines how well a language is covered by NLP resources?
> **Commercial interest** and **military relevance**.

> [!card]- Is speaker count a good predictor of how well a language is resourced for NLP?
> **No.** Coverage follows commercial and military interest; Dutch, for example, is only medium resource.

> [!card]- Which languages make up Monz's high-resource tier?
> English and Mandarin, then after a **big gap** French, Spanish, Arabic, Russian.

> [!card]- How do most non-English NLP resources come about?
> As **by-products** (translations, reviews) or **translations of English resources**; rarely as dedicated efforts.

> [!card]- Why do claims that an NLP method is "language independent" deserve suspicion?
> They are usually **untested** outside English or a very small set of languages.

> [!card]- How does Chinese mark past tense in 我不知道她去哪儿了 ("I don't know where she went")?
> With the **sentence-final particle** 了; the verb 去 itself is unchanged.

> [!card]- What does Russian *knígi* (of the book, the books, books as object) show for NLP?
> One surface form fills several cells, so the **case cannot be read off the string**.

> [!card]- What does it mean for a language such as Finnish to be agglutinative?
> Morphemes are **glued on in sequence**, each adding one piece of meaning.

> [!card]- Why can more training data not fix a fixed word-level vocabulary for Finnish?
> Agglutination yields **combinatorially many** word forms, so unseen ones always remain. This motivates **subword segmentation**.

> [!card]- Which four English-first assumptions are built into standard NLP pipelines?
> Words separated by **whitespace**; grammar encoded by **word order**; **one word = one meaning**; text flows **left to right**.

> [!card]- Which languages break the assumption that words are separated by whitespace?
> Chinese, Japanese, Thai, Lao, Khmer, Burmese.

> [!card]- Why do word-position features carry little information in Russian, Finnish or Korean?
> Grammatical relations are marked by **case on the noun**, so word order is free.

> [!card]- Which two kinds of language break the assumption "one word = one unit of meaning"?
> **Agglutinative** languages (Finnish, Turkish, Hungarian) and **compounding** languages (German, Dutch, Swedish).

> [!card]- In Arabic or Hebrew text, is the first character of the Unicode string the leftmost one on screen?
> **No.** Text is stored left to right in reading order but **rendered right to left**.

> [!card]- Which type of writing system has each symbol represent a consonant or vowel sound?
> **Alphabetic** (Latin, Cyrillic, Hangul), typically a small inventory.

> [!card]- What does each symbol represent in a syllabic writing system such as Hiragana or Katakana?
> A **whole syllable**, a combination of consonant and vowel.

> [!card]- What does each symbol represent in a logographic writing system such as Chinese Hanzi?
> A **word or morpheme** directly, with little sound information.

> [!card]- How does writing-system type affect a character-level vocabulary?
> Alphabets need **tens** of symbols; logographic scripts need **thousands**, which the vocabulary must budget for.

> [!card]- What is Devanagari called, where each consonant carries an inherent vowel that a vowel sign can replace?
> **Alphasyllabic** (outside the course material also called an abugida).

> [!card]- What does Arabic script leave out of a word such as كتب (ktb)?
> The **vowels**: only the consonant root is written.

> [!card]- Why is Arabic's vowel-less writing a problem no tokenizer can fix?
> One written form (كتب: *kataba*, *kutiba*, *kutub*) maps to several words; the vowels were **never written**, so context must disambiguate.

> [!card]- What do the letter shapes of Korean Hangul encode, beyond the sound itself?
> **Articulatory features**: how the sound is produced.

> [!card]- Why can a single "script of this text" label fail for a Japanese sentence?
> One Japanese sentence mixes **Katakana, Kanji and Hiragana**.

> [!card]- Does detecting the script of a text identify its language?
> **No.** Latin serves many languages; Arabic script covers Arabic, Persian, Urdu, Pashto, Sorani.

> [!card]- Does knowing a text's language tell you its script?
> **No.** Serbian is written in Cyrillic and Latin, sometimes in the same document.

> [!card]- What happens to a model trained on Cyrillic Serbian and tested on Latin Serbian?
> It sees **no shared characters**, although the language is the same.

> [!card]- How many codes did ASCII define?
> **128**: enough for English letters, digits and punctuation.

> [!card]- How did code pages extend ASCII?
> By **reinterpretation**: the same 128 or 256 byte codes, read differently per language or script.

> [!card]- What are the two disadvantages of code pages?
> Gibberish when decoded with the **wrong table**; **one script per document**.

> [!card]- Why does decoding text with the wrong code page usually raise no error?
> The bytes are **legal in both tables**, so you silently get wrong characters (mojibake).

> [!card]- What is Unicode?
> A **single universal character set** giving every character in every supported script one unique number, its **codepoint** (written `U+number`).

> [!card]- Does Unicode define how characters are stored as bytes?
> **No.** Unicode defines codepoints; storing them as bytes is a separate **encoding** question (UTF-8, UTF-16, UTF-32).

> [!card]- How does Unicode relate to ASCII?
> Unicode is a **superset**: `U+0000` to `U+007F` are identical to ASCII.

> [!card]- How many codepoints does each Unicode plane contain?
> **65,536** ($2^{16}$), in 17 planes.

> [!card]- What follows from a codepoint lying outside plane 0, the Basic Multilingual Plane?
> It needs **more than 16 bits**, so it takes four bytes in UTF-8 and UTF-16, as emoji do.

> [!card]- What is the difference between a codepoint, a character and a glyph?
> Codepoint: abstract **number**. Character: what a reader **perceives**, possibly several codepoints. Glyph: a stylistic **rendering**.

> [!card]- Why does a naive comparison fail on two identical-looking "café" strings?
> One is NFC (é **one precomposed codepoint**), the other NFD (e **plus combining accent**): different sequences, lengths, hashes.

> [!card]- How does the Unicode stack run from what a reader sees down to what a byte-level tokenizer eats?
> A **grapheme**; one or more **codepoints**; 1 to 4 **bytes** per codepoint.

> [!card]- Which Unicode normalization forms are lossy?
> The **compatibility** forms NFKC and NFKD; the canonical forms NFC and NFD are reversible.

> [!card]- What does the K in NFKC and NFKD stand for?
> **Compatibility**: these forms also merge stylistic variants such as ligatures.

> [!card]- What does NFKD do that NFD does not?
> Breaks characters into their **most basic parts**, e.g. splitting ligatures, normalizing visual styling away.

> [!card]- Which normalization form should you use to compare rendered text?
> **NFD** (non-lossy, preserves semantic and visual equivalence).

> [!card]- Which normalization form should you use for search indexing or NLP analysis?
> **NFKD**.

> [!card]- Why is NFKD's lossiness useful for search indexing?
> It unifies variants: the ligature **ﬁ matches fi**, ① becomes 1.

> [!card]- Why is NFKD not necessarily suitable for generation targets?
> A model trained on them **never sees, so cannot produce**, the characters NFKD folded away.

> [!card]- What happens to the long s ſ under NFD, and under NFKD?
> NFD **keeps** ſ; NFKD replaces it with plain **s**, irreversibly.

> [!card]- How many bytes per codepoint does UTF-8 use?
> **1 to 4**, variable.

> [!card]- Which Unicode encoding uses exactly 4 bytes for every codepoint?
> **UTF-32**: simple for indexing, memory-hungry, rare in practice.

> [!card]- Which of UTF-8, UTF-16 and UTF-32 is ASCII compatible?
> Only **UTF-8**; the other two use at least two bytes per character.

> [!card]- How many UTF-8 bytes does a common CJK character such as 中 take?
> **3**.

> [!card]- In UTF-8, what does a byte starting with `0` signal?
> A complete **single-byte** character (ASCII).

> [!card]- In UTF-8, what does a byte starting with `11` signal?
> The **first byte** of a multi-byte sequence; the number of leading 1s gives its length.

> [!card]- What is a UTF-8 byte of the form `10xxxxxx`?
> A **continuation byte** inside a multi-byte sequence.

> [!card]- Why does 中 (U+4E2D, 15 significant bits) need the 3-byte UTF-8 form?
> Its bits exceed the **11 payload bits** of the two-byte form; the three-byte form carries sixteen.

> [!card]- Why can a reader find character boundaries from any offset in a UTF-8 stream?
> No byte is ambiguous about its role: skip `10xxxxxx` continuation bytes to reach a character start.

> [!card]- How is byte inflation defined for byte-level tokenizers?
> **UTF-8 bytes** of a string divided by its number of **perceived characters**.

> [!card]- Which has the highest byte inflation: "hello", "café" or "中文"?
> **中文**: 3 bytes per character, against 1 per character for "hello".

> [!card]- How does higher byte inflation affect byte-level tokenization?
> Tokens are split **more aggressively**, so the same meaning costs more tokens.

> [!card]- Which preprocessing steps silently go wrong when crawled text mixes NFC and NFD?
> **Deduplication**, **search**, string equality and prefix matching: all assume identical codepoint sequences.

> [!card]- What is the fix for crawled data that mixes normalization forms?
> **Consistent normalization** in the pipeline, e.g. first decompose, then compose.

> [!card]- What happens when `iconv` converts text using the wrong source encoding?
> It **fails**, which is useful: silent wrong decoding is what produces mojibake.

> [!card]- What goes wrong when breaking a string at the byte level?
> Multi-byte codepoints get split into fragments that are **not characters** at all.

> [!card]- What goes wrong when breaking a string at the codepoint level?
> A base letter can be **separated from its combining diacritic** (e + ◌́).

> [!card]- Why is grapheme-level breaking hard?
> Grapheme boundaries are **very language-specific**.

> [!card]- Where does mixing string-breaking strategies cause misalignment bugs?
> **Span annotation** (SQuAD-style offsets shift) and **counting across corpora**.

> [!card]- Why does lowercasing the letter I go wrong for Turkish?
> Turkish lowercase I is dotless **ı**; dotted İ pairs with i. English rules change the word.

> [!card]- Why can strings not be sorted alphabetically by codepoint?
> Accented letters' codepoints are **not necessarily near** their base letter's.

> [!card]- How do Swedish and German differ in sorting å, ä, ö?
> Swedish puts them **after z** as separate letters; German **interleaves** them (ä as ae).

> [!card]- Why is punctuation normalization (“...” against «...» against 「...」) hard?
> There is **no simple solution**: the mappings between traditions must be **encoded by hand**.

> [!card]- How should a multilingual tokenizer match punctuation?
> By **Unicode general category P** (`\p{P}`), not a hand-written class like `[.!?]`, which misses 。 and ।.

> [!card]- True or false: Unicode defines both each character's codepoint and how it is stored as bytes.
> **False.** Unicode defines codepoints only; bytes are the job of an encoding such as UTF-8.

> [!card]- True or false: How well a language is covered by NLP resources depends mainly on its number of speakers.
> **False.** It depends on commercial interest and military relevance.

> [!card]- True or false: UTF-8 is ASCII compatible.
> **True.** `U+0000` to `U+007F` are encoded as single bytes identical to ASCII.

> [!card]- True or false: UTF-32 is ASCII compatible.
> **False.** It uses exactly 4 bytes for every codepoint, ASCII included.

> [!card]- True or false: UTF-16 encodes every codepoint in exactly 2 bytes.
> **False.** 2 or 4 bytes; codepoints outside plane 0 need 4.

> [!card]- True or false: NFKD normalization is reversible.
> **False.** Compatibility forms are lossy (ſ becomes s); only the canonical forms NFC and NFD are reversible.

> [!card]- True or false: NFKD normalization makes the ligature ﬁ and the pair fi identical.
> **True.** Compatibility decomposition splits ligatures, which is why NFKD suits search indexing.

> [!card]- True or false: Code pages allow several scripts in the same document.
> **False.** A document is read with one table, so one script per document.

> [!card]- True or false: Detecting a text's script is enough to identify its language.
> **False.** Arabic script alone covers Arabic, Persian, Urdu, Pashto and Sorani.

> [!card]- True or false: NFC and NFD versions of "café" render identically but are unequal as strings.
> **True.** Same appearance, different codepoint sequences (4 against 5 codepoints).

> [!card]- True or false: Deduplicating a crawled corpus that mixes NFC and NFD raises an error on the mismatched copies.
> **False.** It fails silently: both copies are kept and the numbers are just wrong.

> [!card]- True or false: Grapheme boundaries are language-independent.
> **False.** They are very language-specific, which makes grapheme-level breaking hard.

> [!card]- True or false: A byte-level tokenizer splits Chinese text into more tokens than English text of the same meaning.
> **True.** CJK characters take 3 UTF-8 bytes against 1 for basic Latin, so splitting is more aggressive.

> [!card]- True or false: Whitespace tokenization works for Thai because Thai uses spaces between words.
> **False.** Thai spaces end clauses or sentences, not words.

> [!card]- Multiple selection: which languages break the assumption that words are separated by whitespace? (A) Thai (B) Russian (C) Khmer (D) German
> **A and C.** Russian breaks the word-order assumption; German breaks one-word-one-meaning through compounding.

> [!card]- Multiple selection: which are true of script and language? (A) Serbian is written in Cyrillic and Latin (B) Arabic script identifies the Arabic language (C) Armenian has a language-specific script
> **A and C.** B is false: Arabic script also writes Persian, Urdu, Pashto and Sorani.

> [!card]- Multiple selection: which are true of Unicode normalization? (A) NFC and NFD are reversible (B) NFKD is the recommended form for generation targets (C) NFKD maps the long s ſ to s (D) NFD merges a base letter and combining mark into one codepoint
> **A and C.** B is false: NFKD is for analysis, not necessarily generation. D is false: NFD decomposes; NFC composes.

> [!card]- Multiple selection: which are true of UTF-8? (A) It uses 1 to 4 bytes per codepoint (B) The number of leading 1s in a lead byte gives the sequence length (C) Emoji fit in 3 bytes (D) From a random offset you cannot tell whether you are at a character start
> **A and B.** C is false: plane-1 emoji need 4 bytes. D is false: continuation bytes start with `10`, so a character start is recognisable.

> [!card]- Multiple selection: which are true of string breaking? (A) Byte-level breaking can produce fragments that are not characters (B) Codepoint-level breaking always keeps a diacritic with its base letter (C) Grapheme boundaries are language-specific (D) Mixing breaking strategies across corpora is harmless
> **A and C.** B is false: e and a combining accent are two codepoints and can be split. D is false: spans misalign and counts become incomparable.

## Links

- **Course:** [[MNLP - Overview|Course overview]]
- **Previous:** [[MNLP-L01 - Overview]]
- **Next:** [[MNLP-L03 - Morphology and Word Formation]]
- **Project:** [[MNLP - Mini Project]]
- **Related concept:** [[Tokenization]]
- **Source:** `multilingual-nlp.pdf`, 33 slides, Christof Monz, Informatics Institute, University of Amsterdam
- **Further reading (recommended in the lecture):** Joel Spolsky (2003), *The Absolute Minimum Every Software Developer Absolutely, Positively Must Know About Unicode and Character Sets*; Nikita Prokopov (2023), *The Absolute Minimum Every Software Developer Must Know About Unicode in 2023* (both blog posts)
