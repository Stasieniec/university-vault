# Archived flashcards: MNLP-L02 - Multilinguality and Writing Systems

The long-form cards this note carried until 2026-10-08, when they were replaced by short single-fact cards. Kept for reference; not published.

Click a question to reveal its answer, or press **Study** to drill the whole set. Cards marked as exam questions are meant to be answered out loud or on paper first, then checked against the points listed.

> [!exam]- State the four English-first assumptions built into standard NLP pipelines. For each, give languages where it fails and what goes wrong concretely.
> 1. **Words are separated by whitespace.** Fails for Chinese, Japanese, Thai, Lao, Khmer, Burmese: there is no word delimiter, so whitespace tokenization returns one token for a whole Chinese sentence. Thai uses spaces to end **clauses or sentences** and has no dedicated full stop, so a space means something different from what the tokenizer assumes.
> 2. **Grammatical relations are encoded by word order.** Fails for languages with rich case marking (Russian, Finnish, Korean): who did what to whom is marked on the noun, word order is free, and position-based features carry much less information.
> 3. **One word = one unit of meaning.** Fails for agglutinative languages (Finnish, Turkish, Hungarian) and compounding languages (German, Dutch, Swedish): *Bundesverfassungsgericht* is one token holding federal + constitutional + court.
> 4. **Text flows left to right.** Fails for Arabic-script and Semitic languages (Arabic, Hebrew, Urdu, Persian): rendering runs right to left, so the first character in the string is not the leftmost on screen.

> [!exam]- Argue that "script is not language" holds in both directions, and explain what each direction breaks in an NLP pipeline.
> - **One script, many languages:** Latin (English, German, French, most European languages, several African, Turkic and Asian languages), Arabic (Arabic, Persian, Urdu, Pashto, Sorani), Cyrillic (Russian, Belarusian, Serbian), Devanagari (Hindi, Marathi, Nepali), Ethiopic (Amharic, Tigrinya). So **detecting the script does not identify the language**: Latin narrows nothing, and Arabic script leaves several languages that are not even all in one family.
> - **One language, many scripts:** Serbian (Cyrillic and Latin), Kurdish (Latin and Arabic), Uzbek (Latin and Cyrillic), Punjabi (Gurmukhi and Arabic). So **detecting the language does not tell you the script**: a Serbian corpus can mix both, even in one document, and a model trained on one script and tested on the other sees no shared characters for the same language.
> - Some scripts are language-specific (Armenian, Georgian, Greek, Sinhala, Khmer), which is the only case where the two coincide.
> - Also: one sentence can mix several writing systems (Japanese uses Katakana, Kanji and Hiragana together), so a single "script of this text" label can fail.

> [!exam]- Why is the English-centric bias of NLP structural, and why do claims of "language independence" deserve suspicion?
> - Almost every resource exists first, often only, in English: annotated data, unannotated data, toolkit coverage (dictionaries, analyzers), benchmarks. Most publications also focus on English.
> - Non-English resources arise mostly as **by-products** (translations, product reviews) or as **translations of English annotated resources** (so the annotation scheme stays English-shaped), and rarely as dedicated efforts.
> - Coverage tracks **commercial interest and military relevance**, and speaker count is a poor predictor: over 7,000 languages, more than 100 with over 10 million speakers, yet Dutch is only medium resource.
> - Standard pipelines encode properties of English (whitespace tokenization, fixed word order, closed vocabularies) as if they were properties of language.
> - "Language independent" usually means tested on English plus a very small set of languages: a method gets the label because nobody checked.
> - The bias is understandable, since resource creation costs time and money, which is exactly why it does not fix itself.

> [!exam]- Compare alphabetic, syllabic and logographic writing systems and explain what each, plus the abjad and Hangul cases, implies for NLP.
> - **Alphabetic:** each symbol is a consonant or vowel sound, small inventory (Latin, Cyrillic, Hangul).
> - **Syllabic:** each symbol is a whole syllable (Hiragana, Katakana).
> - **Logographic:** each symbol is a word or morpheme directly, little sound information (Chinese Hanzi, Japanese Kanji).
> - **Inventory size** sets the cost of a character: tens of symbols for an alphabet, thousands for a logographic script, which a character-level vocabulary must budget for.
> - **Abjad** (Arabic): consonantal skeleton, vowels omitted. كتب can be *kataba*, *kutiba* or *kutub*, so the orthography is lossier than the language and the model must resolve the ambiguity from context. No tokenizer can recover information that was never written.
> - **Hangul** breaks the neat taxonomy: alphabetic by symbol-to-sound mapping (each jamo is a consonant or vowel), syllabic by visual arrangement (jamo grouped into square blocks), and letter shapes depict tongue and lip positions.

> [!exam]- Explain the difference between a grapheme, a codepoint and a byte, and show how confusing them corrupts tokenization, deduplication and span annotation.
> - **Grapheme:** what the reader sees as one character. **Codepoint:** Unicode's abstract number for a character, one or more per grapheme. **Byte:** UTF-8 stores each codepoint in 1 to 4 bytes. They coincide only in ASCII.
> - Example: NFD é is 1 grapheme, 2 codepoints (U+0065 U+0301), 3 bytes (65 CC 81).
> - **Tokenization:** byte-level tokenizers see the byte stream. Byte-level splits can produce non-existent characters; codepoint splits can strand a diacritic from its base letter.
> - **Deduplication, search, equality, prefix matching** assume identical codepoint sequences, so NFC and NFD copies of the same text silently fail to match. No error is raised, the numbers are just wrong.
> - **Span annotation** (SQuAD-style QA): offsets recorded under one breaking convention and read under another give shifted or truncated answers. Counts across corpora with different conventions are not comparable.
> - Fix: normalize consistently in the pipeline, and pick one breaking convention and enforce it everywhere. Consistency matters more than which level you pick.

> [!exam]- Crawled multilingual data mixes normalization forms. Which Unicode normalization form would you use for search indexing, for comparing rendered text, and for generation targets, and why?
> - The four forms form a grid: canonical (reversible) NFC and NFD; compatibility (lossy) NFKC and NFKD; composed (C) versus decomposed (D).
> - **Comparing rendered text:** NFD, which splits into base characters plus combining marks and preserves semantic and visual equivalence. It is non-lossy, so you can decompose and recompose.
> - **Search indexing or NLP analysis:** NFKD. Its lossiness is a feature: the ligature ﬁ and the pair fi become the same, ① becomes 1.
> - **Generation:** avoid compatibility forms. If training targets are NFKD-normalized, the model can never learn to produce the character the user actually typed (for example the long s ſ becomes plain s).
> - Whatever you choose, apply it consistently in the pipeline (for example decompose then compose), otherwise dedup, search and equality checks are silently inaccurate.
>
> Losing marks: getting the analysis versus generation rule backwards, or claiming NFKD is reversible.

> [!exam]- Explain byte inflation for byte-level tokenizers, compute it for "hello", "café" and "中文", and argue why it is a fairness problem.
> - $\text{inflation}(s) = \lvert\text{UTF-8 bytes}(s)\rvert \,/\, \lvert\text{characters}(s)\rvert$, characters meaning what a reader perceives.
> - "hello": 5 bytes / 5 = **1.0x**. "café" (NFC, é is 2 bytes): 5 / 4 = **1.25x**. "中文" (3 bytes each): 6 / 2 = **3.0x**.
> - More bytes per character means more aggressive splitting, so the same meaning costs more tokens; byte splits can also produce fragments that are not characters at all.
> - Fairness: a Latin-script user pays about 1 byte per character, a CJK user 3. The same sentence costs CJK users more tokens: shorter effective context window, more compute for identical content, and less linguistically meaningful units. The English-first assumption reaches the byte layer.

> [!exam]- Encode U+4E2D (中) in UTF-8 by hand, stating which byte rules you use and why UTF-8's design is self-synchronising.
> - 0x4E2D = `0100111000101101`: 15 significant bits, more than the 11 payload bits of the 2-byte form, so use the 3-byte form `1110yyyy 10yyyyxx 10xxxxxx` with $4 + 6 + 6 = 16$ payload bits.
> - Split 4 + 6 + 6: `0100 | 111000 | 101101`.
> - Bytes: `11100100 10111000 10101101` (hex E4 B8 AD).
> - Rules: a byte starting `0` is a single byte (ASCII); a byte starting `11` begins a multi-byte sequence, with the number of leading 1s giving its length; a byte starting `10` is a continuation byte.
> - Self-synchronising: no byte value is ambiguous about its role, so from any offset you can tell whether you are at a character start and walk back by skipping `10xxxxxx` bytes. UTF-8 therefore survives truncation, concatenation and byte-level search.

> [!card]- Which four kinds of NLP resource exist first or only in English, and by which three routes do non-English resources come into being?
> - Resources: **annotated/labeled training data**, **unannotated training data**, **coverage of existing toolkits** (dictionaries, analyzers), **benchmarks/competitions**.
> - Routes: (1) **naturally occurring by-products** such as translations and product reviews, never made for NLP; (2) **translations of existing annotated English resources**, so the annotation scheme is English-shaped even when the text is not; (3) **less so dedicated efforts**, the route that would actually fix the problem.

> [!card]- How many languages are there, how many are large, and what actually determines how well a language is covered by NLP resources?
> Over **7,000** spoken languages; **more than 100** have over 10 million speakers. Coverage depends on **commercial interest** and **military relevance**; speaker count predicts it badly. Dutch, spoken by roughly 24 million people in a rich country, is only medium resource.

> [!card]- Give Monz's three resource tiers: what exists in each and which languages belong where, including the gap inside the top tier.
> - **High resource:** large labeled and unlabeled data, many tools. English and Chinese (Mandarin), then an explicit **big gap**, then French, Spanish, Arabic, Russian.
> - **Medium resource:** limited labeled and reasonable unlabeled data, some tools. German, Italian, Dutch, Korean, Japanese.
> - **Low resource:** no labeled data, limited unlabeled data, no tools. All the remaining roughly 7,000 languages.
> - The gap means "high resource" itself covers two very different situations.

> [!card]- Using 我不知道她去哪儿了。 ("I don't know where she went"), name three ways Chinese defeats English-style processing assumptions, and say what is shared with English.
> - **No spaces**: whitespace tokenization yields one token, against 6 for the English sentence.
> - **No tense marking on the verb** 去 (go): English changes go to went, Chinese leaves the verb alone.
> - **Past tense comes from the sentence-final particle** 了, at the end of the clause, arbitrarily far from the verb, and a tokenizer treats it as an unrelated token. Models assuming morphological features live on the word they modify break.
> - **Characters are not words**: 9 characters give 7 words plus punctuation (知道 "know" and 哪儿 "where" are two characters each).
> - Shared: both the matrix clause (I NEG know) and the embedded clause (she go where) are **SVO**, so word order is not the difference.

> [!card]- Define morphology, morpheme, root and affix, with an example of each.
> - **Morphology:** the study of the internal structure of words and how they are formed from smaller units of meaning.
> - **Morpheme:** the smallest unit of meaning or grammatical function. *dogs* = *dog* + *-s* (plural).
> - **Root/base:** the core part carrying the primary meaning: *friend* in *unfriendly*.
> - **Affix:** prefixes (front) and suffixes (back) that modify a root's meaning or grammatical role: *un-* and *-ly* in *unfriendly*.

> [!card]- How many surface forms does English build from "happy", and why did that make closed word-level vocabularies seem reasonable?
> Eight: happy, happier, happiest, unhappy, unhappier, unhappiest, happiness, unhappiness. With so little inflection, a vocabulary of a few tens of thousands of word types covers English acceptably, so a closed vocabulary looked like a sensible design decision.

> [!card]- The Russian noun *kniga* ("book"): how many case and number cells, how many distinct forms, which forms are syncretic, and what does syncretism mean for NLP?
> - 6 cases (nom, gen, dat, acc, inst, prep) x 2 numbers = **12 cells**, but only **9 distinct forms**.
> - *knígye* covers dat.sg and prep.sg; *knígi* covers gen.sg, nom.pl and acc.pl.
> - For NLP it cuts both ways: fewer types to learn, but the surface form is ambiguous between grammatical functions, so case cannot be read off the string.

> [!card]- What does "agglutinative" mean? Break Finnish *onnellisimmillanikin* ("even at my happiest") into its morphemes.
> Agglutinative: morphemes stack, each contributing one piece of meaning, and the word can grow without bound.
> 1. *onnellis*: stem, "happy"
> 2. *-imm-*: superlative marker
> 3. *-i-*: plural oblique-case marker
> 4. *-lla*: adessive case ("at/on/in")
> 5. *-ni*: 1st person singular possessive ("my")
> 6. *-kin*: "even"
>
> Six morphemes, one orthographic word; English needs four words, one of them a preposition.

> [!card]- How are Finnish *onnellinen* ("happy") and *onnellisempi* ("happier") formed?
> - *onnellinen* = *onni* ("happiness") + *-llinen* (adjective-forming suffix, roughly "having the quality of"), literally "having happiness".
> - *onnellisempi*: *onnellinen* has an oblique stem *onnellis* (adjectives in *-nen* often swap this for *-s* before suffixes), plus *-empi* (*-mpi* is the comparative suffix, with a linking vowel).

> [!card]- Why can more training data not fix a fixed word-level vocabulary for Finnish, and what does this motivate?
> Agglutination produces combinatorially many word forms like *onnellisimmillanikin*, most of which a fixed vocabulary will never have seen however much data you collect. This motivates **subword modelling**.

> [!card]- What is the directionality trap for right-to-left scripts, and which operations does it break?
> Codepoints are stored in **logical order** (Unicode is "written left to right"), and the visual right-to-left order is produced by the **bidirectional algorithm** at render time. So in Arabic and Hebrew "the first character of the string" and "the leftmost character on screen" differ. Code assuming they are the same breaks **character offsets, span annotation and truncation**, invisibly to a developer who cannot read the script.

> [!card]- Order the linguistic units from small to large and give what each level studies.
> 1. **Characters:** size of the character set, writing systems.
> 2. **Words:** word formation (morphology), lexical meaning/semantics.
> 3. **Clauses/sentences:** grammar or syntax, sentence semantics.
> 4. **Paragraphs:** discourse structure, anaphora, ellipsis.
>
> Every higher level inherits whatever goes wrong at the character and word levels.

> [!card]- From "Amsterdam is the capital of the Netherlands" in eleven scripts, what four observations matter for NLP?
> - **Japanese mixes three writing systems** in one sentence: アムステルダム and オランダ are Katakana, 首都 is Kanji, は, の and です are Hiragana.
> - **Thai has no spaces anywhere** and no full stop at the end.
> - **Sentence terminators differ**: `:` (Armenian as reproduced; the proper Armenian full stop is ։, U+0589), ። (Amharic), 。 (Japanese), । (Bengali), nothing (Thai), `.` in the Russian, Georgian, Pashto, Inuktitut, Somali and Vietnamese lines. A `[.!?]` regex finds the end of only 6 of 11.
> - **Same script, different languages**: Somali and Vietnamese are both Latin, and Vietnamese adds tone marks and a stroked đ.

> [!card]- Define the three basic types of writing system and give examples of each.
> - **Alphabetic:** each symbol represents a consonant or vowel sound; typically a small inventory. Latin, Cyrillic, Hangul.
> - **Syllabic:** symbols represent whole syllables (vowel and consonant combinations). Japanese Hiragana and Katakana.
> - **Logographic:** each symbol represents words or morphemes directly; typically little sound information. Chinese Hanzi, Japanese Kanji.

> [!card]- What does each script encode in water (Latin), पानी (Devanagari), 水 (Han), كتب (Arabic) and 한글 (Korean)?
> - **Latin** *water*: alphabetic, sounds spelled out with letters.
> - **Devanagari** पानी *pani*: alphasyllabic, consonant plus an inherent vowel.
> - **Han** 水 *shui*: logographic, one symbol is one morpheme.
> - **Arabic** كتب *ktb*: root only, vowels omitted.
> - **Korean** 한글 (Hangul): letter shapes encode articulatory features.

> [!card]- What is an abugida? Explain with how पानी (*pani*) is built.
> The typological name for the Devanagari pattern: each consonant letter carries an **inherent vowel** that other marks override. पानी is *pā* + *nī*, two akshara: प (*pa*) with its inherent *a* overridden by the vowel sign ा (*ā*), and न (*na*) overridden by ी (*ī*). It is not spelled *p* + *ā* + *n* + *ī* as separate letters.

> [!card]- What is an abjad? Explain with the Arabic root k-t-b and the readings of كتب.
> The typological name for the Arabic pattern: the consonantal skeleton is written and (short) vowels are left out. The root k-t-b underlies *kataba* ("he wrote"), *kitāb* ("book") and *maktab* ("office"). The written form كتب itself can be read *kataba* ("he wrote"), *kutiba* ("it was written") or *kutub* ("books"). *kitāb* and *maktab* are spelled differently (كتاب, مكتب) because long vowels and the *m-* prefix are written.

> [!card]- Which languages are written with more than one script, and which scripts are language-specific?
> - **Multiple scripts:** Serbian (Cyrillic and Latin), Kurdish (Latin and Arabic), Uzbek (Latin and Cyrillic), Punjabi (Gurmukhi and Arabic).
> - **Language-specific scripts:** Armenian, Georgian, Greek, Sinhala, Khmer.

> [!card]- What did ASCII provide, how did code pages extend it, and what were their two fatal disadvantages?
> - **ASCII (1963):** 128 codes, enough for English letters, digits and punctuation.
> - **Code pages:** the same 128 (or 256) codes in a byte, **interpreted differently** per language or script: Latin-1 (Western Europe), Shift-JIS (Japanese), GBK (Chinese), KOI8-R (Russian).
> - Disadvantages: (1) text decoded with the wrong table becomes **gibberish (mojibake)**, and since the bytes are legal in both tables no error is raised; (2) **no multiple scripts in one document**.

> [!card]- What happens when the UTF-8 bytes of Привет are decoded as Windows-1252, and why is this kind of error dangerous for corpora?
> The bytes `D0 9F D1 80 D0 B8 D0 B2 D0 B5 D1 82` decode as UTF-8 to Привет but as Windows-1252 to ÐŸÑ€Ð¸Ð²ÐµÑ‚. Same bytes, **no error raised**, so the garbage silently enters training data. (Strict Latin-1 would map `9F`, `80`, `82` to invisible C1 control characters; the visible garbage is the Windows-1252 superset's output, which is what most real mojibake looks like.)

> [!card]- Define Unicode and a codepoint, give two example codepoints, and state the two properties that carry most practical weight.
> - **Unicode:** a single universal character set, maintained by the Unicode Consortium, in which every character in every supported script gets exactly one unique number.
> - **Codepoint:** that number, written `U+number`: `U+0041` is Latin "A", `U+4E2D` is Han 中.
> - (1) **Unicode defines meaning only**: how a codepoint is stored as bytes is a separate encoding question (UTF-8, UTF-16, UTF-32). (2) **Unicode is a superset of ASCII**: `U+0000` to `U+007F` are identical to ASCII.

> [!card]- Give the scale of Unicode: code space, assigned codepoints, scripts, planes, and codepoints per plane, and show the arithmetic that links them.
> - Over **1.1 million** possible codepoints, **150,000+** assigned, **168 scripts**, **17 planes** spanning `U+0000` to `U+10FFFF`.
> - Each plane holds $2^{16} = 65{,}536$ codepoints, and $17 \times 65{,}536 = 1{,}114{,}112$.
> - Watch the typo 65,356 that circulates in course material: the correct figure is 65,536.

> [!card]- Name the Unicode planes with their ranges and contents, and say why plane 0 matters most.
> - **0, BMP** (`0000`–`FFFF`): most modern languages, the most common CJK Unified Ideographs, symbols such as currency, a Private Use Area.
> - **1, SMP** (`10000`–`1FFFF`): non-CJK historic ideographic scripts, modern scripts, symbols such as musical notation, emoji.
> - **2, SIP** (`20000`–`2FFFF`): additional CJK ideographs, mostly historical, uncommon or variants.
> - **3, TIP** (`30000`–`3FFFF`): historical CJK ideographs not in the BMP or SIP.
> - **4 to 13** (`40000`–`DFFFF`): unassigned.
> - **14, SSP** (`E0000`–`EFFFF`): tags and variation sequence selectors.
> - **15 to 16, SPUA-A/B** (`F0000`–`10FFFF`): Private Use Area.
>
> Almost everything normally handled is in plane 0; anything outside needs more than 16 bits, which makes UTF-16 awkward and emoji four bytes in UTF-8.

> [!card]- What does a spread of codepoints such as a (U+0061), ã (U+00E3), & (U+0026), ⻩ (U+2EE9), ❁ (U+2741) and the dog face (U+1F436) show about the word "character" in Unicode?
> That one namespace and numbering scheme covers ASCII letters, a precomposed accented letter (Latin-1 Supplement), ASCII punctuation, a CJK radical (the "yellow" radical, visually a form of 黄, yet a separate codepoint), a Dingbats ornament, and an emoji in plane 1.

> [!card]- Distinguish codepoint, character and glyph, and show with "café" why a naive string comparison fails.
> - **Codepoint:** abstract number such as `U+00E9`, Unicode's unit of identity. **Character:** what a reader perceives as one unit, possibly several codepoints. **Glyph:** a stylistic rendering of a character.
> - NFC café: `U+0063 U+0061 U+0066 U+00E9` (4 codepoints, precomposed é). NFD café: `U+0063 U+0061 U+0066 U+0065 U+0301` (5 codepoints, e + COMBINING ACUTE ACCENT).
> - They render identically but are not equal, differ in length and hash differently.

> [!card]- What are the NFC, NFD, NFKC and NFKD forms of ẛ̣ (long s with dot above plus combining dot below), and what does the comparison show?
> - **NFC:** ẛ + ◌̣ = `U+1E9B U+0323`
> - **NFD:** ſ + ◌̣ + ◌̇ = `U+017F U+0323 U+0307`
> - **NFKC:** ṩ = `U+1E69`
> - **NFKD:** s + ◌̣ + ◌̇ = `U+0073 U+0323 U+0307`
>
> Canonical forms keep the long s ſ, since ſ and s are different characters under canonical equivalence. Compatibility forms replace it with plain s: information destroyed deliberately. (Simpler case: ã is `U+00E3` in NFC and `U+0061 U+0303` in NFD.)

> [!card]- Contrast NFD and NFKD: what each does, which is lossy, when to use each, and what the K stands for.
> - **NFD (canonical decomposition):** splits into base characters and combining marks, preserving semantic and visual equivalence. **Non-lossy**: decompose then compose loses nothing. Use to **compare rendered text**.
> - **NFKD (compatibility decomposition):** breaks characters into their most basic parts, including splitting ligatures. **Lossy**: visual styling is normalized to a base form that composition cannot reconstruct. Use for **search indexing or NLP analysis**, not necessarily generation.
> - **K** = compatibility, from German *Kompatibilität*, because C was already taken by composition.

> [!card]- Compare UTF-8, UTF-16 and UTF-32: bytes per codepoint, ASCII compatibility, and where each is used.
> - **UTF-8:** 1 to 4 bytes, ASCII compatible. The web, Linux, most file formats, virtually all modern NLP tooling; what people usually mean by "unicode".
> - **UTF-16:** 2 or 4 bytes, not ASCII compatible. Java, JavaScript strings, Windows internals, .NET.
> - **UTF-32:** exactly 4 bytes, not ASCII compatible. Rare; simple for indexing but memory-hungry.

> [!card]- Give the four UTF-8 byte patterns with their codepoint ranges and payload sizes.
> - `U+0000`–`U+007F`: `0xxxxxxx` (7 bits)
> - `U+0080`–`U+07FF`: `110yyyxx 10xxxxxx` ($5 + 6 = 11$ bits)
> - `U+0800`–`U+FFFF`: `1110yyyy 10yyyyxx 10xxxxxx` ($4 + 6 + 6 = 16$ bits)
> - `U+10000`–`U+10FFFF`: `11110zzz 10zzyyyy 10yyyyxx 10xxxxxx` ($3 + 6 + 6 + 6 = 21$ bits)
>
> Examples: A (`U+0041`) is 1 byte, é (`U+00E9`) 2, 中 (`U+4E2D`) 3, grinning face (`U+1F600`) 4.

> [!card]- Encode é (U+00E9) in UTF-8 step by step.
> 1. 0xE9 = 233 = `11101001`: 8 significant bits, more than the 7 of the 1-byte form, so use the 2-byte form (11 payload bits).
> 2. Pad to 11 bits and split 5 + 6: `00011 | 101001`.
> 3. Byte 1 = `110` + `00011` = `11000011`; byte 2 = `10` + `101001` = `10101001` (hex C3 A9).

> [!card]- Encode the grinning face emoji (U+1F600) in UTF-8 step by step.
> 1. 0x1F600 needs 17 bits, beyond the 16 of the 3-byte form, so use the 4-byte form (21 payload bits).
> 2. Pad to 21 bits: `000011111011000000000`; split 3 + 6 + 6 + 6: `000 | 011111 | 011000 | 000000`.
> 3. Bytes: `11110000 10011111 10011000 10000000` (hex F0 9F 98 80).

> [!card]- Worked example: encode COMBINING ACUTE ACCENT (U+0301) in UTF-8, and check it against the bytes of NFD "café".
> 1. 0x301 = `1100000001`: 10 significant bits, more than 7, at most 11, so the 2-byte form.
> 2. Pad to 11 bits, split 5 + 6: `01100 | 000001`.
> 3. Byte 1 = `110` + `01100` = `11001100` (CC); byte 2 = `10` + `000001` = `10000001` (81).
>
> So NFD café is `63 61 66 65 CC 81`: 6 bytes, where the last two together are the accent.

> [!card]- Worked example: decode the UTF-8 byte pair D0 9F (the first two bytes of Привет) to a codepoint.
> 1. `D0` = `11010000`: starts with `110`, so it opens a 2-byte sequence, payload `10000`.
> 2. `9F` = `10011111`: starts with `10`, a continuation byte, payload `011111`.
> 3. Concatenate: `10000011111` = 0x41F, so the codepoint is **U+041F**, the first letter П.

> [!card]- State the byte inflation formula, define its terms, and give the three standard values.
> $$\text{inflation}(s) = \frac{\lvert \text{UTF-8 bytes}(s) \rvert}{\lvert \text{characters}(s) \rvert}$$
> Numerator: length of $s$ encoded as UTF-8. Denominator: number of characters a reader perceives.
> - "hello": 5 / 5 = **1.0x**
> - "café": 5 / 4 = **1.25x**
> - "中文": 6 / 2 = **3.0x**

> [!card]- Worked example: compute the byte inflation of Привет, and of "café" in NFD, and say what the second result shows.
> - **Привет:** its UTF-8 form `D0 9F D1 80 D0 B8 D0 B2 D0 B5 D1 82` is 12 bytes for 6 characters, so 12 / 6 = **2.0x** (each Cyrillic letter costs 2 bytes).
> - **NFD café:** `63 61 66 65 CC 81` is 6 bytes for 4 perceived characters, so 6 / 4 = **1.5x**, against 1.25x in NFC.
> - The normalization form alone changes what a byte-level tokenizer sees for the same visible text.

> [!card]- What two consequences does byte-level representation have for byte-level tokenizers?
> - **Byte inflation:** the more bytes a character needs, the more aggressively tokens tend to be split, so a 3-byte script is chopped into more pieces than a 1-byte one for the same meaning.
> - **Non-existent characters:** cutting a multi-byte sequence (for example a 3-byte sequence after byte 2) leaves pieces that are not valid characters, so the model learns over fragments no reader would recognise.

> [!card]- Which preprocessing steps break on unnormalized crawled text, how do they fail, and what is the fix (including the Python and command-line tools)?
> - Crawled Unicode can mix forms (some pages NFC, some NFD) with nothing announcing which.
> - Steps assuming identical codepoint sequences: **deduplication**, **search**, **string equality and prefix matching**. They fail silently: dedup keeps two copies differing only in form, and the reported corpus size is wrong.
> - Fix: **always normalize consistently in the pipeline**, for example decompose then compose.
> - Python `unicodedata` can **normalize** (`unicodedata.normalize("NFC", s)`) and **test** a form (`unicodedata.is_normalized("NFC", s)`).
> - For code-page text (GB2312, KOI8-R): `iconv -f GB2312 -t UTF-8`, which **fails if the from-encoding is wrong**. That failure is useful, since silent wrong decoding is what creates mojibake.

> [!card]- Compare breaking a string at byte, codepoint and grapheme level, using "café" (NFD).
> - **Bytes:** `63 | 61 | 66 | 65 | CC | 81`, 6 units, the last two are not characters. Byte breaking can split codepoints.
> - **Codepoints:** c | a | f | e | ◌́, 5 units, the accent is stranded from its base letter.
> - **Graphemes:** c | a | f | é, 4 units, what the reader sees. Ideal, but grapheme boundaries are very language-specific, so it is the expensive option.

> [!card]- Where do mixed breaking strategies cause misalignments, and what is the rule to follow?
> - **Span annotation systems** (SQuAD-style QA): spans stored as character offsets under one convention and read under another give shifted or truncated answers.
> - **Counting across corpora** with different strategies: length statistics, token counts and coverage numbers become incomparable, with no warning.
> - Rule: choose one convention and enforce it across every corpus, tool and metric. Any string breaking or length computation must be consistent across documents.

> [!card]- What is the Turkish-I problem?
> Lowercasing is not language-independent. Turkish has two letter pairs, dotted İ / i and dotless I / ı, so lowercase(I) = ı in Turkish but i in English. Running `.lower()` on Turkish text with the wrong locale changes which word you are looking at.

> [!card]- Why can sorting not be done by codepoint, and how do Swedish and German order å, ä, ö differently?
> - Codepoints of accented letters are not necessarily near their base letter's, so codepoint order puts å nowhere near a.
> - **Swedish:** a ... z, then å, ä, ö as separate letters **after z**.
> - **German:** ä sorts as ae, ö as oe, ü as ue, **interleaved** with the plain vowels.
> - So sorting is a locale-dependent operation.

> [!card]- How should punctuation be tokenized? Give the steps for `He'd say "I promise you!", but then disappear.` and the general-category approach.
> - Agreed principle: punctuation adjacent to a word is not part of the word.
> - Step 1, separate every mark: He ' d say " I promise you ! " , but then disappear .
> - Step 2, optionally expand the contraction: He would say ..., a further normalization decision and a lossy one.
> - Match the **Unicode general category P** with `\p{P}`, as in `regex.sub(r'(\p{P})', r' \1 ', string)`. This needs the third-party `regex` package: Python's built-in `re` does not support `\p{...}`.
> - A hand-written class like `[.,!?]` is the English-first assumption again: it misses 。, ،, । and ።.

> [!card]- What is punctuation normalization, why is it hard, and where does it arise?
> The same function is served by different characters across traditions, for example quotation marks: “...” (English), «...» (French, Russian), 「...」 (Japanese). There is **no simple solution**: the mappings must be **manually encoded**. It arises in quotes, end-of-sentence markers, question marks and single quotes.

> [!card]- Summarise the Unicode stack in one sentence and list the main recap points.
> A reader sees a **grapheme**, represented by **one or more codepoints**, each encoded in **1 to 4 bytes**.
> - Unicode can encode over a million characters, so writing-system-specific code pages are unnecessary.
> - Normalization (NFC, NFD, NFKC, NFKD) can unify different encodings of the same text.
> - UTF-8 is the most common form of Unicode in documents and NLP tooling.
> - Any string breaking or length computation, byte- or codepoint-level, must be consistent across documents.
