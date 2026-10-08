# Archived flashcards: MNLP-L01 - Overview

The long-form cards this note carried until 2026-10-08, when they were replaced by short single-fact cards. Kept for reference; not published.

Click a question to reveal its answer, or press **Study** to drill the whole set. Cards marked as exam questions are meant to be answered out loud or on paper first, then checked against the points listed.

> [!exam]- Distinguish multilingual from crosslingual NLP. Give examples of each and explain why the English-centric nature of LLMs keeps both open problems.
> - **Multilingual:** building systems that work for **multiple languages**. Examples: NER for multiple languages (can it be trained without being language-specific?), language-independent parsing (are there universal features?), cross-language classification (can it be trained without language-specific data?).
> - **Crosslingual:** **transferring information or knowledge across languages**. Examples: machine translation, crosslingual QA, crosslingual unlearning, crosslingual reasoning.
> - **LLMs:** trained predominantly on Internet resources, so they are **English-centric**. Multilingual models such as Llama 3.1 and DeepSeek do better on **high-resource languages**; dedicated multilingual models such as Aya-Expanse tend to **lag behind**. LLMs are still fragile on non-English text.
>
> Losing marks: treating the two terms as synonyms. One is about covering many languages, the other about moving information between them.

> [!exam]- Contrast pre-deep-learning NLP with deep-learning NLP. What did neural networks gain, what did they cost, and why do insights now transfer between fields?
> - **Before:** a different methodology per application. Classical ML (SVMs, decision trees, generative Bayesian models, discriminative max-ent models) **weighed hand-engineered features**: POS tags, morphology, parse trees, named entities, taxonomies, argument roles.
> - **Advantages of deep learning:** state-of-the-art performance across NLP tasks; very little or no feature engineering; a limited repertoire of network types covers most or all tasks.
> - **Disadvantages:** needs large amounts of training data; errors are hard to trace (opacity); can fail spectacularly.
> - **Transfer:** the same success appears in computer vision, handwriting recognition, speech, robotics and IR. Because network types are uniform and application-specific features are few, insights carry over between these areas.

> [!exam]- What roles do large language models now play in NLP, and what are their limits? Use the Arabic to English translation comparison as evidence.
> - LLMs originated within NLP and now dominate research.
> - **Uses:** end-to-end NLP (QA, summarization, MT); substituting or supplementing humans in data annotation, evaluation (**LLM-as-judge**) and dialog.
> - **Limit:** still fragile on non-English text, because training data is English-centric.
> - **Evidence (Monz's Arabic to English example):** GPT-5.6 Terra, Gemini 3.6 Flash and Claude Sonnet 4.6 all render the sentence as the embassy in South Sudan calling for an investigation into an attack that killed Ethiopian peacekeepers, differing only in wording. **Mistral Small 3.2** produced "The Ethiopian embassy in Khartoum, where the Ethiopian peacekeepers were detained, is closed": it **catastrophically hallucinated** the embassy's location, and its sentence no longer matches the other three.

> [!card]- Name the four ways information can be conveyed or stored, with an example of each, and say where written and spoken language fall.
> - **Structured:** tables, databases.
> - **Unstructured:** language, images and videos.
> - **Continuous signals:** spoken language and audio, images and video.
> - **Discrete signals:** written language.
>
> So language is unstructured; **spoken** language is a continuous signal and **written** language a discrete one.

> [!card]- Why is human language the medium of choice for complex information, and what is the catch for NLP?
> A **limited repertoire of words** allows **infinite expressivity**. The catch: after decades of research, formally modelling language has proven surprisingly hard.

> [!card]- Define NLP: which three fields does it sit between, and which four aspects of language does it try to model?
> NLP sits at the intersection of **computer science, artificial intelligence and linguistics**. Its goal is to model aspects of human language **algorithmically and formally**:
> 1. Word formation (**morphology**)
> 2. Sentence structure (**syntax/grammar**)
> 3. Sentence meaning (**semantics**)
> 4. Document or **discourse** structure

> [!card]- List six core NLP tasks and what each one does.
> - **Text categorization:** assign documents to categories.
> - **Document summarization:** extract or generate condensed versions.
> - **Machine translation:** translate between languages.
> - **Question answering:** return actual answers rather than ranked documents.
> - **Named entity recognition:** identify persons, organizations, dates, locations.
> - **Sentiment analysis:** estimate the attitude in reviews, either positive/negative or fine-grained.

> [!card]- How does question answering differ from information retrieval? Give an example.
> **IR** returns a **ranked list of documents**; **QA** returns the **actual answer**. For "When was the Cuba Crisis?", QA returns "1962" instead of documents that might contain it.

> [!card]- What does named entity recognition output? Show how an entity is tagged in a sentence.
> NER identifies spans that name **persons, organizations, dates and locations** and labels each with its type. Example: "President [Biden]PER has received ...", where the span "Biden" is tagged as a person (PER).

> [!card]- Describe the pre-deep-learning approach to NLP: how methods related to applications, which ML methods and features were used, and what the ML actually did.
> - **Different application, different methodology.**
> - **Methods:** SVMs, decision trees, generative Bayesian models, discriminative max-ent models.
> - **Features:** POS tags, morphology, parse trees, named entities, taxonomies, argument roles.
> - **Role of ML:** to weigh the importance of individual (hand-designed) features for prediction.

> [!card]- Name the three multilingual scenarios and the question each one raises.
> 1. **NER for multiple languages:** can it be trained without being language-specific?
> 2. **Language-independent parsing:** can universal features be identified?
> 3. **Cross-language classification:** can it be trained without language-specific data?

> [!card]- Name the four crosslingual scenarios and what each one does with information.
> 1. **Machine translation:** makes information understandable across languages.
> 2. **Crosslingual QA:** extracts information from resources in other languages.
> 3. **Crosslingual unlearning:** manipulates information across languages.
> 4. **Crosslingual reasoning:** combines information across languages.

> [!card]- Why are LLMs English-centric, and how do general multilingual LLMs compare with dedicated multilingual models?
> They are trained predominantly on **Internet resources**, which are English-dominated. General multilingual models (**Llama 3.1, DeepSeek**) perform better on **high-resource languages**; dedicated multilingual models (**Aya-Expanse**) tend to **lag behind**.
