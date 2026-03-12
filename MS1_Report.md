## Section: Design Choices & Reasoning



### 1. Orthographic Normalization vs. Morphological Segmentation

A primary design decision was to implement **Orthographic Normalization** rather than full **Morphological Segmentation**. While segmenting clitics (e.g., separating "و" or "ال" from the stem) can reduce vocabulary sparsity for the neural models in Milestone 2, we chose to keep them attached for the following reasons:



* **Preservation of Extractive Spans:** The project requirements specify that answers must be extractive spans taken directly from the transcript. Separating clitics introduces artificial spacing that breaks the parallel structure and full traceability between the raw transcript and the supervised QA data.

* **Context for RAG:** In Milestone 3, the Retrieval-Augmented Generation system relies on utilizing the cleaned dataset. Maintaining the original word structures ensures that the semantic meaning of the conversational Egyptian dialect is preserved without losing the nuance of the speaker's original phrasing.

* **Tokenization Trade-offs:** During our EDA, we discovered that standard whitespace tokenization combined with a frequency threshold caused a severe reduction in vocabulary, mapping critical text to the `<UNK>` token due to Arabic's rich morphology. To prevent this information loss, we lowered the frequency threshold to **1**. While this inflates the total vocabulary, it ensures no semantic data is destroyed before MS2.



---



### 2. Unified Normalization & Fuzzy Entity Matching

To resolve linguistic irregularities, we applied a strict global normalization function to both the `.txt` transcripts and the `.csv` QA files.



* **Dialectal Standardization:** Our EDA profiling confirmed a heavy bias toward the Egyptian dialect (an **EG/MSA ratio of 1.72**). We standardized *Ta Marbuta* (mapping "ة" to "ه") and *Hamza* variants (أ, إ, آ to ا) to bridge the gap between Modern Standard Arabic (MSA) and the conversational Egyptian dialect.

* **Programmatic Entity Unification:** The dataset exhibits variation in named entity spelling, especially transliterated foreign names. Instead of hardcoding replacements, we implemented a fuzzy string-matching algorithm (**SequenceMatcher at a 0.95 threshold**) to programmatically merge misspellings (e.g., unifying over 150 variations in a single transcript).

* **Traceability Assurance:** By capturing this dynamic entity mapping in a dictionary and applying it identically to the QA files, we achieved a **100% alignment rate**. Every answer in the CSV remains a perfect substring of the cleaned transcript, fulfilling the traceability requirement.



---



## Section: Analysis of Limitations and Linguistic Risks



While the implemented cleaning pipeline successfully unifies orthographic inconsistencies, it introduces specific linguistic and architectural risks that must be managed in later milestones.



### 1. The Risk of Semantic Collisions (Homographs)

A primary limitation of aggressive normalization is the creation of **homographs**—words that share the same spelling but different meanings.



> **Normalization Over-reach:** > * Mapping *Ya* (ي) and *Alef Maqsura* (ى) to a single character can make the name "Ali" (علي) indistinguishable from the preposition "on" (على).

> * **Entity vs. Verb Ambiguity:** Stripping prefixes or normalizing roots may turn a unique named entity into a common verb, causing the model to lose the distinction between a subject and an action.

> * **Loss of Specificity:** In the "Samurai" transcripts, the definite article (الـ) often distinguishes a general warrior from a specific historical title; separating it indiscriminately could degrade the system's ability to identify specific entities.



### 2. Vocabulary Bloat and Sequence Lengths

Because we prioritized extractive spans over morphological segmentation, our word-level vocabulary remains artificially inflated by attached prefixes (like ب, و, ف).



* **Computational Bottleneck:** When training the RNN and Transformer architectures from scratch in Milestone 2, this inflated vocabulary will result in a highly sparse, memory-intensive embedding matrix.

* **Context Chunking (MS3):** Our current preparation uses a sliding window chunking strategy (500 words, 50-word overlap) to fit model context limits. In MS3, we will need to experiment with context window strategies, such as **sentence-boundary-aware chunking** or **summarized-history**, to evaluate the trade-offs between token consumption and response accuracy.



### 3. Mitigation Strategies

To minimize these risks, we have adopted the following approach:



* **Conservative Mapping:** We prioritize "Mapping" over "Stripping." Instead of deleting prefixes, we maintain them to preserve the extractive span required for the RAG framework.

* **Selective Normalization:** Normalization is targeted at documented patterns found in our analysis (e.g., English-Arabic code-switching boundaries), rather than broad, rule-based changes that affect every word.

* **Reliance on Attention Mechanisms:** We acknowledge that Milestone 1 cannot solve all ambiguities. We anticipate that the simple transformer architecture in Milestone 2 will utilize its **Attention layers** to resolve these homographs by analyzing bidirectional context.-
