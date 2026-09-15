# Dynamic AI Knowledge Assistant using Automated Wikipedia Knowledge Base and Advanced RAG

## Overview
The **Dynamic AI Knowledge Assistant** is an end-to-end Generative AI application that automatically builds its own knowledge base directly from Wikipedia entries based on user-provided keywords[cite: 1]. Unlike static Retrieval-Augmented Generation (RAG) systems, this application dynamic generates its dataset at runtime—collecting articles, cleaning text, indexing chunk embeddings, and serving context-grounded responses with full source citations[cite: 1].

The platform features a multi-page **Streamlit Dashboard** complete with dynamic dataset building, interactive knowledge base exploration, streaming chat interactions, and real-time retrieval analytics[cite: 1].

---

## Key Features & Use Cases
* **Automated Knowledge Acquisition:** Accepts dynamic topic keywords, fetches English Wikipedia entries via REST API, strips duplicates, and generates metadata tracking[cite: 1].
* **Flexible Text Chunking & Embeddings:** Benchmarks semantic chunking strategies paired with high-performance vector indexes using **FAISS** and **ChromaDB**[cite: 1].
* **Grounded Answer Generation:** Utilizes open-source LLMs (**Llama 3, Mistral, Gemma**) to answer user queries with explicit source article citations[cite: 1].
* **Multi-Turn Conversational Memory:** Maintains context across chat iterations for interactive, topic-specific learning[cite: 1].
* **Production Dashboard:** Includes interactive pages for Dataset Building, Knowledge Base Browsing, Streaming Chat Assistant, and Performance Analytics[cite: 1].

---

## Technical Stack
* **Language:** Python[cite: 1]
* **APIs & Web Scraping:** Wikipedia API, REST APIs[cite: 1]
* **RAG & Orchestration:** LangChain, LlamaIndex, Sentence Transformers[cite: 1]
* **Vector Databases:** FAISS, ChromaDB[cite: 1]
* **LLMs:** Llama 3, Mistral, Gemma[cite: 1]
* **Frontend & Analytics:** Streamlit, Pandas, Matplotlib[cite: 1]

---

