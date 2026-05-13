# STRATEGY.md — Pipeline Architecture & Engineering Decisions
**Revin Techno Solutions Technical Assessment**
**Author:** Alphin Das
**System:** High-Precision Layout-Aware RAG for BERT Paper

---

## 1. Pipeline Architecture

```
PDF Input (BERT Paper)
        │
        ▼
┌─────────────────────────────┐
│   INGESTION LAYER           │
│                             │
│  PyMuPDF layout-aware parse │
│  ↓                          │
│  2-column sort (L→R)        │
│  ↓                          │
│  Gemini Vision (figures)    │
│  ↓                          │
│  RecursiveTextSplitter      │
│  chunk_size=800, overlap=150│
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   VECTOR STORE LAYER        │
│                             │
│  Google embedding-001       │
│  ↓                          │
│  ChromaDB (local)           │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   RETRIEVAL LAYER           │
│                             │
│  MMR Search                 │
│  k=6, fetch_k=20            │
│  lambda_mult=0.7            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   GENERATION LAYER          │
│                             │
│  Gemini 2.0 Flash           │
│  Temperature: 0.1           │
│  Context-only system prompt │
└─────────────┬───────────────┘
              │
              ▼
        Answer + Sources
```

---

## 2. Discovery & Fix Log

### Problem 1 — 2-Column Reading Order Breaks
**What happened:** Standard PyMuPDF `.get_text("text")` reads across
the full page width. On a 2-column paper, this means it reads:
`[left col line 1] [right col line 1] [left col line 2]...`
producing completely incoherent text.

**Fix:** Used `.get_text("dict")` to get block-level bounding boxes.
Separated blocks into left column (x0 < page_width/2) and right column
(x0 >= page_width/2). Sorted each column by vertical position (y).
Combined left column first, then right column. This produces
correct reading order for every page.

---

### Problem 2 — Figure/Diagram Content Invisible to Text Extraction
**What happened:** The BERT input representation diagram (Figure 1)
contains critical architecture details — token types, position embeddings,
segment embeddings. Pure text extraction returns nothing for these.

**Fix:** Detected image blocks using PyMuPDF's block type=1. Rendered
each figure page at 2x resolution. Sent to Gemini Vision API with a
targeted prompt to extract technical details, labels, and architectural
meaning. Stored descriptions as separate Document chunks tagged as
type="figure" — not split, preserved whole.

---

### Problem 3 — Table Row/Column Misalignment
**What happened:** Dense result tables (Table 1, Table 2) in the paper
have complex multi-column headers. Naive text extraction scrambles
row-column relationships.

**Fix:** Used overlapping chunks (150 token overlap) to ensure table
headers are always included with their data rows. System prompt
explicitly instructs the model to quote exact numerical values and
never approximate. MMR retrieval fetches multiple table chunks to
ensure complete coverage.

---

### Problem 4 — Context Fragmentation (Cut-off Concepts)
**What happened:** Small chunk sizes (256-512 tokens) split explanations
of MLM and NSP pre-training tasks mid-concept, causing incomplete answers.

**Fix:** Increased chunk_size to 800 tokens with 150 token overlap.
Used RecursiveCharacterTextSplitter with paragraph-aware separators
["\n\n", "\n", ". "] to prefer splitting at natural boundaries.

---

## 3. Design Decisions

### Why PyMuPDF over pdfplumber or pypdf?
PyMuPDF's `get_text("dict")` returns block-level bounding boxes with
pixel coordinates — essential for column detection. pdfplumber has good
table extraction but weaker block-level position data. pypdf has no
layout awareness at all.

### Why Google embedding-001 over OpenAI or sentence-transformers?
Consistent API ecosystem with Gemini. Produces 768-dimensional embeddings
optimized for semantic similarity. No additional API key required beyond
the existing Gemini key. Performs comparably to text-embedding-ada-002
on academic text.

### Why MMR over simple cosine similarity search?
Standard similarity search returns the top-6 most similar chunks — which
are often 6 nearly identical fragments from the same paragraph. MMR
(Maximal Marginal Relevance) balances relevance with diversity, ensuring
the retrieved chunks cover different aspects of the answer. Critical for
multi-hop questions.

### Why ChromaDB over Pinecone or Weaviate?
Local deployment — no API key, no latency, no cost. Sufficient for
single-document RAG. Production deployment would use Pinecone for
scalability across thousands of documents.

### Why temperature=0.1 for generation?
Academic QA requires factual precision. Higher temperatures introduce
creative variation which causes hallucination on numerical data. 0.1
produces deterministic, citation-grounded answers.

---

## 4. Quality Assurance Strategy (Production)

### Retrieval Accuracy
- Build ground truth QA pairs from the paper (50+ questions)
- Measure Hit@k — does the correct chunk appear in top-k results?
- Target: Hit@6 > 85%

### Answer Quality
- Use Gemini as judge: score each answer 1-5 on accuracy, completeness, grounding
- RAGAS framework: measure faithfulness, answer relevancy, context precision
- Target: Faithfulness > 0.90 (answer grounded in retrieved context)

### Hallucination Detection
- Cross-reference all numerical claims against source chunks
- Flag any number in the answer not present in retrieved context
- Target: 0 hallucinated numbers in table/comparison questions

### Regression Testing
- The 3 challenge queries become permanent CI test cases
- Run after any pipeline change — answer must remain semantically equivalent
- Use embedding similarity between expected and actual answer > 0.85

---

## 5. Known Limitations & Future Improvements

1. **Table extraction** — PyMuPDF text extraction for tables is imperfect.
   Production improvement: use `pdfplumber` specifically for table-heavy pages.

2. **Multi-page figures** — Currently renders full page for figures.
   Improvement: detect exact figure bounding box and crop precisely.

3. **Citation tracking** — Currently cites page numbers.
   Improvement: cite section names (Abstract, Section 3.1, etc.)

4. **Scalability** — ChromaDB is single-node.
   Production: migrate to Pinecone or Weaviate for multi-document corpora.

### Problem 5 — Embedding Model Deprecation
During development, `models/embedding-001` returned a 404 error.
Discovered via ListModels API call that the correct model name
is `models/gemini-embedding-001`. Updated both ingest.py and
retriever.py accordingly.

### Problem 6 — Rate Limiting on Free Tier
The free tier allows 100 embedding requests per minute.
With 134 document chunks, the pipeline hit quota on batch 3.
Fixed by adding 65-second delays between batches of 40 documents.