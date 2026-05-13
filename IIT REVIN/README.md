# BERT Paper — High-Precision Layout-Aware RAG System
**Revin Techno Solutions Technical Assessment**
**Author:** Alphin Das

A high-precision AI backend capable of semantic retrieval and complex reasoning
over the BERT research paper (Devlin et al., 2019), handling 2-column layouts,
figures, tables, and multi-hop questions.

---

## Demo Video
🎥 [Watch Walkthrough Video](#) ← replace with your Loom/YouTube link

---

## Architecture

```
PDF → PyMuPDF (layout-aware) → Gemini Vision (figures) →
RecursiveTextSplitter → Google Embeddings → ChromaDB →
MMR Retrieval → Gemini 2.0 Flash → Answer + Sources
```

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key
```

4. Download BERT paper and place in root folder:
```
https://aclanthology.org/N19-1423.pdf → save as N19-1423.pdf
```

5. Run ingestion:
```bash
python ingest.py
```

6. Launch Streamlit app:
```bash
streamlit run app.py
```

---

## Key Engineering Decisions

- **2-column fix:** PyMuPDF block-level bbox sorting (left col → right col)
- **Figure comprehension:** Gemini Vision API renders and describes diagrams
- **MMR retrieval:** Avoids redundant chunks, maximizes answer coverage
- **Temperature 0.1:** Minimizes hallucination on numerical/table data

See [STRATEGY.md](STRATEGY.md) for full engineering log.

---

## Project Structure

```
revin-bert-rag/
├── ingest.py          # Layout-aware PDF ingestion pipeline
├── retriever.py       # MMR retrieval + Gemini QA engine
├── app.py             # Streamlit UI
├── STRATEGY.md        # Architecture & engineering decisions
├── requirements.txt   # Dependencies
├── .env               # API keys (not committed)
└── N19-1423.pdf       # BERT paper (download separately)
```