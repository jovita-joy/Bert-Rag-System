
import os
import fitz  # PyMuPDF
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import google.generativeai as genai
from PIL import Image
import io
import base64

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
PDF_PATH = "N19-1423.pdf"           # BERT paper — place in same folder
CHROMA_DB_PATH = "./chroma_db"      # where vectors are stored
COLLECTION_NAME = "bert_paper"

# ─────────────────────────────────────────────
# STEP 1 — LAYOUT-AWARE TEXT EXTRACTION
# The key challenge: 2-column PDFs read incorrectly with
# standard loaders. PyMuPDF's "dict" mode gives us block
# positions so we can sort by column then by vertical position.
# ─────────────────────────────────────────────

def extract_text_layout_aware(pdf_path: str) -> list[dict]:
    """
    Extract text from each page respecting 2-column reading order.
    Returns list of {page_num, text, type} dicts.
    """
    doc = fitz.open(pdf_path)
    all_chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_width = page.rect.width

        # Get blocks with position data
        blocks = page.get_text("dict")["blocks"]

        # Separate text blocks and image blocks
        text_blocks = []
        image_blocks = []

        for block in blocks:
            if block["type"] == 0:  # text block
                text_blocks.append(block)
            elif block["type"] == 1:  # image block
                image_blocks.append(block)

        # ── CRITICAL: Sort blocks by column then by vertical position ──
        # Left column: x0 < page_width/2
        # Right column: x0 >= page_width/2
        # This fixes the reading order break in 2-column papers

        left_col = sorted(
            [b for b in text_blocks if b["bbox"][0] < page_width / 2],
            key=lambda b: b["bbox"][1]  # sort by y (top position)
        )
        right_col = sorted(
            [b for b in text_blocks if b["bbox"][0] >= page_width / 2],
            key=lambda b: b["bbox"][1]
        )

        # Combine: left column first, then right column
        ordered_blocks = left_col + right_col

        # Extract text from ordered blocks
        page_text = ""
        for block in ordered_blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    page_text += span["text"] + " "
            page_text += "\n"

        if page_text.strip():
            all_chunks.append({
                "page_num": page_num + 1,
                "text": page_text.strip(),
                "type": "text",
                "source": f"Page {page_num + 1}"
            })

        # ── HANDLE IMAGES/FIGURES ──
        for i, img_block in enumerate(image_blocks):
            all_chunks.append({
                "page_num": page_num + 1,
                "text": f"[FIGURE on page {page_num + 1}]",
                "type": "image",
                "bbox": img_block["bbox"],
                "source": f"Page {page_num + 1} - Figure {i+1}"
            })

    doc.close()
    print(f"✅ Extracted {len(all_chunks)} blocks from {pdf_path}")
    return all_chunks


# ─────────────────────────────────────────────
# STEP 2 — FIGURE COMPREHENSION WITH GEMINI VISION
# Figures contain crucial architecture diagrams.
# We render each figure page and describe it with Gemini Vision.
# ─────────────────────────────────────────────

def describe_figures_with_gemini(pdf_path: str) -> list[dict]:
    """
    Render pages containing figures and use Gemini Vision
    to generate textual descriptions of diagrams.
    """
    doc = fitz.open(pdf_path)
    figure_descriptions = []
    model = genai.GenerativeModel("gemini-2.0-flash")

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        image_blocks = [b for b in blocks if b["type"] == 1]

        if not image_blocks:
            continue

        # Render full page as image at high resolution
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for clarity
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        # Send to Gemini Vision for description
        try:
            img_part = {
                "mime_type": "image/png",
                "data": base64.b64encode(img_bytes).decode()
            }
            prompt = """You are analyzing a page from the BERT research paper.
            Describe all figures, diagrams, and architectural illustrations on this page
            in detail. Include: what the figure shows, labels, arrows, components,
            and what concept it represents. Be specific and technical."""

            response = model.generate_content([prompt, img_part])
            description = response.text

            figure_descriptions.append({
                "page_num": page_num + 1,
                "text": f"[FIGURE DESCRIPTION - Page {page_num + 1}]: {description}",
                "type": "figure_description",
                "source": f"Page {page_num + 1} - Figure (Vision Analysis)"
            })
            print(f"  📊 Described figure on page {page_num + 1}")

        except Exception as e:
            print(f"  ⚠️ Could not describe figure on page {page_num + 1}: {e}")

    doc.close()
    return figure_descriptions


# ─────────────────────────────────────────────
# STEP 3 — SMART CHUNKING
# Standard chunking breaks concepts mid-sentence.
# We use RecursiveCharacterTextSplitter with overlap
# to maintain complete thought boundaries.
# ─────────────────────────────────────────────

def chunk_documents(raw_blocks: list[dict]) -> list[Document]:
    """
    Convert raw blocks into LangChain Documents with smart chunking.
    Text blocks get chunked. Figure descriptions stay whole.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # large enough for complete concepts
        chunk_overlap=150,     # overlap prevents context loss at boundaries
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    documents = []

    for block in raw_blocks:
        if block["type"] == "image":
            continue  # skip raw image placeholders

        if block["type"] == "figure_description":
            # Keep figure descriptions as single chunks — don't split them
            documents.append(Document(
                page_content=block["text"],
                metadata={
                    "source": block["source"],
                    "page": block["page_num"],
                    "type": "figure"
                }
            ))
        else:
            # Split regular text blocks
            chunks = splitter.split_text(block["text"])
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # skip tiny fragments
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": block["source"],
                            "page": block["page_num"],
                            "type": "text"
                        }
                    ))

    print(f"✅ Created {len(documents)} document chunks")
    return documents


# ─────────────────────────────────────────────
# STEP 4 — EMBED AND STORE IN CHROMADB
# ─────────────────────────────────────────────

def build_vector_store(documents: list[Document]) -> None:
    print("⏳ Building vector store — this takes 3-4 minutes...")

    import time
    from langchain_chroma import Chroma as ChromaNew

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    batch_size = 40  # reduced to stay under rate limit
    all_docs = documents

    print(f"Total documents to embed: {len(all_docs)}")

    # First batch creates the store
    first_batch = all_docs[:batch_size]
    vectorstore = ChromaNew.from_documents(
        documents=first_batch,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    print(f"  ✅ Batch 1 done ({len(first_batch)} docs)")
    print("  ⏳ Waiting 65 seconds for rate limit reset...")
    time.sleep(65)  # wait for rate limit window to reset

    # Remaining batches
    for i in range(batch_size, len(all_docs), batch_size):
        batch = all_docs[i:i+batch_size]
        vectorstore.add_documents(batch)
        batch_num = i//batch_size + 1
        print(f"  ✅ Batch {batch_num} done ({len(batch)} docs)")
        if i + batch_size < len(all_docs):
            print("  ⏳ Waiting 65 seconds for rate limit reset...")
            time.sleep(65)

    print(f"✅ Vector store built with {len(all_docs)} chunks!")

def run_ingestion():
    print("\n" + "="*50)
    print("REVIN BERT RAG — INGESTION PIPELINE")
    print("="*50 + "\n")

    # Step 1: Extract text with layout awareness
    print("📄 Step 1: Extracting text with layout-aware parsing...")
    raw_blocks = extract_text_layout_aware(PDF_PATH)

    # Step 2: Describe figures with Gemini Vision
    print("\n🖼️  Step 2: Analyzing figures with Gemini Vision...")
    figure_blocks = describe_figures_with_gemini(PDF_PATH)

    # Combine text and figure descriptions
    all_blocks = raw_blocks + figure_blocks
    print(f"\n📦 Total blocks: {len(all_blocks)}")

    # Step 3: Smart chunking
    print("\n✂️  Step 3: Chunking documents...")
    documents = chunk_documents(all_blocks)

    # Step 4: Build vector store
    print("\n🧠 Step 4: Embedding and storing in ChromaDB...")
    vectorstore = build_vector_store(documents)

    print("\n" + "="*50)
    print("✅ INGESTION COMPLETE — Ready for querying!")
    print("="*50 + "\n")

    return vectorstore


if __name__ == "__main__":
    run_ingestion()
