
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from groq import Groq

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
if api_key:
    print(f"DEBUG: Google API Key loaded: {api_key[:10]}...")
if groq_key:
    print(f"DEBUG: Groq API Key loaded: {groq_key[:10]}...")
else:
    print("DEBUG WARNING: No GROQ_API_KEY found in .env!")

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "bert_paper"

# ─────────────────────────────────────────────
# LOAD VECTOR STORE
# ─────────────────────────────────────────────

def load_vectorstore() -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return vectorstore


# ─────────────────────────────────────────────
# HIGH-PRECISION RETRIEVAL
# Uses MMR (Maximal Marginal Relevance) to avoid
# redundant chunks and maximize coverage of the answer.
# ─────────────────────────────────────────────

def retrieve_relevant_chunks(
    vectorstore: Chroma,
    query: str,
    k: int = 2
) -> list[Document]:
    """
    Retrieve top-k most relevant chunks using MMR.
    MMR balances relevance + diversity — avoids returning
    5 nearly identical chunks from the same paragraph.
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 30,        # Reduced from 20 to save processing
            "lambda_mult": 0.7
        }
    )
    docs = retriever.invoke(query)
    return docs


# ─────────────────────────────────────────────
# QA ENGINE — STATELESS SINGLE-TURN
# System prompt engineered to:
# 1. Answer ONLY from retrieved context
# 2. Handle tables and numerical data carefully
# 3. Reference specific pages
# 4. Refuse to hallucinate
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise academic research assistant analyzing the BERT paper
(Devlin et al., 2019): "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding".

STRICT RULES:
1. Answer ONLY using the provided context chunks. Never use outside knowledge.
2. For numerical/table data: quote exact values. Never approximate or guess numbers.
3. If a figure description is in the context, use it to answer visual questions.
4. If the answer spans multiple chunks, synthesize them coherently.
5. If the context does not contain enough information, say exactly:
   "The retrieved context does not contain sufficient information to answer this question accurately."
6. Always cite which page(s) your answer comes from.
7. Be precise, technical, and complete. Do not truncate answers.

Context chunks from the BERT paper:
{context}

Question: {question}

Answer (with page citations):"""


def answer_question(query: str, vectorstore: Chroma) -> dict:
    # Retrieve relevant chunks
    docs = retrieve_relevant_chunks(vectorstore, query)

    if not docs:
        return {
            "answer": "No relevant content found in the document.",
            "sources": [],
            "query": query,
            "chunks_retrieved": 0
        }

    # Build context string from retrieved chunks
    # Trim each chunk to 800 chars to stay within free-tier token limits
    MAX_CHUNK_CHARS = 800
    context_parts = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "?")
        doc_type = doc.metadata.get("type", "text")
        content = doc.page_content[:MAX_CHUNK_CHARS]
        context_parts.append(
            f"[Chunk {i+1} | Page {page} | Type: {doc_type}]\n{content}"
        )
    context = "\n\n---\n\n".join(context_parts)
    print(f"DEBUG: Total context chars being sent to AI: {len(context)}")

    # Build prompt
    prompt = SYSTEM_PROMPT.format(context=context, question=query)

    # Generate answer using Groq (free tier — Llama 3)
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    try:
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        answer = chat_completion.choices[0].message.content
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            return {
                "answer": "⚠️ **Rate limit hit on Groq.** Please wait 30 seconds and try again.",
                "sources": [],
                "query": query,
                "chunks_retrieved": 0
            }
        raise e

    # Format sources
    sources = []
    for doc in docs:
        sources.append({
            "page": doc.metadata.get("page", "?"),
            "type": doc.metadata.get("type", "text"),
            "source": doc.metadata.get("source", ""),
            "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
        })

    return {
        "answer": answer,
        "sources": sources,
        "query": query,
        "chunks_retrieved": len(docs)
    }

# ─────────────────────────────────────────────
# TEST — Run sample queries
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading vector store...")
    vs = load_vectorstore()

    # Test with the 3 challenge queries from the task
    test_queries = [
        "What are the two pre-training tasks used in BERT and how do they work?",
        "Compare the BERT-BASE and BERT-LARGE model sizes — how many parameters does each have?",
        "What does the input representation diagram show about how BERT constructs its input?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print(f"{'='*60}")
        result = answer_question(query, vs)
        print(f"A: {result['answer']}")
        print(f"\n📚 Retrieved {result['chunks_retrieved']} chunks")
