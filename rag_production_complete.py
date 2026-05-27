# ============================================================
#   PRODUCTION RAG APPLICATION — "Ask My Docs"
#   Components: Hybrid Retrieval + Reranking + Citations
#               + RAGAS Evaluation + Gradio UI
# ============================================================

# ── STEP 0: Install all dependencies ────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for package in [
    "pypdf",
    "gradio",
    "ragas",
    "datasets",
    "langchain",
    "langchain-google-genai",   # ✅ replaces langchain-community for Gemini
    "google-genai",             # ✅ new Google Gemini SDK
    "sentence-transformers",
    "qdrant-client[fastembed]",
    "cohere"
]:
    install(package)

# ── STEP 1: Imports ─────────────────────────────────────────
import os, uuid, io, time, json
from google.colab import userdata
from qdrant_client import QdrantClient
from google import genai
import cohere
import gradio as gr
import pypdf

# ── STEP 2: API Keys ────────────────────────────────────────
os.environ["COHERE_API_KEY"] = userdata.get("COHERE_API_KEY")
GEMINI_KEY                   = userdata.get("OPENAI_API_KEY")   # your Gemini key

# ✅ RAGAS needs this env variable name to find Gemini
os.environ["GOOGLE_API_KEY"] = GEMINI_KEY

cohere_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
gemini_client = genai.Client(api_key=GEMINI_KEY)

COLLECTION_NAME = "production_rag_docs"

# ── STEP 3: Qdrant Vector Store ──────────────────────────────
qdrant = QdrantClient(location=":memory:")
qdrant.set_model("BAAI/bge-small-en-v1.5")
qdrant.set_sparse_model("prithivida/Splade_PP_en_v1")

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=qdrant.get_fastembed_vector_params(),
    sparse_vectors_config=qdrant.get_fastembed_sparse_vector_params()
)

# ── STEP 4: Sliding Window Chunker ──────────────────────────
def sliding_window_chunks(text: str, chunk_size: int = 400, overlap: int = 80):
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 60:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ── STEP 5: PDF / TXT Text Extractor ────────────────────────
def extract_text(filename: str, content: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages  = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n\n".join(pages)
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")

# ── STEP 6: Document Indexer ────────────────────────────────
indexed_doc_names = []

def index_documents(files_dict: dict) -> str:
    global indexed_doc_names
    documents, metadata = [], []

    for filename, content in files_dict.items():
        text = extract_text(filename, content)
        if not text.strip():
            continue

        chunks = sliding_window_chunks(text, chunk_size=400, overlap=80)
        indexed_doc_names.append(filename)

        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadata.append({
                "chunk_id": f"chunk_{filename.replace('.','_').replace(' ','_')}_{i}",
                "source"  : filename,
                "page"    : i + 1
            })

    if not documents:
        return "❌ No content found. Please upload valid PDF or TXT files."

    qdrant.add(
        collection_name=COLLECTION_NAME,
        documents=documents,
        metadata=metadata,
        ids=[str(uuid.uuid4()) for _ in documents]
    )
    return f"✅ Indexed {len(documents)} chunks from {len(files_dict)} file(s)."

# ── STEP 7: Hybrid Retriever + Cohere Reranker ──────────────
def retrieve_and_rerank(query: str, top_k: int = 3) -> list:
    raw = qdrant.query(
        collection_name=COLLECTION_NAME,
        query_text=query,
        limit=10
    )

    docs = [r.metadata["document"] for r in raw]
    if not docs:
        return []

    reranked = cohere_client.rerank(
        query=query,
        documents=docs,
        top_n=top_k,
        model="rerank-english-v3.0"
    )

    return [raw[hit.index].metadata for hit in reranked.results]

# ── STEP 8: Gemini Answer Generator with Retry + Fallback ───
def generate_answer(query: str, context_blocks: list) -> str:
    if not context_blocks:
        return "⚠️ No relevant context found. Please upload documents first."

    context_str = ""
    for b in context_blocks:
        context_str += (
            f"---\n"
            f"ID: {b['chunk_id']}\n"
            f"Source: {b['source']}\n"
            f"Text: {b['document']}\n"
            f"---\n"
        )

    system_instruction = (
        "You are a strict, factual enterprise assistant. "
        "Answer ONLY using the verified context blocks provided.\n"
        "Rules:\n"
        "1. Cite the exact Chunk ID after every claim, e.g. [chunk_policy_01].\n"
        "2. If the answer is absent from context, reply: "
        "'I cannot answer this based on the provided documentation.'\n"
        "3. Never use outside knowledge."
    )

    user_prompt = f"Context Blocks:\n{context_str}\n\nQuestion: {query}"
    models      = ["gemini-2.5-flash", "gemini-2.0-flash"]

    for model in models:
        for attempt in range(3):
            try:
                resp = gemini_client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config={"system_instruction": system_instruction, "temperature": 0.0}
                )
                return resp.text
            except Exception as e:
                err = str(e)
                if "503" in err or "UNAVAILABLE" in err:
                    wait = 2 ** attempt
                    print(f"⚠️ [{model}] busy, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"❌ [{model}] error: {err[:120]}")
                    break
        print(f"↩️ Switching to fallback model...")

    return "⚠️ All models unavailable. Please try again shortly."

# ── STEP 9: RAGAS Evaluation Pipeline ───────────────────────
# ✅ FIX: Configure RAGAS to use Gemini via LangChain, NOT Vertex AI
EVAL_LOG_FILE = "eval_results.json"

def setup_ragas_llm():
    """
    Tells RAGAS to use Gemini Flash via LangChain instead of
    trying to use Vertex AI (which needs Google Cloud credentials).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm        = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_KEY)

    ragas_llm  = LangchainLLMWrapper(llm)
    ragas_emb  = LangchainEmbeddingsWrapper(embeddings)

    return ragas_llm, ragas_emb

def run_ragas_evaluation(test_questions: list) -> dict:
    print("\n🧪 Running RAGAS Evaluation Pipeline...")

    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset

    # ✅ Wire RAGAS to use Gemini
    ragas_llm, ragas_emb = setup_ragas_llm()
    faithfulness.llm           = ragas_llm
    faithfulness.embeddings    = ragas_emb
    answer_relevancy.llm       = ragas_llm
    answer_relevancy.embeddings = ragas_emb
    context_precision.llm      = ragas_llm
    context_precision.embeddings = ragas_emb

    questions, answers, contexts, ground_truths = [], [], [], []

    for q in test_questions:
        ctx_blocks = retrieve_and_rerank(q, top_k=3)
        answer     = generate_answer(q, ctx_blocks)
        ctx_texts  = [b["document"] for b in ctx_blocks]

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx_texts)
        ground_truths.append("")

    dataset = Dataset.from_dict({
        "question"     : questions,
        "answer"       : answers,
        "contexts"     : contexts,
        "ground_truth" : ground_truths
    })

    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    scores  = dict(results)

    FAITHFULNESS_THRESHOLD = 0.70
    passed = scores.get("faithfulness", 0) >= FAITHFULNESS_THRESHOLD

    output = {
        "scores" : scores,
        "passed" : passed,
        "gate"   : f"{'✅ PASSED' if passed else '❌ FAILED'} — faithfulness={scores.get('faithfulness', 0):.2f} (threshold={FAITHFULNESS_THRESHOLD})"
    }

    with open(EVAL_LOG_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{output['gate']}")
    print(f"Full scores: {scores}")
    return output

# ── STEP 10: GitHub Actions CI YAML Generator ───────────────
def generate_ci_yaml():
    yaml = """# .github/workflows/eval.yml
name: RAG Evaluation Gate

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run RAGAS Evaluation
        env:
          COHERE_API_KEY: ${{ secrets.COHERE_API_KEY }}
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
        run: python eval_runner.py

      - name: Check Gate
        run: |
          python -c "
          import json, sys
          r = json.load(open('eval_results.json'))
          print(r['gate'])
          sys.exit(0 if r['passed'] else 1)
          "
"""
    os.makedirs(".github/workflows", exist_ok=True)
    with open(".github/workflows/eval.yml", "w") as f:
        f.write(yaml)

# ── STEP 11: Gradio UI ──────────────────────────────────────
def upload_handler(files):
    if not files:
        return "No files uploaded."
    files_dict = {}
    for f in files:
        with open(f.name, "rb") as fp:
            files_dict[os.path.basename(f.name)] = fp.read()
    return index_documents(files_dict)

def chat_handler(message, history):
    ctx    = retrieve_and_rerank(message, top_k=3)
    answer = generate_answer(message, ctx)
    history.append((message, answer))
    return "", history

def eval_handler(questions_text):
    questions = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]
    if not questions:
        return "Please enter at least one test question."
    result = run_ragas_evaluation(questions)
    lines  = [result["gate"]]
    for k, v in result["scores"].items():
        lines.append(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return "\n".join(lines)

with gr.Blocks(title="Ask My Docs — Production RAG", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 📚 Ask My Docs — Production RAG System
    **Hybrid Retrieval · Cross-Encoder Reranking · Citation Enforcement · RAGAS Evaluation**
    """)

    with gr.Tab("📂 Upload Documents"):
        gr.Markdown("Upload your PDF or TXT files. They will be chunked and indexed automatically.")
        upload_input  = gr.File(file_count="multiple", label="Upload Files (.pdf or .txt)")
        upload_button = gr.Button("Index Documents", variant="primary")
        upload_status = gr.Textbox(label="Status", interactive=False)
        upload_button.click(fn=upload_handler, inputs=upload_input, outputs=upload_status)

    with gr.Tab("💬 Ask Questions"):
        gr.Markdown("Ask anything about your uploaded documents. Every answer includes citations.")
        chatbot   = gr.Chatbot(height=400, label="Document Assistant")
        msg_input = gr.Textbox(placeholder="Type your question here...", label="Your Question")
        send_btn  = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Chat")
        send_btn.click(fn=chat_handler,  inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
        clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_input])

    with gr.Tab("🧪 Run Evaluation (CI Gate)"):
        gr.Markdown("""
        Enter one test question per line. The system will:
        1. Retrieve context for each question
        2. Generate answers
        3. Score with RAGAS (Faithfulness, Relevancy, Precision)
        4. Apply CI gate — fails if faithfulness < 0.70
        """)
        eval_input  = gr.Textbox(
            lines=5,
            placeholder="What is the main topic?\nWhat are the key findings?",
            label="Test Questions (one per line)"
        )
        eval_button = gr.Button("Run Evaluation", variant="primary")
        eval_output = gr.Textbox(label="Evaluation Results", interactive=False, lines=8)
        eval_button.click(fn=eval_handler, inputs=eval_input, outputs=eval_output)

    with gr.Tab("⚙️ CI Setup"):
        gr.Markdown("""
        ## GitHub Actions CI Gate Setup

        **`eval_runner.py`** — add this file to your repo:
        ```python
        from rag_production_complete import run_ragas_evaluation
        import sys
        TEST_QUESTIONS = [
            "What is the main topic of the document?",
            "What are the key findings?"
        ]
        result = run_ragas_evaluation(TEST_QUESTIONS)
        sys.exit(0 if result["passed"] else 1)
        ```

        **`requirements.txt`**:
        ```
        pypdf
        gradio
        ragas
        datasets
        qdrant-client[fastembed]
        cohere
        langchain-google-genai
        google-genai
        ```

        **GitHub Secrets to add** (Settings → Secrets → Actions):
        - `COHERE_API_KEY`
        - `GOOGLE_API_KEY`  ← your Gemini key
        """)

# ── STEP 12: Launch ─────────────────────────────────────────
print("\n" + "="*60)
print("🚀 Launching Production RAG System...")
print("="*60 + "\n")

generate_ci_yaml()
demo.launch(share=True, debug=False)
