# Ask My Docs — Production RAG Application

A production-grade Retrieval Augmented Generation (RAG) system 
that lets you upload any PDF or text document and ask questions 
about it in natural language.

# Features
1. **Hybrid Retrieval** — combines semantic search + keyword search (BM25 + dense vectors)
2. **Cross-Encoder Reranking** — Cohere reranks results for maximum precision  
3. **Citation Enforcement** — every answer cites the exact source chunk
4. **RAGAS Evaluation Pipeline** — automated quality scoring with CI gate
5.  **Gradio Web Interface** — clean 4-tab UI for upload, chat, and evaluation

## Tech Stack
1. **Qdrant** — vector database with hybrid search
2. **Cohere** — cross-encoder reranking  
3. **Google Gemini** — answer generation
4. **RAGAS** — evaluation framework
5. **Gradio** — web interface
6. **GitHub Actions** — CI/CD evaluation gate

## How to Run
1. Open in Google Colab
2. Add your API keys in Colab Secrets:
   - `COHERE_API_KEY`
   - `OPENAI_API_KEY` (your Gemini key)
3. Run all cells
4. Upload your documents and start asking questions
   
