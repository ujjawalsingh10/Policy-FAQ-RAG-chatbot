"""
build_index.py
Builds a LangChain-compatible FAISS vector store.
"""

import json
import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def load_chunks(path: str):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def build_faiss_index(
    chunks_path: str,
    out_dir: str = "vectordb",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    records = load_chunks(chunks_path)

    # Convert records into LangChain Documents
    documents = [
        Document(
            page_content=r["text"],
            metadata={k: v for k, v in r.items() if k != "text"}
        )
        for r in records
    ]

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs={
        "normalize_embeddings": True
    }
)

    # Create vector store
    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    # Save in LangChain format
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(out))

    print(f"FAISS vector DB saved → {out_dir}")
    print(f"Total chunks indexed: {len(documents)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--chunks", required=True)
    parser.add_argument("--out_dir", default="vectordb")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    args = parser.parse_args()

    build_faiss_index(
        args.chunks,
        args.out_dir,
        args.model
    )