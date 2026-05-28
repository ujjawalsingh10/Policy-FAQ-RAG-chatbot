"""
document_processor.py
Loads a PDF, cleans it, splits into sentence-aware chunks, and saves to JSONL.
Uses LangChain's RecursiveCharacterTextSplitter.
"""
import json
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    records = []
    for i, chunk in enumerate(chunks, 1):
        records.append({
            "chunk_id": f"chunk_{i:04d}",
            "page": chunk.metadata.get("page", 0) + 1,
            "section": "General",
            "word_count": len(chunk.page_content.split()),
            "text": chunk.page_content.strip(),
            "source": chunk.metadata.get("source", pdf_path),
        })
    return records, chunks


def save_chunks(records, out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", default="chunks/chunks.jsonl")
    args = parser.parse_args()

    records, _ = process_pdf(args.pdf)
    save_chunks(records, args.out)
    print(f"Saved {len(records)} chunks → {args.out}")