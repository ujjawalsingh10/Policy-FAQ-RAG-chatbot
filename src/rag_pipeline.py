import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PROMPT_TEMPLATE = """
You are a document-grounded assistant specialized in legal and policy documents.
Answer ONLY using the provided context excerpts.

If the answer is not present in the context, say:
"The answer is not explicitly stated in the provided document excerpts."

Context:
{context}

Question:
{question}
"""

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        page_num = doc.metadata.get("page", 0)
        formatted_docs.append(f"[Page {page_num}]\n{doc.page_content}")
    return "\n\n".join(formatted_docs)

class RAGPipeline:
    def __init__(
        self,
        db_dir: str = "vectordb",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_name: str = "llama-3.1-8b-instant",
        top_k: int = 4
    ):
        # Resolve to project root
        self.db_path = Path(__file__).resolve().parent.parent / db_dir
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Vector DB not found at: {self.db_path}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vectorstore = FAISS.load_local(
            str(self.db_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

        self.llm = ChatGroq(
            model=llm_name,
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.2
        )

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def get_sources(self, query: str, top_k: int = 4):
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return [
            {
                "chunk_id": f"Page {doc.metadata.get('page', 0)}",
                "page": doc.metadata.get('page', 0),
                "score": round(float(score), 4),
                "text": doc.page_content
            }
            for doc, score in results
        ]

    def stream(self, query: str):
        return self.chain.stream(query)