from .abstract_model import AbstractModel
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


from typing import List

import logging
log = logging.getLogger("chromadb")
log.setLevel(logging.WARNING)


def _load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )


class RagModel(AbstractModel):
    db_query = ""

    def __init__(self, model_name: str, chunk_size: int, chunk_overlap: int):
        super().__init__(model_name)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._docs = []
        self._splits = []
        self._embedding = _load_embeddings()
        self._vectordb = None

    def _load_docs(self):
        self._docs = WikipediaLoader(query=self.db_query, load_max_docs=2).load()

    def _split_docs(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self._docs = text_splitter.split_documents(self._docs)

    def _store_vector(self):
        self._vectordb = Chroma.from_documents(
            documents=self._docs,
            embedding=self._embedding,
        )

    def _vector_search(self, prompt):
        return self._vectordb.similarity_search(query=prompt, k=3)

    def _process_query(self, query):
        if not self.db_query:
            prompt = f"""Who is the subject of the following sentence?:
            {query}
            The subject of the sentence is: """
            self.db_query = self._get_response(prompt)
            self._process_query(query=query)
        self._load_docs()
        self._split_docs()
        self._store_vector()
        context: List[str] = self._vector_search(query)
        context_text = [doc.page_content for doc in context]
        context_str = '\n'.join(context_text)
        prompt = f"""Using the following context, answer the question below:
        {context_str}
        Question: {query}
        Helpful Answer:"""

        return self._get_response(prompt)

    def chat(self, text: str):
        return self._process_query(text)
