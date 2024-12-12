from dataclasses import dataclass
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

import voyageai # for optional reranking.

import bm25s
from tqdm import tqdm
from typing import Dict, List, Tuple
from uuid import uuid4

@dataclass 
class SearchConfig:
    chunk_size: int = 512
    chunk_overlap: int = 128
    semantic_weight: float = 0.8 
    lexical_weight: float = 0.2
    initial_k: int = 20
    final_k: int = 10

class RAGSearchEngine:
    def __init__(
        self,
        contextualizer: BaseChatModel = None,
        embedding_model: Embeddings = None,
        vector_store: VectorStore = None,
        reranker : voyageai.Client = None,
        config: SearchConfig = None
    ):
        self.contextualizer = contextualizer

        self.embeddings = embedding_model
            
        self.vector_store = vector_store(self.embeddings)
        self.bm25 = bm25s.BM25()
        self.reranker = reranker
            
        self.config = config

        self.document_map: Dict[str, str] = {}  # doc_id -> content
        self.chunk_map: List[Tuple[str, str]] = []  # list of (doc_id, chunk)
        self.contextualized_chunks: List[str] = []

    def _contextualize_chunk(self, doc_content: str, chunk_content: str) -> str:
        prompt = f"""

        Document content: 
        {doc_content}

        Chunk content: 
        {chunk_content}

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """
        return self.contextualizer.invoke(prompt).content

    def _chunk_documents(self, documents: List[str]) -> None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        for doc in documents:
            doc_id = str(uuid4())
            self.document_map[doc_id] = doc
            
            chunks = text_splitter.split_text(doc)
            for chunk in chunks:
                self.chunk_map.append((doc_id, chunk))

    def process_documents(self, documents: List[str]) -> None:
        self._chunk_documents(documents)
        
        for doc_id, chunk in tqdm(self.chunk_map, desc="Processing chunks"):
            doc_content = self.document_map[doc_id]
            context = self._contextualize_chunk(doc_content, chunk)
            enhanced = f"{chunk}\n\nContext: {context}"
            self.contextualized_chunks.append(enhanced)

        self.vector_store.add_texts(self.contextualized_chunks)
        self.bm25.index(bm25s.tokenize(self.contextualized_chunks))

    def search(self, query: str) -> list[str]:

        num_chunks = len(self.contextualized_chunks)

        safe_initial_k = min(num_chunks, self.config.initial_k)
        safe_final_k = min(safe_initial_k, self.config.final_k)

        if safe_initial_k == safe_final_k:
            # Prevents reranking if the chunk size is lower than initial_k
            self.reranker = None

        vector_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=safe_initial_k
        )

        bm25_doc_ids, bm25_scores = self.bm25.retrieve(
            bm25s.tokenize(query),
            k=safe_initial_k
        )

        combined_scores = {}
        
        for doc, score in vector_results:
            doc_idx = self.contextualized_chunks.index(doc.page_content)
            combined_scores[doc_idx] = self.config.semantic_weight * score
            
        for doc_idx, score in zip(bm25_doc_ids[0], bm25_scores[0]):
            if doc_idx in combined_scores:
                combined_scores[doc_idx] += self.config.lexical_weight * score
            else:
                combined_scores[doc_idx] = self.config.lexical_weight * score

        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:safe_initial_k]
        
        results = [self.contextualized_chunks[idx] for idx, _ in sorted_results]

        if self.reranker:
            reranked = self.reranker.rerank(
                query,
                results,
                model="rerank-2",
                top_k=self.config.final_k
            )
            return [r.document for r in reranked.results]
            
        return results[:safe_final_k]