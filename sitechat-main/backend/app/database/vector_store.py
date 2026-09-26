"""
Vector store operations using FAISS.

Tenant isolation note:
  FAISS does not support in-place deletion of individual vectors.  Rather than
  rebuilding the whole index on every delete (expensive and risky), we use a
  *logical deletion* approach:

  - A sidecar file ``deleted_filters.json`` is stored alongside the FAISS index.
  - Each entry is a dict of metadata key→value pairs that describe what is deleted
    (e.g. ``{"site_id": "abc123"}`` or ``{"url": "doc://abc123/some-doc-id"}``).
  - ``delete_by_metadata(filter)`` appends to this list and persists the sidecar.
  - ``is_logically_deleted(metadata)`` returns True if the metadata matches any filter.
  - The RAG retrieval layer calls ``is_logically_deleted`` during candidate filtering,
    so deleted content never surfaces to users.

  Physical compaction (actually removing vectors from FAISS) can be triggered by an
  admin via ``POST /api/crawl/reindex``, which rebuilds the index from MongoDB.
"""
import json
import os
import pickle
from typing import Any, Dict, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from loguru import logger

from app.config import settings


class VectorStore:
    """FAISS vector store with HuggingFace embeddings."""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self._initialized = False
        self.index_path = os.path.join(settings.CHROMA_PERSIST_DIR, "faiss_index")
        # Logical deletion: list of metadata-filter dicts
        self._deleted_filters: List[Dict[str, Any]] = []
        self._sidecar_path: str = os.path.join(settings.CHROMA_PERSIST_DIR, "deleted_filters.json")
    
    def initialize(self):
        """Initialize the vector store and embeddings."""
        if self._initialized:
            return
        
        try:
            # Create directories
            os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
            
            # Initialize embeddings (HuggingFace - runs locally)
            logger.info("Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Try to load existing index
            if os.path.exists(self.index_path):
                logger.info("Loading existing FAISS index...")
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                logger.info("Creating new FAISS index...")
                # Create empty index with a dummy document
                self.vector_store = FAISS.from_texts(
                    ["Initial document"],
                    self.embeddings,
                    metadatas=[{"source": "init"}]
                )
                self._save_index()
            
            self._initialized = True
            dim = self.vector_store.index.d if self.vector_store and hasattr(self.vector_store, 'index') else 384
            logger.info(f"[RAG][EMBED] model=sentence-transformers/all-MiniLM-L6-v2 dimension={dim}")
            logger.info("Vector store initialized successfully")
            
            # Load logical-deletion sidecar
            self._load_sidecar()
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _save_index(self):
        """Save the FAISS index to disk."""
        if self.vector_store:
            self.vector_store.save_local(self.index_path)

    def _load_sidecar(self):
        """Load logically-deleted metadata filters from sidecar file."""
        if os.path.exists(self._sidecar_path):
            try:
                with open(self._sidecar_path, "r", encoding="utf-8") as fp:
                    self._deleted_filters = json.load(fp)
                logger.info(f"Loaded {len(self._deleted_filters)} logical deletion filter(s)")
            except Exception as e:
                logger.warning(f"Could not load deletion sidecar: {e}")
                self._deleted_filters = []
        else:
            self._deleted_filters = []

    def _save_sidecar(self):
        """Persist logically-deleted metadata filters to sidecar file."""
        try:
            with open(self._sidecar_path, "w", encoding="utf-8") as fp:
                json.dump(self._deleted_filters, fp, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist deletion sidecar: {e}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        if not self._initialized:
            self.initialize()
        
        try:
            if not documents:
                return []
            
            # Add documents
            ids = self.vector_store.add_documents(documents)
            
            # Save to disk
            self._save_index()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Dict = None
    ) -> List[Document]:
        """Search for similar documents."""
        if not self._initialized:
            self.initialize()
        
        k = k or settings.RETRIEVAL_K
        
        try:
            # FAISS doesn't support filtering directly, so we fetch more and filter
            results = self.vector_store.similarity_search(query, k=k)
            
            # Exclude logically deleted docs
            results = [doc for doc in results if not self.is_logically_deleted(doc.metadata)]

            # Apply filter if provided
            if filter:
                results = [
                    doc for doc in results
                    if all(doc.metadata.get(key) == value for key, value in filter.items())
                ]
            
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def similarity_search_with_score(
    self,
    query: str,
    k: int = None,
    filter: Dict = None,
    fetch_k: int = None
) -> List[tuple]:
     """Search for similar documents with relevance scores."""
     if not self._initialized:
        self.initialize()

     k = k or settings.RETRIEVAL_K
     fetch_k = fetch_k or k

     try:
         results = self.vector_store.similarity_search_with_score(
            query,
            k=fetch_k
        )

        # Exclude logically deleted docs
         results = [
            (doc, score) for doc, score in results
            if not self.is_logically_deleted(doc.metadata)
        ]

        # Apply filter if provided
         if filter:
            results = [
                (doc, score) for doc, score in results
                if all(
                    doc.metadata.get(key) == value
                    for key, value in filter.items()
                )
            ]

         return results

     except Exception as e:
        logger.error(f"Similarity search with score failed: {e}")
        return []
    
    def is_logically_deleted(self, metadata: Dict) -> bool:
        """Check if metadata matches any active logical-deletion filter."""
        if not metadata or not self._deleted_filters:
            return False
        
        for flt in self._deleted_filters:
            match = True
            for k, expected in flt.items():
                actual = metadata.get(k)
                if isinstance(expected, dict) and "$regex" in expected:
                    import re
                    pattern = expected["$regex"]
                    if not actual or not re.search(pattern, str(actual)):
                        match = False
                        break
                elif expected != actual:
                    # Also check source fallback if k is 'url'
                    if k == "url" and metadata.get("source") == expected:
                        continue
                    # Also check url/source if k is 'source_url'
                    if k == "source_url" and (metadata.get("url") == expected or metadata.get("source") == expected):
                        continue
                    match = False
                    break
            if match:
                return True
        return False

    def delete_by_metadata(self, filter: Dict) -> bool:
        """Logically delete documents by recording metadata filter."""
        if not self._initialized:
            self.initialize()
        
        try:
            if not filter:
                return False
            if filter not in self._deleted_filters:
                self._deleted_filters.append(filter)
                self._save_sidecar()
            logger.info(f"[RAG][DELETE] Logically deleted vectors matching filter: {filter}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        if not self._initialized:
            self.initialize()
        
        try:
            # Remove the index file and reinitialize
            if os.path.exists(self.index_path):
                import shutil
                shutil.rmtree(self.index_path, ignore_errors=True)
            
            # Reinitialize with empty index
            self.vector_store = FAISS.from_texts(
                ["Initial document"],
                self.embeddings,
                metadatas=[{"source": "init"}]
            )
            self._save_index()

            # Clear logical deletions
            self._deleted_filters = []
            if os.path.exists(self._sidecar_path):
                try:
                    os.remove(self._sidecar_path)
                except Exception:
                    pass
            
            logger.info("Cleared vector store collection")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        if not self._initialized:
            self.initialize()
        
        try:
            count = self.vector_store.index.ntotal if self.vector_store else 0
            return {
                "name": "faiss_index",
                "count": count
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"name": "faiss_index", "count": 0}


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        _vector_store.initialize()
    return _vector_store
