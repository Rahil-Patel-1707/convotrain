"""
Document upload and management API routes.
"""
import os
import uuid
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from pydantic import BaseModel
from loguru import logger

from app.database import get_mongodb, get_vector_store
from app.services.document_processor import get_document_processor, DocumentProcessor
from app.services.indexer import get_indexer_service
from app.routes.auth import require_auth


router = APIRouter(prefix="/api/documents", tags=["Documents"])


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    word_count: int
    status: str
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


@router.get("/supported-types")
async def get_supported_types():
    """Get list of supported document types."""
    processor = get_document_processor()
    return {
        "supported_extensions": processor.get_supported_types(),
        "max_file_size_mb": processor.MAX_FILE_SIZE // (1024 * 1024)
    }


@router.post("/upload/{site_id}")
async def upload_documents(
    site_id: str,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user: dict = Depends(require_auth)
):
    """
    Upload documents to a site's knowledge base.
    Supports: PDF, DOCX, TXT, MD, CSV, PPTX, XLSX
    """
    mongodb = await get_mongodb()
    processor = get_document_processor()
    
    # Verify site exists and user has access
    site = await mongodb.db.sites.find_one({"site_id": site_id})
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    
    user_id = str(user["_id"])
    if site.get("user_id") and site["user_id"] != user_id and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    results = []
    documents_to_index = []
    
    for file in files:
        # Validate file type
        if not processor.is_supported(file.filename):
            results.append({
                "filename": file.filename,
                "success": False,
                "error": f"Unsupported file type. Supported: {', '.join(processor.get_supported_types())}"
            })
            continue
        
        try:
            # Read file content
            content = await file.read()
            
            # Process document
            result = await processor.process_file(content, file.filename, file.content_type)
            
            if not result['success']:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": result['error']
                })
                continue
            
            # Generate document ID
            doc_id = str(uuid.uuid4())[:12]
            
            # Save document metadata to MongoDB
            doc_record = {
                "doc_id": doc_id,
                "site_id": site_id,
                "filename": file.filename,
                "file_type": result['file_type'],
                "word_count": result['word_count'],
                "char_count": result['char_count'],
                "metadata": result['metadata'],
                "status": "processing",
                "uploaded_at": datetime.utcnow(),
                "uploaded_by": user_id
            }
            
            await mongodb.db.documents.insert_one(doc_record)
            
            # Queue for indexing
            documents_to_index.append({
                "doc_id": doc_id,
                "site_id": site_id,
                "filename": file.filename,
                "text": result['text'],
                "metadata": result['metadata']
            })
            
            results.append({
                "filename": file.filename,
                "success": True,
                "doc_id": doc_id,
                "word_count": result['word_count']
            })
            
        except Exception as e:
            logger.error(f"Error uploading {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    # Index documents in background
    if documents_to_index:
        background_tasks.add_task(index_documents, documents_to_index, site_id)
    
    # Update site to indicate it has documents
    await mongodb.db.sites.update_one(
        {"site_id": site_id},
        {"$set": {"has_documents": True, "updated_at": datetime.utcnow()}}
    )
    
    successful = sum(1 for r in results if r.get('success'))
    
    return {
        "message": f"Uploaded {successful} of {len(files)} documents",
        "results": results,
        "total_uploaded": successful
    }


async def index_documents(documents: List[dict], site_id: str):
    """Index uploaded documents into the vector store."""
    mongodb = await get_mongodb()
    indexer = get_indexer_service()
    vector_store = get_vector_store()
    
    for doc in documents:
        try:
            # Create page-like data for indexer
            pages = [{
                "url": f"doc://{site_id}/{doc['doc_id']}",
                "title": doc['filename'],
                "content": doc['text'],
                "site_id": site_id,
                "metadata": {
                    **doc['metadata'],
                    "source_type": "document",
                    "source_name": doc['filename'],
                    "doc_id": doc['doc_id'],
                    "site_id": site_id
                }
            }]
            
            # Index the document
            stats = await indexer.index_pages(pages, site_id=site_id)
            
            # Update document status
            await mongodb.db.documents.update_one(
                {"doc_id": doc['doc_id']},
                {"$set": {
                    "status": "indexed",
                    "chunks_created": stats.get('total_chunks', 0),
                    "indexed_at": datetime.utcnow()
                }}
            )
            
            logger.info(f"Indexed document {doc['filename']} with {stats.get('total_chunks', 0)} chunks")
            
        except Exception as e:
            logger.error(f"Error indexing document {doc['filename']}: {e}")
            await mongodb.db.documents.update_one(
                {"doc_id": doc['doc_id']},
                {"$set": {"status": "error", "error": str(e)}}
            )


@router.get("/{site_id}")
async def list_documents(
    site_id: str,
    user: dict = Depends(require_auth)
):
    """Get all documents for a site."""
    mongodb = await get_mongodb()
    
    # Verify site access
    site = await mongodb.db.sites.find_one({"site_id": site_id})
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    
    user_id = str(user["_id"])
    if site.get("user_id") and site["user_id"] != user_id and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get documents
    cursor = mongodb.db.documents.find({"site_id": site_id})
    documents = await cursor.to_list(length=1000)
    
    return {
        "documents": [
            {
                "id": doc["doc_id"],
                "filename": doc["filename"],
                "file_type": doc["file_type"],
                "word_count": doc.get("word_count", 0),
                "chunks": doc.get("chunks_created", 0),
                "status": doc["status"],
                "uploaded_at": doc["uploaded_at"].isoformat() if doc.get("uploaded_at") else None
            }
            for doc in documents
        ],
        "total": len(documents)
    }


@router.delete("/{site_id}/{doc_id}")
async def delete_document(
    site_id: str,
    doc_id: str,
    user: dict = Depends(require_auth)
):
    """Delete a document from a site."""
    mongodb = await get_mongodb()
    vector_store = get_vector_store()
    
    # Verify site access
    site = await mongodb.db.sites.find_one({"site_id": site_id})
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    
    user_id = str(user["_id"])
    if site.get("user_id") and site["user_id"] != user_id and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Find document
    doc = await mongodb.db.documents.find_one({"doc_id": doc_id, "site_id": site_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from vector store
    try:
        doc_url = f"doc://{site_id}/{doc_id}"
        vector_store.delete_by_metadata({"url": doc_url})
    except Exception as e:
        logger.warning(f"Error deleting document vectors: {e}")
    
    # Delete from MongoDB
    await mongodb.db.documents.delete_one({"doc_id": doc_id})
    
    # Check if site still has documents
    remaining = await mongodb.db.documents.count_documents({"site_id": site_id})
    if remaining == 0:
        await mongodb.db.sites.update_one(
            {"site_id": site_id},
            {"$set": {"has_documents": False}}
        )
    
    return {"success": True, "message": "Document deleted"}


@router.put("/replace/{site_id}/{doc_id}")
async def replace_document(
    site_id: str,
    doc_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(require_auth)
):
    """
    Replace an existing document with a new file.
    Safely invalidates old FAISS vectors and immediately re-indexes the new content.
    """
    mongodb = await get_mongodb()
    processor = get_document_processor()
    vector_store = get_vector_store()
    indexer = get_indexer_service()

    # 1. Verify site exists and user has access
    site = await mongodb.db.sites.find_one({"site_id": site_id})
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    user_id = str(user["_id"])
    if site.get("user_id") and site["user_id"] != user_id and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied")

    # 2. Find old document
    old_doc = await mongodb.db.documents.find_one({"doc_id": doc_id, "site_id": site_id})
    if not old_doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # 3. Validate new file format
    if not processor.is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(processor.get_supported_types())}"
        )

    # 4. Process new file content
    content = await file.read()
    result = await processor.process_file(content, file.filename, file.content_type)
    if not result['success']:
        raise HTTPException(status_code=400, detail=result.get('error', 'Failed to process document'))

    # 5. Invalidate old vectors from FAISS using logical deletion sidecar
    old_doc_url = f"doc://{site_id}/{doc_id}"
    try:
        vector_store.delete_by_metadata({"url": old_doc_url})
        logger.info(f"[RAG][REPLACE] Invalidated old vectors for {old_doc_url}")
    except Exception as e:
        logger.warning(f"Error invalidating old document vectors: {e}")

    # 6. Generate fresh doc_id for new content so logical deletion on old_doc_url does not filter new vectors
    new_doc_id = str(uuid.uuid4())[:12]
    new_doc_url = f"doc://{site_id}/{new_doc_id}"

    pages = [{
        "url": new_doc_url,
        "title": file.filename,
        "content": result['text'],
        "site_id": site_id,
        "metadata": {
            **result['metadata'],
            "source_type": "document",
            "source_name": file.filename,
            "doc_id": new_doc_id,
            "site_id": site_id
        }
    }]

    stats = await indexer.index_pages(pages, site_id=site_id)

    # 7. Update document record in MongoDB with new doc_id and metadata
    await mongodb.db.documents.update_one(
        {"doc_id": doc_id, "site_id": site_id},
        {"$set": {
            "doc_id": new_doc_id,
            "filename": file.filename,
            "file_type": result['file_type'],
            "word_count": result['word_count'],
            "char_count": result['char_count'],
            "metadata": result['metadata'],
            "status": "indexed",
            "chunks_created": stats.get('total_chunks', 0),
            "updated_at": datetime.utcnow(),
            "indexed_at": datetime.utcnow()
        }}
    )

    return {
        "success": True,
        "message": f"Document replaced and re-indexed with {stats.get('total_chunks', 0)} chunks",
        "doc_id": new_doc_id,
        "old_doc_id": doc_id,
        "filename": file.filename,
        "word_count": result['word_count'],
        "chunks": stats.get('total_chunks', 0)
    }


class DocumentSiteSetupRequest(BaseModel):
    """Request to create a document-only knowledge base."""
    name: str


@router.post("/setup")
async def setup_document_site(
    request: DocumentSiteSetupRequest,
    user: dict = Depends(require_auth)
):
    """Create a document-only knowledge base without starting a web crawl."""
    name = request.name.strip()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base name is required"
        )

    mongodb = await get_mongodb()
    user_id = str(user["_id"])

    # Generate a unique site ID without relying on a website URL.
    base_name = "".join(
        char for char in name.lower()
        if char.isalnum()
    )[:6] or "docs"

    site_id = f"{base_name}{uuid.uuid4().hex[:6]}"

    await mongodb.create_site({
        "site_id": site_id,
        "url": f"document://{site_id}",
        "name": name,
        "user_id": user_id,
        "status": "ready",
        "has_documents": False,
        "config": {
            "security": {
                "allowed_domains": [],
                "enforce_domain_validation": False,
                "require_referrer": False,
                "rate_limit_per_session": 60
            }
        }
    })

    logger.info(
        f"Created document-only knowledge base: {site_id}"
    )

    return {
        "site_id": site_id,
        "status": "ready",
        "message": f"Knowledge base '{name}' created successfully"
    }
