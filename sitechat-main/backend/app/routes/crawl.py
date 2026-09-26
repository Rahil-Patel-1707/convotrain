"""
Crawl API routes.
"""
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from loguru import logger

from app.models.schemas import CrawlRequest, CrawlResponse, CrawlStatus, PageInfo
from app.services.crawler import CrawlerService
from app.services.indexer import IndexerService
from app.database import get_mongodb, get_vector_store
from app.config import settings
from app.routes.auth import require_auth, require_admin
from app.core.site_access import can_manage_site, is_admin

router = APIRouter(prefix="/api/crawl", tags=["Crawl"])


async def _crawl_and_index(
    job_id: str,
    url: str,
    max_pages: int,
    include_patterns: list,
    exclude_patterns: list
):
    """Background task for crawling and indexing."""
    mongodb = await get_mongodb()
    
    try:
        # Crawl
        crawler = CrawlerService()
        pages = await crawler.crawl(
            start_url=url,
            max_pages=max_pages,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            job_id=job_id
        )
        
        if not pages:
            await mongodb.update_crawl_job(job_id, status="failed", error="No pages found")
            return
        
        # Index with site_id if site exists for this URL
        site = await mongodb.db.sites.find_one({"url": url}) or await mongodb.db.sites.find_one({"url": url.rstrip("/")})
        site_id = site.get("site_id") if site else None

        indexer = IndexerService()
        stats = await indexer.index_pages(pages, job_id=job_id, site_id=site_id)
        
        # Update job status
        await mongodb.update_crawl_job(
            job_id,
            status="completed",
            pages_crawled=len(pages),
            pages_indexed=stats["indexed_pages"]
        )
        
        logger.info(f"Crawl job {job_id} completed: {stats}")
        
    except Exception as e:
        logger.error(f"Crawl job {job_id} failed: {e}")
        await mongodb.update_crawl_job(job_id, status="failed", error=str(e))


@router.post("", response_model=CrawlResponse)
async def start_crawl(
    body: CrawlRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_auth)
):
    """
    Start a crawl job to index a website.
    
    - **url**: The URL to start crawling from
    - **max_pages**: Maximum number of pages to crawl
    - **include_patterns**: URL patterns to include (regex)
    - **exclude_patterns**: URL patterns to exclude (regex)
    
    Returns a job ID that can be used to check status.
    """
    try:
        mongodb = await get_mongodb()
        
        # Verify site ownership if site already exists for this URL
        clean_url = body.url.rstrip("/")
        site = await mongodb.db.sites.find_one({"url": body.url}) or await mongodb.db.sites.find_one({"url": clean_url})
        if site and not can_manage_site(user, site):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create job
        job_id = await mongodb.create_crawl_job(body.url)
        
        # Start background crawl
        background_tasks.add_task(
            _crawl_and_index,
            job_id,
            body.url,
            body.max_pages,
            body.include_patterns,
            body.exclude_patterns
        )
        
        return CrawlResponse(
            job_id=job_id,
            message=f"Crawl job started for {body.url}",
            status="running"
        )
        
    except Exception as e:
        logger.error(f"Start crawl error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=CrawlStatus)
async def get_crawl_status(job_id: str):
    """
    Get the status of a crawl job.
    
    - **job_id**: The crawl job ID
    """
    try:
        mongodb = await get_mongodb()
        job = await mongodb.get_crawl_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return CrawlStatus(
            job_id=job_id,
            status=job["status"],
            pages_crawled=job.get("pages_crawled", 0),
            pages_indexed=job.get("pages_indexed", 0),
            errors=job.get("errors", []),
            started_at=job["created_at"],
            completed_at=job.get("updated_at") if job["status"] == "completed" else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest", response_model=CrawlStatus)
async def get_latest_crawl():
    """Get the latest crawl job status."""
    try:
        mongodb = await get_mongodb()
        job = await mongodb.get_latest_crawl_job()
        
        if not job:
            raise HTTPException(status_code=404, detail="No crawl jobs found")
        
        return CrawlStatus(
            job_id=str(job["_id"]),
            status=job["status"],
            pages_crawled=job.get("pages_crawled", 0),
            pages_indexed=job.get("pages_indexed", 0),
            errors=job.get("errors", []),
            started_at=job["created_at"],
            completed_at=job.get("updated_at") if job["status"] == "completed" else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get latest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex")
async def reindex_all(
    background_tasks: BackgroundTasks,
    admin: dict = Depends(require_admin)
):
    """
    Re-index all existing pages (admin only).
    
    Useful when you want to regenerate embeddings or update chunking.
    """
    try:
        async def _reindex():
            indexer = IndexerService()
            await indexer.reindex_all()
        
        background_tasks.add_task(_reindex)
        
        return {"message": "Reindexing started", "status": "running"}
        
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pages", response_model=list)
async def get_pages(user: dict = Depends(require_auth)):
    """Get all indexed pages accessible to the current user."""
    try:
        mongodb = await get_mongodb()
        pages = await mongodb.get_all_pages(status="indexed")
        
        # Non-admins only see pages belonging to their sites
        if not is_admin(user):
            sites = await mongodb.list_sites(user_id=str(user["_id"]))
            user_sids = {s["site_id"] for s in sites if s.get("site_id")}
            user_urls = [s["url"].rstrip("/") for s in sites if s.get("url")]
            filtered = []
            for p in pages:
                pmeta = p.get("metadata", {})
                psid = pmeta.get("site_id")
                purl = p.get("url", "")
                if psid and psid in user_sids:
                    filtered.append(p)
                elif any(purl.startswith(u) for u in user_urls if u and not u.startswith("document://")):
                    filtered.append(p)
            pages = filtered
        
        return [
            PageInfo(
                url=p["url"],
                title=p.get("title", ""),
                chunk_count=p.get("chunk_count", 0),
                last_crawled=p.get("last_crawled"),
                status=p.get("status", "unknown")
            )
            for p in pages
        ]
        
    except Exception as e:
        logger.error(f"Get pages error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pages/{url:path}")
async def delete_page(url: str, admin: dict = Depends(require_admin)):
    """
    Delete a page from the index (admin only).
    
    - **url**: The URL of the page to delete
    """
    try:
        indexer = IndexerService()
        result = await indexer.delete_page_index(url)
        
        return {"success": result, "message": f"Page {url} deleted"}
        
    except Exception as e:
        logger.error(f"Delete page error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
