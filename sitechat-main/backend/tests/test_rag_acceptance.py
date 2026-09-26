"""
End-to-end acceptance tests for the ConvoTrain AI RAG pipeline:
TEST 1 — Local document ingestion, indexing, and question answering
TEST 2 — Website crawling, indexing, and question answering
TEST 3 — Mixed knowledge (both uploaded document and crawled website on one site)
TEST 4 — Unknown information (controlled fallback, zero hallucination)
TEST 5 — Multi-tenant isolation (Site A vs Site B)
"""
import asyncio
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import uuid

# Ensure backend root is on path and utf-8 output on Windows
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)
os.chdir(backend_dir)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from loguru import logger
from app.database import get_mongodb, get_vector_store
from app.services.rag_engine import get_rag_engine
from app.services.document_processor import get_document_processor
from app.services.crawler import CrawlerService
from app.services.indexer import get_indexer_service
from app.routes.documents import index_documents


# =========================================================================
# Mock Local Web Server for Web Crawling Tests
# =========================================================================

HTML_PAGES = {
    "/spicegarden/delivery": """
<!DOCTYPE html>
<html>
<head><title>Spice Garden Delivery Information</title></head>
<body>
  <h1>Delivery Services</h1>
  <p>Spice Garden delivers within 8 km of the restaurant.</p>
  <p>Our drivers ensure fast, hot, and contactless delivery for every order.</p>
</body>
</html>
""",
    "/spicegarden/delivery-hours": """
<!DOCTYPE html>
<html>
<head><title>Spice Garden Delivery Hours</title></head>
<body>
  <h1>Delivery Timing</h1>
  <p>Home delivery is available from 12 PM to 10 PM every day.</p>
  <p>Late night orders must be placed before 9:30 PM.</p>
</body>
</html>
"""
}


class MockHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        content = HTML_PAGES.get(self.path)
        if content:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logging noise


def start_mock_server(port=8899):
    server = HTTPServer(("127.0.0.1", port), MockHTTPHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# =========================================================================
# Acceptance Tests
# =========================================================================

async def run_acceptance_tests():
    print("\n========================================================")
    print("CONVOTRAIN AI — RAG PIPELINE ACCEPTANCE TEST SUITE")
    print("========================================================\n")

    # Start mock server
    server = start_mock_server(8899)
    print("Mock HTTP server started on http://127.0.0.1:8899")

    mongodb = await get_mongodb()
    rag_engine = get_rag_engine()
    doc_processor = get_document_processor()
    indexer = get_indexer_service()

    test_results = {}

    # ---------------------------------------------------------------------
    # Setup Test Tenant Sites
    # ---------------------------------------------------------------------
    site_main_id = f"spice_garden_{uuid.uuid4().hex[:6]}"
    site_a_id = f"site_a_{uuid.uuid4().hex[:6]}"
    site_b_id = f"site_b_{uuid.uuid4().hex[:6]}"

    await mongodb.create_site({
        "site_id": site_main_id,
        "name": "Spice Garden Restaurant",
        "url": "http://127.0.0.1:8899/spicegarden",
        "status": "ready"
    })
    await mongodb.create_site({
        "site_id": site_a_id,
        "name": "Site A Bistro",
        "url": f"document://{site_a_id}",
        "status": "ready"
    })
    await mongodb.create_site({
        "site_id": site_b_id,
        "name": "Site B Lounge",
        "url": f"document://{site_b_id}",
        "status": "ready"
    })

    print(f"Created Test Sites: {site_main_id}, {site_a_id}, {site_b_id}\n")

    # =====================================================================
    # TEST 1 — LOCAL DOCUMENT
    # =====================================================================
    print("--- TEST 1: LOCAL DOCUMENT UPLOAD & QA ---")
    doc_text_1 = "Spice Garden Restaurant is open Monday to Sunday from 11:00 AM to 11:00 PM."
    processed = await doc_processor.process_file(
        file_content=doc_text_1.encode("utf-8"),
        filename="opening_hours.txt"
    )
    doc_id_1 = str(uuid.uuid4())[:12]
    await mongodb.db.documents.insert_one({
        "doc_id": doc_id_1,
        "site_id": site_main_id,
        "filename": "opening_hours.txt",
        "file_type": ".txt",
        "word_count": processed["word_count"],
        "status": "processing"
    })
    # Index document
    await index_documents([{
        "doc_id": doc_id_1,
        "site_id": site_main_id,
        "filename": "opening_hours.txt",
        "text": processed["text"],
        "metadata": processed["metadata"]
    }], site_id=site_main_id)

    # Test standalone retrieval first
    debug_res_1 = await rag_engine.debug_retrieve("What time does Spice Garden open?", site_id=site_main_id)
    print(f"Test 1 Retrieval Results: {len(debug_res_1)} chunks found")
    assert len(debug_res_1) > 0, "TEST 1 FAILED: Retrieval returned 0 chunks!"
    assert "11:00 AM" in debug_res_1[0]["text"], "TEST 1 FAILED: Retrieved chunk does not contain expected text!"

    # Test full chat with Groq
    chat_res_1 = await rag_engine.chat(
        message="What time does Spice Garden open?",
        session_id=str(uuid.uuid4()),
        site_id=site_main_id
    )
    print(f"Question: 'What time does Spice Garden open?'")
    print(f"Answer: {chat_res_1.answer}")
    print(f"Sources: {[s.url for s in chat_res_1.sources]}")
    
    ans_1_lower = chat_res_1.answer.lower()
    assert ("11" in ans_1_lower and ("am" in ans_1_lower or "11:00" in ans_1_lower)), (
        f"TEST 1 FAILED: Groq answer does not mention 11 AM! Answer: {chat_res_1.answer}"
    )
    assert len(chat_res_1.sources) > 0, "TEST 1 FAILED: No sources returned in ChatResponse!"
    print(">>> TEST 1 PASSED! Local document indexed, retrieved, and answered accurately.\n")
    test_results["TEST 1 (Local Document)"] = "PASSED"

    # =====================================================================
    # TEST 2 — WEBSITE DATA
    # =====================================================================
    print("--- TEST 2: WEBSITE CRAWLING & QA ---")
    crawler = CrawlerService()
    crawled_pages = await crawler.crawl(
        start_url="http://127.0.0.1:8899/spicegarden/delivery",
        max_pages=2
    )
    print(f"Crawled {len(crawled_pages)} pages from http://127.0.0.1:8899/spicegarden/delivery")
    assert len(crawled_pages) > 0, "TEST 2 FAILED: Crawler fetched 0 pages!"
    
    # Index crawled pages
    await indexer.index_pages(crawled_pages, site_id=site_main_id)

    # Test standalone retrieval
    debug_res_2 = await rag_engine.debug_retrieve("How far does Spice Garden deliver?", site_id=site_main_id)
    print(f"Test 2 Retrieval Results: {len(debug_res_2)} chunks found")
    assert len(debug_res_2) > 0, "TEST 2 FAILED: Retrieval returned 0 chunks for crawled page!"
    assert "8 km" in debug_res_2[0]["text"], "TEST 2 FAILED: Chunk missing '8 km'!"

    # Test full chat with Groq
    chat_res_2 = await rag_engine.chat(
        message="How far does Spice Garden deliver?",
        session_id=str(uuid.uuid4()),
        site_id=site_main_id
    )
    print(f"Question: 'How far does Spice Garden deliver?'")
    print(f"Answer: {chat_res_2.answer}")
    print(f"Sources: {[s.url for s in chat_res_2.sources]}")

    assert "8" in chat_res_2.answer and ("km" in chat_res_2.answer.lower() or "kilometer" in chat_res_2.answer.lower()), (
        f"TEST 2 FAILED: Groq answer does not mention 8 km! Answer: {chat_res_2.answer}"
    )
    print(">>> TEST 2 PASSED! Website crawled, indexed, retrieved, and answered accurately.\n")
    test_results["TEST 2 (Website Crawl)"] = "PASSED"

    # =====================================================================
    # TEST 3 — MIXED KNOWLEDGE
    # =====================================================================
    print("--- TEST 3: MIXED KNOWLEDGE (UPLOAD + WEBSITE ON SAME SITE) ---")
    # Upload 2: Jain meals
    doc_text_3 = "Jain meals are available on request at Spice Garden."
    proc_3 = await doc_processor.process_file(doc_text_3.encode("utf-8"), "jain_meals.txt")
    doc_id_3 = str(uuid.uuid4())[:12]
    await index_documents([{
        "doc_id": doc_id_3,
        "site_id": site_main_id,
        "filename": "jain_meals.txt",
        "text": proc_3["text"],
        "metadata": proc_3["metadata"]
    }], site_id=site_main_id)

    # Webpage 2: Home delivery hours
    crawled_pages_3 = await crawler.crawl(
        start_url="http://127.0.0.1:8899/spicegarden/delivery-hours",
        max_pages=1
    )
    await indexer.index_pages(crawled_pages_3, site_id=site_main_id)

    # Ask Q1 (from document)
    chat_3a = await rag_engine.chat(
        message="Are Jain meals available?",
        session_id=str(uuid.uuid4()),
        site_id=site_main_id
    )
    print(f"Q1 (Doc target): 'Are Jain meals available?'")
    print(f"A1: {chat_3a.answer}")
    print(f"Sources 1: {[s.url for s in chat_3a.sources]}")
    assert any("jain" in s.url.lower() or "jain" in s.title.lower() or "doc://" in s.url for s in chat_3a.sources), (
        "TEST 3 FAILED: Expected uploaded document as source for Jain meals!"
    )
    assert "jain" in chat_3a.answer.lower() and ("request" in chat_3a.answer.lower() or "available" in chat_3a.answer.lower())

    # Ask Q2 (from website)
    chat_3b = await rag_engine.chat(
        message="What are the home delivery hours?",
        session_id=str(uuid.uuid4()),
        site_id=site_main_id
    )
    print(f"Q2 (Web target): 'What are the home delivery hours?'")
    print(f"A2: {chat_3b.answer}")
    print(f"Sources 2: {[s.url for s in chat_3b.sources]}")
    assert any("http://127.0.0.1:8899" in s.url for s in chat_3b.sources), (
        "TEST 3 FAILED: Expected website URL as source for delivery hours!"
    )
    assert ("12" in chat_3b.answer and "10" in chat_3b.answer), (
        f"TEST 3 FAILED: Answer missing 12 PM - 10 PM! Answer: {chat_3b.answer}"
    )
    print(">>> TEST 3 PASSED! Mixed knowledge correctly retrieves the appropriate source for each question.\n")
    test_results["TEST 3 (Mixed Knowledge)"] = "PASSED"

    # =====================================================================
    # TEST 4 — UNKNOWN INFORMATION
    # =====================================================================
    print("--- TEST 4: UNKNOWN INFORMATION (NO HALLUCINATION) ---")
    chat_res_4 = await rag_engine.chat(
        message="What is Spice Garden's annual revenue?",
        session_id=str(uuid.uuid4()),
        site_id=site_main_id
    )
    print(f"Question: 'What is Spice Garden's annual revenue?'")
    print(f"Answer: {chat_res_4.answer}")
    print(f"Sources: {[s.url for s in chat_res_4.sources]}")

    ans_4_lower = chat_res_4.answer.lower().replace("’", "'")
    # Check that assistant does not invent an answer and expresses lack of information
    assert ("couldn't find" in ans_4_lower or 
            "don't have" in ans_4_lower or 
            "dont have" in ans_4_lower or
            "not have" in ans_4_lower or 
            "not available" in ans_4_lower or
            "no information" in ans_4_lower), (
        f"TEST 4 FAILED: Model fabricated or did not express lack of knowledge! Answer: {chat_res_4.answer}"
    )
    print(">>> TEST 4 PASSED! Model correctly refused to hallucinate unknown revenue information.\n")
    test_results["TEST 4 (Unknown Info / No Hallucination)"] = "PASSED"

    # =====================================================================
    # TEST 5 — TENANT ISOLATION
    # =====================================================================
    print("--- TEST 5: MULTI-TENANT ISOLATION (SITE A vs SITE B) ---")
    doc_a = "Site A closes at 9 PM every night without exception."
    proc_a = await doc_processor.process_file(doc_a.encode("utf-8"), "site_a_rules.txt")
    doc_id_a = str(uuid.uuid4())[:12]
    await index_documents([{
        "doc_id": doc_id_a,
        "site_id": site_a_id,
        "filename": "site_a_rules.txt",
        "text": proc_a["text"],
        "metadata": proc_a["metadata"]
    }], site_id=site_a_id)

    doc_b = "Site B closes at 11 PM every night without exception."
    proc_b = await doc_processor.process_file(doc_b.encode("utf-8"), "site_b_rules.txt")
    doc_id_b = str(uuid.uuid4())[:12]
    await index_documents([{
        "doc_id": doc_id_b,
        "site_id": site_b_id,
        "filename": "site_b_rules.txt",
        "text": proc_b["text"],
        "metadata": proc_b["metadata"]
    }], site_id=site_b_id)

    # Ask Site A
    chat_a = await rag_engine.chat(
        message="When does the restaurant close?",
        session_id=str(uuid.uuid4()),
        site_id=site_a_id
    )
    print(f"Site A Answer: {chat_a.answer}")
    assert "9" in chat_a.answer, f"TEST 5 FAILED: Site A should say 9 PM! Got: {chat_a.answer}"
    assert "11" not in chat_a.answer, f"TEST 5 FAILED: Site A leaked 11 PM from Site B! Got: {chat_a.answer}"

    # Ask Site B
    chat_b = await rag_engine.chat(
        message="When does the restaurant close?",
        session_id=str(uuid.uuid4()),
        site_id=site_b_id
    )
    print(f"Site B Answer: {chat_b.answer}")
    assert "11" in chat_b.answer, f"TEST 5 FAILED: Site B should say 11 PM! Got: {chat_b.answer}"
    assert "9" not in chat_b.answer, f"TEST 5 FAILED: Site B leaked 9 PM from Site A! Got: {chat_b.answer}"

    print(">>> TEST 5 PASSED! Site A and Site B are strictly isolated with zero cross-talk.\n")
    test_results["TEST 5 (Tenant Isolation)"] = "PASSED"

    # =====================================================================
    # Clean up test sites from DB
    # =====================================================================
    await mongodb.db.sites.delete_many({"site_id": {"$in": [site_main_id, site_a_id, site_b_id]}})
    await mongodb.db.documents.delete_many({"site_id": {"$in": [site_main_id, site_a_id, site_b_id]}})

    print("========================================================")
    print("FINAL SUMMARY OF ALL ACCEPTANCE TESTS:")
    print("========================================================")
    all_passed = True
    for test_name, status in test_results.items():
        print(f"  {test_name:<40} : {status}")
        if status != "PASSED":
            all_passed = False

    print("========================================================")
    if all_passed:
        print("\nALL 5 ACCEPTANCE TESTS PASSED SUCCESSFULLY! ✅\n")
    else:
        print("\nSOME TESTS FAILED! ❌\n")


if __name__ == "__main__":
    asyncio.run(run_acceptance_tests())
