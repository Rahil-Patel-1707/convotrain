"""
Phase 4 Cross-Tenant Security and Isolation Test Suite.

Verifies:
1. Owner A cannot retrieve Owner B's site.
2. Owner A cannot upload a document to Owner B's site.
3. Owner A cannot delete Owner B's site.
4. Owner A cannot access Owner B's conversations.
5. Owner A cannot access Owner B's leads.
6. Owner A cannot access Owner B's Q&A.
7. Owner A cannot access Owner B's triggers.
8. Site A RAG query returns only Site A chunks.
9. Site B RAG query returns only Site B chunks.
10. A document uploaded to Site A cannot appear in Site B retrieval.
11. Deleting Site A prevents Site A content from being returned by RAG.
12. Public Site A chatbot cannot retrieve Site B content.
13. Changing site_id in a public request cannot expose another site's private content.
14. Phase 4K: Exact dual-restaurant RAG context verification (Spice Garden vs Blue Ocean).
"""
import uuid
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport
from datetime import datetime
from langchain_core.documents import Document

from app.providers.database import MockDatabaseProvider
from app.services.rag_engine import RAGEngine
from app.database.vector_store import VectorStore


# =====================================================================
# Fixtures for Multi-Tenant Testing
# =====================================================================

@pytest.fixture
def owner_a():
    return {
        "_id": "user_id_owner_a",
        "user_id": "user_id_owner_a",
        "email": "owner_a@example.com",
        "name": "Owner A",
        "role": "user"
    }


@pytest.fixture
def owner_b():
    return {
        "_id": "user_id_owner_b",
        "user_id": "user_id_owner_b",
        "email": "owner_b@example.com",
        "name": "Owner B",
        "role": "user"
    }


@pytest.fixture
def site_a():
    return {
        "site_id": "site_a_123",
        "name": "Spice Garden Restaurant",
        "url": "https://spicegarden.example.com",
        "status": "ready",
        "user_id": "user_id_owner_a",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


@pytest.fixture
def site_b():
    return {
        "site_id": "site_b_456",
        "name": "Blue Ocean Restaurant",
        "url": "https://blueocean.example.com",
        "status": "ready",
        "user_id": "user_id_owner_b",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


@pytest.fixture
async def multi_tenant_setup(owner_a, owner_b, site_a, site_b):
    """Setup mock database with two tenants and their respective sites."""
    db = MockDatabaseProvider()
    db.seed_user(owner_a)
    db.seed_user(owner_b)
    db.seed_site(site_a)
    db.seed_site(site_b)

    # In-memory store for leads and QA
    db._leads = {}
    async def mock_save_lead(lead_data):
        lid = str(uuid.uuid4())[:8]
        lead = {
            **lead_data,
            "id": lid,
            "lead_id": lid,
            "captured_at": datetime.utcnow()
        }
        db._leads[lid] = lead
        return lead

    async def mock_get_leads(site_id, page=1, limit=20, search=None):
        matching = [l for l in db._leads.values() if l.get("site_id") == site_id]
        return matching, len(matching)

    async def mock_get_lead_by_id(lead_id):
        return db._leads.get(lead_id)

    async def mock_delete_lead(lead_id):
        if lead_id in db._leads:
            del db._leads[lead_id]
            return True
        return False

    db.save_lead = mock_save_lead
    db.get_leads = mock_get_leads
    db.get_lead_by_id = mock_get_lead_by_id
    db.delete_lead = mock_delete_lead

    db._qa_pairs = {}
    async def mock_create_qa_pair(qa_data):
        qid = str(uuid.uuid4())[:8]
        qa = {
            **qa_data,
            "id": qid,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        db._qa_pairs[qid] = qa
        return qa

    async def mock_get_qa_pairs(site_id, page=1, limit=20, search=None, enabled_only=False):
        matching = [q for q in db._qa_pairs.values() if q.get("site_id") == site_id]
        return matching, len(matching)

    async def mock_get_qa_pair(qa_id):
        return db._qa_pairs.get(qa_id)

    async def mock_update_qa_pair(qa_id, updates):
        qa = db._qa_pairs.get(qa_id)
        if qa:
            qa.update(updates)
            return qa
        return None

    async def mock_delete_qa_pair(qa_id):
        if qa_id in db._qa_pairs:
            del db._qa_pairs[qa_id]
            return True
        return False

    async def mock_get_qa_for_rag(site_id):
        return [q for q in db._qa_pairs.values() if q.get("site_id") == site_id]

    db.create_qa_pair = mock_create_qa_pair
    db.get_qa_pairs = mock_get_qa_pairs
    db.get_qa_pair = mock_get_qa_pair
    db.update_qa_pair = mock_update_qa_pair
    db.delete_qa_pair = mock_delete_qa_pair
    db.get_qa_for_rag = mock_get_qa_for_rag

    # Mock collection for direct mongodb.db access used by some endpoints
    class MockCol:
        def __init__(self, store):
            self._store = store

        async def find_one(self, filter_dict):
            for item in self._store.values():
                if all(item.get(k) == v for k, v in filter_dict.items()):
                    return item
                if "_id" in filter_dict and item.get("site_id") == filter_dict["_id"]:
                    return item
            return None

        async def insert_one(self, doc):
            key = doc.get("doc_id") or doc.get("id") or str(len(self._store))
            self._store[key] = doc
            return MagicMock(inserted_id=key)

        async def update_one(self, filter_dict, update_dict):
            item = await self.find_one(filter_dict)
            if item and "$set" in update_dict:
                item.update(update_dict["$set"])
            return MagicMock(modified_count=1 if item else 0)

        async def delete_one(self, filter_dict):
            item = await self.find_one(filter_dict)
            if item:
                key = item.get("doc_id") or item.get("id")
                if key in self._store:
                    del self._store[key]
            return MagicMock(deleted_count=1 if item else 0)

        def find(self, filter_dict):
            results = []
            for item in self._store.values():
                if all(item.get(k) == v for k, v in filter_dict.items()):
                    results.append(item)
            mock_cursor = MagicMock()
            async def to_list(length=1000):
                return results[:length]
            mock_cursor.to_list = to_list
            return mock_cursor

        async def count_documents(self, filter_dict):
            count = 0
            for item in self._store.values():
                if all(item.get(k) == v for k, v in filter_dict.items()):
                    count += 1
            return count

    db.db = MagicMock()
    db.db.sites = MockCol(db._sites)
    db.db.documents = MockCol(db._documents)
    db.db.conversations = MockCol(db._conversations)

    return db


# =====================================================================
# API Client Helpers
# =====================================================================

def make_client_for_user(fastapi_app, user_obj):
    """Creates an AsyncClient where require_auth returns the given user."""
    from app.routes.auth import require_auth
    fastapi_app.dependency_overrides[require_auth] = lambda: user_obj
    transport = ASGITransport(app=fastapi_app)
    return AsyncClient(transport=transport, base_url="http://test")


def make_public_client(fastapi_app):
    """Creates an AsyncClient without any auth override (anonymous)."""
    fastapi_app.dependency_overrides.clear()
    transport = ASGITransport(app=fastapi_app)
    return AsyncClient(transport=transport, base_url="http://test")


# =====================================================================
# Tests 1-7: Authorization Boundaries Across Owners
# =====================================================================

@pytest.mark.asyncio
async def test_01_owner_a_cannot_retrieve_site_b(multi_tenant_setup, owner_a, site_b):
    """TEST 1: Owner A cannot retrieve Site B."""
    db = multi_tenant_setup
    with patch("app.routes.sites.get_mongodb", AsyncMock(return_value=db)):
        from app.main import app
        async with make_client_for_user(app, owner_a) as client:
            resp = await client.get(f"/api/sites/{site_b['site_id']}")
            assert resp.status_code == 403


@pytest.mark.asyncio
async def test_02_owner_a_cannot_upload_document_to_site_b(multi_tenant_setup, owner_a, site_b):
    """TEST 2: Owner A cannot upload a document to Owner B's site."""
    db = multi_tenant_setup
    with patch("app.routes.documents.get_mongodb", AsyncMock(return_value=db)):
        from app.main import app
        async with make_client_for_user(app, owner_a) as client:
            files = [("files", ("test.txt", b"Secret data", "text/plain"))]
            resp = await client.post(f"/api/documents/upload/{site_b['site_id']}", files=files)
            assert resp.status_code == 403


@pytest.mark.asyncio
async def test_03_owner_a_cannot_delete_site_b(multi_tenant_setup, owner_a, site_b):
    """TEST 3: Owner A cannot delete Owner B's site."""
    db = multi_tenant_setup
    with patch("app.routes.sites.get_mongodb", AsyncMock(return_value=db)), \
         patch("app.routes.sites.get_vector_store") as mock_vs:
        from app.main import app
        async with make_client_for_user(app, owner_a) as client:
            resp = await client.delete(f"/api/sites/{site_b['site_id']}")
            assert resp.status_code == 403
            # Ensure site B is still in DB
            assert site_b["site_id"] in db._sites


@pytest.mark.asyncio
async def test_04_owner_a_cannot_access_owner_b_conversations(multi_tenant_setup, owner_a, owner_b, site_b):
    """TEST 4: Owner A cannot access Owner B's conversations."""
    db = multi_tenant_setup
    session_id_b = "session_b_789"
    db.seed_conversation({
        "session_id": session_id_b,
        "site_id": site_b["site_id"],
        "messages": [{"role": "user", "content": "Private question for Site B"}],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })

    with patch("app.routes.conversations.get_mongodb", AsyncMock(return_value=db)):
        from app.main import app
        async with make_client_for_user(app, owner_a) as client:
            # Attempt to read Site B's conversation
            resp = await client.get(f"/api/conversations/{session_id_b}")
            assert resp.status_code == 403

            # Attempt to filter conversations by Site B
            resp_list = await client.get(f"/api/conversations?site_id={site_b['site_id']}")
            assert resp_list.status_code == 403


@pytest.mark.asyncio
async def test_05_owner_a_cannot_access_owner_b_leads(multi_tenant_setup, owner_a, site_b):
    """TEST 5: Owner A cannot access Owner B's leads."""
    db = multi_tenant_setup
    lead_b = await db.save_lead({
        "site_id": site_b["site_id"],
        "session_id": "sess_lead_b",
        "email": "lead_b@customer.com",
        "name": "Customer B",
        "source": "chat"
    })

    with patch("app.routes.leads.get_mongodb", AsyncMock(return_value=db)):
        from app.main import app
        async with make_client_for_user(app, owner_a) as client:
            # Attempt to list Site B leads
            resp = await client.get(f"/api/sites/{site_b['site_id']}/leads")
            assert resp.status_code == 403

            # Attempt to delete Site B lead
            resp_del = await client.delete(f"/api/leads/{lead_b['id']}")
            assert resp_del.status_code == 403


@pytest.mark.asyncio
async def test_06_owner_a_cannot_access_owner_b_qa(multi_tenant_setup, owner_a, site_b):
    """TEST 6: Owner A cannot access Owner B's Q&A."""
    db = multi_tenant_setup
    qa_b = await db.create_qa_pair({
        "site_id": site_b["site_id"],
        "question": "What are your secret ingredients?",
        "answer": "Seafood special recipe",
        "created_by": "user_id_owner_b"
    })

    with patch("app.routes.qa.get_mongodb", AsyncMock(return_value=db)):
        from app.main import app
        async with make_client_for_user(app, owner_a) as client:
            # Cannot list Site B QA
            resp_list = await client.get(f"/api/sites/{site_b['site_id']}/qa")
            assert resp_list.status_code == 403

            # Cannot create QA on Site B
            resp_create = await client.post(
                f"/api/sites/{site_b['site_id']}/qa",
                json={"question": "Injected Q", "answer": "Injected A"}
            )
            assert resp_create.status_code == 403

            # Cannot update QA on Site B
            resp_update = await client.put(
                f"/api/sites/{site_b['site_id']}/qa/{qa_b['id']}",
                json={"question": "Tampered Q"}
            )
            assert resp_update.status_code == 403

            # Cannot delete QA on Site B
            resp_del = await client.delete(f"/api/sites/{site_b['site_id']}/qa/{qa_b['id']}")
            assert resp_del.status_code == 403


@pytest.mark.asyncio
async def test_07_owner_a_cannot_access_owner_b_triggers(multi_tenant_setup, owner_a, site_b):
    """TEST 7: Owner A cannot access Owner B's triggers."""
    db = multi_tenant_setup
    trigger_b = await db.save_trigger(site_b["site_id"], {
        "name": "Site B Discount",
        "enabled": True,
        "priority": 10,
        "message": "Get 20% off seafood!"
    })

    with patch("app.routes.triggers.get_mongodb", AsyncMock(return_value=db)):
        from app.main import app
        async with make_client_for_user(app, owner_a) as client:
            # Cannot get Site B triggers
            resp = await client.get(f"/api/sites/{site_b['site_id']}/triggers")
            assert resp.status_code == 403

            # Cannot create trigger on Site B
            resp_create = await client.post(
                f"/api/sites/{site_b['site_id']}/triggers",
                json={
                    "name": "Owner A Injected",
                    "message": "Spam",
                    "enabled": True,
                    "priority": 1,
                    "conditions": [{"type": "time", "value": 15, "operator": "gte"}]
                }
            )
            assert resp_create.status_code == 403

            # Cannot delete trigger on Site B
            resp_del = await client.delete(f"/api/sites/{site_b['site_id']}/triggers/{trigger_b['id']}")
            assert resp_del.status_code == 403


# =====================================================================
# Tests 8-13 & 4K: RAG Vector Isolation & Public Chat Safety
# =====================================================================

@pytest.mark.asyncio
async def test_08_and_09_site_rag_query_isolation(site_a, site_b):
    """
    TEST 8 & 9 & 4K:
    Index both Site A (Spice Garden Restaurant) and Site B (Blue Ocean Restaurant).
    Query Site A: returns ONLY Site A chunks, never Site B.
    Query Site B: returns ONLY Site B chunks, never Site A.
    """
    doc_a = Document(
        page_content="Spice Garden serves authentic Gujarati thali. Address: Site A Test Address.",
        metadata={
            "site_id": site_a["site_id"],
            "url": "https://spicegarden.example.com/menu",
            "title": "Spice Garden Menu",
            "source": "https://spicegarden.example.com/menu"
        }
    )
    doc_b = Document(
        page_content="Blue Ocean serves fresh seafood. Address: Site B Test Address.",
        metadata={
            "site_id": site_b["site_id"],
            "url": "https://blueocean.example.com/menu",
            "title": "Blue Ocean Menu",
            "source": "https://blueocean.example.com/menu"
        }
    )

    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = [
        (doc_a, 0.2),
        (doc_b, 0.3)
    ]
    mock_vs.is_logically_deleted.return_value = False

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Mocked LLM answer"

    with patch("app.services.rag_engine.get_llm_service", return_value=mock_llm):
        rag = RAGEngine()
        rag.vector_store = mock_vs

    mock_mongodb = AsyncMock()
    mock_mongodb.db.sites.find_one = AsyncMock(side_effect=lambda q: site_a if q.get("site_id") == site_a["site_id"] else site_b)

    with patch("app.services.rag_engine.get_mongodb", AsyncMock(return_value=mock_mongodb)):
        # Query Site A
        results_a = await rag.debug_retrieve(query="What food does this restaurant serve?", site_id=site_a["site_id"], k=5)
        texts_a = [r["text"] for r in results_a]
        sources_a = [r["source"] for r in results_a]

        # Site A must have Gujarati thali, must NOT have Blue Ocean or seafood
        assert any("Gujarati thali" in t for t in texts_a)
        assert not any("Blue Ocean" in t for t in texts_a)
        assert not any("seafood" in t for t in texts_a)
        assert not any("Site B Test Address" in t for t in texts_a)
        assert all(site_a["site_id"] == r["site_id"] for r in results_a)

        # Query Site B
        results_b = await rag.debug_retrieve(query="What food does this restaurant serve?", site_id=site_b["site_id"], k=5)
        texts_b = [r["text"] for r in results_b]

        # Site B must have seafood, must NOT have Spice Garden or Gujarati thali
        assert any("seafood" in t for t in texts_b)
        assert not any("Spice Garden" in t for t in texts_b)
        assert not any("Gujarati thali" in t for t in texts_b)
        assert not any("Site A Test Address" in t for t in texts_b)
        assert all(site_b["site_id"] == r["site_id"] for r in results_b)


@pytest.mark.asyncio
async def test_10_uploaded_document_to_site_a_does_not_leak_to_site_b(site_a, site_b):
    """TEST 10: Document uploaded to Site A cannot appear in Site B retrieval."""
    uploaded_doc_a = Document(
        page_content="Spice Garden Confidential Recipe Document.",
        metadata={
            "site_id": site_a["site_id"],
            "url": f"doc://{site_a['site_id']}/doc_123",
            "source": f"doc://{site_a['site_id']}/doc_123",
            "source_type": "document"
        }
    )

    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = [(uploaded_doc_a, 0.1)]
    mock_vs.is_logically_deleted.return_value = False

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Mocked LLM answer"

    with patch("app.services.rag_engine.get_llm_service", return_value=mock_llm):
        rag = RAGEngine()
        rag.vector_store = mock_vs

    mock_mongodb = AsyncMock()
    mock_mongodb.db.sites.find_one = AsyncMock(return_value=site_b)

    with patch("app.services.rag_engine.get_mongodb", AsyncMock(return_value=mock_mongodb)):
        # Site B queries; uploaded doc belongs to Site A
        results_b = await rag.debug_retrieve(query="Confidential recipe", site_id=site_b["site_id"], k=5)
        assert len(results_b) == 0


@pytest.mark.asyncio
async def test_11_deleting_site_prevents_rag_retrieval(site_a):
    """TEST 11: Logical deletion of a site prevents its content from being returned."""
    doc_a = Document(
        page_content="Spice Garden Menu information.",
        metadata={
            "site_id": site_a["site_id"],
            "url": "https://spicegarden.example.com/menu"
        }
    )

    vs = VectorStore()
    vs._initialized = True
    # Simulate logical deletion of site_a
    vs.delete_by_metadata({"site_id": site_a["site_id"]})

    # Verify is_logically_deleted recognizes doc_a
    assert vs.is_logically_deleted(doc_a.metadata) is True

    # When similarity_search runs, doc_a is filtered out
    vs.vector_store = MagicMock()
    vs.vector_store.similarity_search_with_score.return_value = [(doc_a, 0.1)]

    res = vs.similarity_search_with_score("Menu", k=5)
    assert len(res) == 0


@pytest.mark.asyncio
async def test_12_and_13_public_chatbot_site_isolation(site_a, site_b):
    """
    TEST 12 & 13:
    Public chatbot for Site A retrieves only Site A content.
    Tampering with site_id to Site B in a public request routes solely to Site B's scope,
    never mixing or cross-contaminating data.
    """
    doc_a = Document(
        page_content="Welcome to Spice Garden!",
        metadata={"site_id": site_a["site_id"], "url": "https://spicegarden.example.com"}
    )
    doc_b = Document(
        page_content="Welcome to Blue Ocean Seafood!",
        metadata={"site_id": site_b["site_id"], "url": "https://blueocean.example.com"}
    )

    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = [(doc_a, 0.2), (doc_b, 0.2)]
    mock_vs.is_logically_deleted.return_value = False

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Here is information about your requested restaurant."

    with patch("app.services.rag_engine.get_llm_service", return_value=mock_llm):
        rag = RAGEngine()
        rag.vector_store = mock_vs
        rag.llm = mock_llm

    mock_mongodb = AsyncMock()
    mock_mongodb.db.sites.find_one = AsyncMock(side_effect=lambda q: site_a if q.get("site_id") == site_a["site_id"] else site_b)
    mock_mongodb.get_conversation_history.return_value = []
    mock_mongodb.save_conversation = AsyncMock()
    mock_mongodb.get_qa_for_rag = AsyncMock(return_value=[])
    mock_mongodb.get_site_handoff_config = AsyncMock(return_value={"enabled": False})

    with patch("app.services.rag_engine.get_mongodb", AsyncMock(return_value=mock_mongodb)), \
         patch("app.routes.chat.get_rag_engine", return_value=rag), \
         patch("app.routes.chat.get_mongodb", AsyncMock(return_value=mock_mongodb)):
        from app.main import app
        async with make_public_client(app) as client:
            # Request for Site A
            resp_a = await client.post("/api/chat", json={
                "message": "Tell me about your food",
                "session_id": "sess_public_a",
                "site_id": site_a["site_id"]
            })
            assert resp_a.status_code == 200
            data_a = resp_a.json()
            # Sources must contain only Site A
            for src in data_a.get("sources", []):
                assert src.get("site_id") == site_a["site_id"] or site_a["site_id"] in src.get("url", "")
                assert "blueocean" not in src.get("url", "")

            # Request for Site B
            resp_b = await client.post("/api/chat", json={
                "message": "Tell me about your food",
                "session_id": "sess_public_b",
                "site_id": site_b["site_id"]
            })
            assert resp_b.status_code == 200
            data_b = resp_b.json()
            for src in data_b.get("sources", []):
                assert src.get("site_id") == site_b["site_id"] or site_b["site_id"] in src.get("url", "")
                assert "spicegarden" not in src.get("url", "")


# =====================================================================
# Phase 5.1 Chunk 3 Task Verification Tests
# =====================================================================

@pytest.mark.asyncio
async def test_task1_and_2_document_replacement_rag_retrieval(site_a, owner_a):
    """
    TASK 1 & TASK 2:
    Verify document replacement flow:
    Old content: 'Restaurant closes at 10 PM.'
    Replaced with: 'Restaurant closes at 11 PM.'
    Verify retrieved chunks contain new content ('11 PM') and old content ('10 PM') is invalidated.
    """
    vs = VectorStore()
    vs._initialized = True
    vs._deleted_filters = []

    old_doc_id = "doc_old_10pm"
    old_url = f"doc://{site_a['site_id']}/{old_doc_id}"
    old_chunk = Document(
        page_content="Restaurant closes at 10 PM.",
        metadata={"url": old_url, "site_id": site_a["site_id"], "doc_id": old_doc_id}
    )

    new_doc_id = "doc_new_11pm"
    new_url = f"doc://{site_a['site_id']}/{new_doc_id}"
    new_chunk = Document(
        page_content="Restaurant closes at 11 PM.",
        metadata={"url": new_url, "site_id": site_a["site_id"], "doc_id": new_doc_id}
    )

    # Invalidate old document vectors via logical deletion sidecar
    vs.delete_by_metadata({"url": old_url})

    # Verify old chunk is marked logically deleted
    assert vs.is_logically_deleted(old_chunk.metadata) is True
    # Verify new chunk is NOT logically deleted
    assert vs.is_logically_deleted(new_chunk.metadata) is False

    # Simulate RAG retrieval with both chunks in index
    vs.similarity_search_with_score = MagicMock(return_value=[
        (old_chunk, 0.15),
        (new_chunk, 0.10)
    ])

    rag = RAGEngine()
    rag.vector_store = vs

    retrieved = await rag._retrieve_documents(
        "What time does the restaurant close?",
        site_id=site_a["site_id"],
        site=site_a
    )

    # Only new chunk must be returned
    assert len(retrieved) == 1
    doc, score = retrieved[0]
    assert "11 PM" in doc.page_content
    assert "10 PM" not in doc.page_content


@pytest.mark.asyncio
async def test_task3_faq_site_specific_and_rag(site_a, site_b):
    """
    TASK 3:
    Verify FAQs are site-specific and work alongside normal RAG.
    """
    mock_mongodb = AsyncMock()
    # FAQ for site A
    faq_a = {
        "id": "faq_1",
        "site_id": site_a["site_id"],
        "question": "What is the dress code?",
        "answer": "Smart casual attire is required.",
        "is_enabled": True
    }
    mock_mongodb.get_qa_for_rag = AsyncMock(side_effect=lambda sid: [faq_a] if sid == site_a["site_id"] else [])

    rag = RAGEngine()
    mock_vs = MagicMock()
    mock_vs.embeddings.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    rag.vector_store = mock_vs

    with patch("app.services.rag_engine.get_mongodb", AsyncMock(return_value=mock_mongodb)):
        # Check Site A matches FAQ
        match_a = await rag._check_qa_match("What is the dress code?", site_id=site_a["site_id"])
        assert match_a is not None
        qa_pair, score = match_a
        assert qa_pair["answer"] == "Smart casual attire is required."

        # Check Site B cannot access Site A FAQ
        match_b = await rag._check_qa_match("What is the dress code?", site_id=site_b["site_id"])
        assert match_b is None


@pytest.mark.asyncio
async def test_task4_lead_capture_persistence_and_check(site_a):
    """
    TASK 4:
    Verify Lead Capture persistence via public endpoint and check endpoint.
    """
    mock_mongodb = MockDatabaseProvider()
    site_record = {
        "_id": "site_db_id",
        "site_id": site_a["site_id"],
        "name": site_a["name"],
        "url": site_a["url"],
        "user_id": site_a["user_id"],
        "config": {
            "lead_capture": {
                "collect_email": True,
                "email_required": True,
                "capture_timing": "before_chat"
            }
        }
    }
    mock_mongodb.seed_site(site_record)

    with patch("app.routes.leads.get_mongodb", AsyncMock(return_value=mock_mongodb)):
        from app.main import app
        async with make_public_client(app) as client:
            # 1. Capture a lead
            resp = await client.post("/api/leads", json={
                "site_id": site_a["site_id"],
                "session_id": "test_lead_sess_1",
                "email": "customer@example.com",
                "name": "Jane Doe",
                "source": "chat"
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True

            # 2. Check lead exists
            check_resp = await client.get(f"/api/leads/check/{site_a['site_id']}/test_lead_sess_1")
            assert check_resp.status_code == 200
            check_data = check_resp.json()
            assert check_data["exists"] is True

            # 3. Non-existent session
            non_exist_resp = await client.get(f"/api/leads/check/{site_a['site_id']}/non_existent_sess")
            assert non_exist_resp.status_code == 200
            assert non_exist_resp.json()["exists"] is False

