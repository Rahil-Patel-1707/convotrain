<div align="center">

# ConvoTrain AI

### AI-powered support chat, trained on your content

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com)
[![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20Ollama-f55036)](https://groq.com)

Build a website assistant from crawled pages and uploaded documents. ConvoTrain combines retrieval-augmented generation (RAG), an embeddable chat widget, and a dashboard for managing sites, conversations, leads, and human handoffs.

[Get started](#-quick-start) · [Features](#-what-you-can-do) · [API](#-api) · [Widget](#-embed-the-widget) · [Configuration](#-configuration) · [Tests](#-tests)

</div>

---

## At a glance

| Knowledge | Customer experience | Operations |
|:--|:--|:--|
| Crawl a site or upload PDF, DOCX, TXT, and Markdown files | Streamed AI chat with source references | Manage conversations, notes, tags, and status |
| Index content with configurable embeddings and vector search | Customize the widget appearance and quick prompts | Route visitors to a human agent and manage handoffs |
| Add curated question-and-answer pairs | Capture leads and collect chat feedback | Review analytics, crawl schedules, and trigger activity |

The backend is a FastAPI application. The dashboard and landing page are served by that app; the widget is a standalone JavaScript bundle that can be embedded on another site.

## Repository layout

The application source is in `sitechat-main/`:

```text
.
├── README.md
├── demo/
└── sitechat-main/
    ├── backend/
    │   ├── app/           # FastAPI routes, services, providers, and configuration
    │   └── tests/         # Pytest suite
    ├── frontend/
    │   ├── landing.html
    │   ├── index.html     # Dashboard
    │   ├── src/widget/    # Widget source
    │   └── widget/        # Built widget assets
    └── e2e/               # Playwright tests
```

## 🚀 Quick start

### Requirements

- Python 3.10 or newer
- MongoDB, available locally or through a connection URI
- A Groq API key for the default LLM, or a configured alternative provider
- Node.js only if you want to build the widget or run frontend/E2E tests

### 1. Create a Python environment

From the repository root, run these commands in PowerShell:

```powershell
cd sitechat-main/backend
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS or Linux, activate the environment with:

```bash
cd sitechat-main/backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure the backend

Edit `sitechat-main/backend/.env` before starting the server. At minimum, configure:

```env
ENVIRONMENT=development
DEBUG=true
SITE_URL=http://localhost:8000
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=convotrain
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key
JWT_SECRET=replace-with-a-random-secret-of-at-least-32-characters
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=replace-with-a-strong-password
```

Generate a JWT secret with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

The example environment file contains production-oriented placeholders. Use your own credentials and set the CORS origins and trusted hosts for your deployment. An initial admin account is created when `ADMIN_PASSWORD` is configured and no admin account exists.

### 3. Start the API and web app

With MongoDB running and the virtual environment active, from `sitechat-main/backend` run:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

| URL | Page |
|:--|:--|
| `http://localhost:8000/` | Landing page |
| `http://localhost:8000/login` | Sign in |
| `http://localhost:8000/app` | Dashboard |
| `http://localhost:8000/api/docs` | Interactive API docs |
| `http://localhost:8000/api/redoc` | ReDoc API docs |
| `http://localhost:8000/demo` | Redirects to the landing page demo |

## ✨ What you can do

### Knowledge and RAG

- Crawl and re-crawl website pages, with URL filtering and scheduled runs.
- Upload PDF, DOCX, TXT, and Markdown knowledge files.
- Configure chunking, retrieval, and supported LLM, embedding, and vector-store providers.
- Maintain site-specific configuration and curated Q&A pairs.

### Chat and engagement

- Send JSON chat requests or receive server-sent event (SSE) streamed responses.
- Display citations to retrieved sources and gather visitor feedback.
- Configure widget appearance, welcome text, and quick prompts per site.
- Capture visitor leads and configure proactive chat triggers.

### Support operations

- Browse, search, annotate, tag, prioritize, and update conversation status.
- Route a conversation to a human, manage agent queues, and exchange messages.
- Configure business hours and monitor handoffs in real time.
- Review conversation and trigger analytics; export leads and supported conversation data.

### Accounts and access

The API uses JWT authentication and role-aware access for administrators, site owners, and support agents. Access to sites and conversations is scoped by role and assignment. Some operations, such as exports and deletes, are intentionally restricted by role.

## 🔌 Embed the widget

Add the built widget script to a page. Replace the site ID and host with values for your deployment:

```html
<script
  src="https://chat.example.com/widget/chatbot.js"
  data-site-id="YOUR_SITE_ID"
  data-api-url="https://chat.example.com"
  data-primary-color="#1B5E3B"
  data-title="Ask us">
</script>
```

`data-api-url` is the API origin (do not append `/api`). When the page and API share an origin, it can be omitted. The widget also reads appearance and behavior settings from the site's saved configuration.

### Build the widget

The source of truth is `sitechat-main/frontend/src/widget/chatbot.js`. From `sitechat-main/frontend`:

```bash
npm install
npm run build
```

The build script writes the distributable widget files under `frontend/widget/`. Edit the source file, then rebuild; do not edit generated bundles directly.

## 📚 API

Open `/api/docs` while the app is running for the complete, interactive schema. Main route groups include:

| Area | Example routes |
|:--|:--|
| Authentication and users | `/api/auth/login`, `/api/auth/me`, `/api/auth/agents` |
| Sites and configuration | `/api/sites`, `/api/sites/{site_id}/config` |
| Chat and history | `/api/chat`, `/api/chat/stream`, `/api/chat/history/{session_id}` |
| Crawling and schedules | `/api/crawl`, `/api/sites/{site_id}/crawl-schedule` |
| Documents | `/api/documents/upload/{site_id}`, `/api/documents/{site_id}` |
| Conversations and analytics | `/api/conversations`, `/api/analytics/overview` |
| Q&A training | `/api/sites/{site_id}/qa` |
| Handoffs | `/api/handoff`, `/api/sites/{site_id}/handoff/queue` |
| Leads and triggers | `/api/sites/{site_id}/leads`, `/api/sites/{site_id}/triggers` |
| Embedding and platform branding | `/api/embed/script/{site_id}`, `/api/platform/whitelabel` |

Most dashboard endpoints require a Bearer token. Public widget endpoints have their own validation and rate limits; consult the API docs for request bodies and access requirements.

## ⚙️ Configuration

The complete set of options and comments is in `sitechat-main/backend/.env.example`.

| Component | Default | Other configured options |
|:--|:--|:--|
| LLM | Groq | Ollama, OpenAI, Anthropic, Azure OpenAI |
| Embeddings | Hugging Face | Ollama, OpenAI |
| Vector store | FAISS | Chroma, Pinecone, Qdrant |
| Database | MongoDB | PostgreSQL is marked coming soon in the example configuration |
| File storage | Local | S3 and GCS are marked coming soon |
| Cache | In-memory | Redis is marked coming soon |

Alternative LLM and vector providers may require additional packages, credentials, and service configuration beyond the base `requirements.txt`. The example configuration explicitly labels PostgreSQL, S3/GCS, and Redis as coming soon; their environment-variable names do not mean those adapters are ready to use.

## 🧪 Tests

Run commands from `sitechat-main/`:

```bash
# Backend
cd backend
pytest

# Frontend unit tests
cd ../frontend
npm install
npm test

# End-to-end tests (Playwright)
cd ../e2e
npm install
npm test
```

The backend pytest configuration is in `backend/pytest.ini`. Frontend tests use Jest; end-to-end tests use Playwright.

## 🔐 Production notes

- Set `ENVIRONMENT=production` and `DEBUG=false` only after configuring production CORS origins, trusted hosts, and a strong JWT secret.
- Keep `.env`, provider keys, and database credentials out of version control.
- Use HTTPS and restrict MongoDB network access in production.
- Security headers, request validation, and rate limiting are implemented in the backend; review your reverse proxy and hosting configuration as well.
- Widget embed security endpoints provide script/security metadata, including SRI-related information. See `/api/docs` for their current contract.

---

<div align="center">

Built with FastAPI, MongoDB, LangChain, and a little patience for good answers.

</div>