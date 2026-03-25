<div align="center">

# 🚀 AI Agent Orchestration Platform

**A production-grade platform for building, orchestrating, and deploying autonomous AI agents with RAG pipelines, tool integrations, and scalable cloud infrastructure.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green?logo=chainlink&logoColor=white)](https://langchain.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)

</div>

---

## 📖 Overview

The **AI Agent Orchestration Platform** is a full-stack system that empowers users to create and manage intelligent AI agents capable of reasoning, accessing external tools, and retrieving knowledge from vector databases — all through a clean web interface.

At its core, the platform uses the **ReAct** (Reasoning + Acting) paradigm: agents iteratively think, act via tools, observe results, and refine their responses. Combined with **Retrieval-Augmented Generation (RAG)** and the **Model Context Protocol (MCP)**, agents can pull in real-time knowledge and interface with external services seamlessly.

### Key Features

- 🤖 **ReAct Agents** — LLM-powered agents that reason and use tools autonomously
- 📚 **RAG Pipelines** — Retrieve relevant context from Pinecone vector stores before answering
- 🔌 **MCP Integration** — Standardised protocol for connecting agents to external tools and data
- ⚡ **Real-time API** — FastAPI backend exposing agent capabilities via REST endpoints
- 🎨 **React Dashboard** — Modern frontend to interact with and monitor agents
- ☁️ **Cloud-Native Deployment** — Designed for AWS Lambda, SQS, and Kubernetes at scale

---

## 🛠 Tech Stack

| Layer            | Technology        | Purpose                                      |
| ---------------- | ----------------- | -------------------------------------------- |
| **Language**     | Python 3.12       | Backend logic and agent orchestration         |
| **API Framework**| FastAPI           | High-performance async REST API               |
| **LLM Framework**| LangChain         | Agent construction, prompt management, chains |
| **LLM Provider** | Groq (LLaMA 3.3) | Ultra-fast LLM inference                      |
| **Vector DB**    | Pinecone          | Embedding storage and similarity search (RAG) |
| **Agent Protocol**| MCP              | Model Context Protocol for tool integration   |
| **Frontend**     | React             | Interactive web dashboard                     |
| **Serverless**   | AWS Lambda        | Event-driven agent execution                  |
| **Messaging**    | AWS SQS           | Async task queuing between services           |
| **Orchestration**| Kubernetes        | Container orchestration for production deploy |

---

## 🏗 System Architecture

```
┌─────────────┐       ┌──────────────────────────────────────────────┐
│   React UI  │◄─────►│              FastAPI Backend                 │
│  (Frontend) │  REST │                                              │
└─────────────┘       │  ┌──────────┐   ┌─────────┐   ┌──────────┐ │
                      │  │  ReAct   │──►│  Tools  │──►│   MCP    │ │
                      │  │  Agent   │   │ (Custom)│   │ Protocol │ │
                      │  └────┬─────┘   └─────────┘   └──────────┘ │
                      │       │                                      │
                      │       ▼                                      │
                      │  ┌──────────┐   ┌───────────┐               │
                      │  │   Groq   │   │ Pinecone  │               │
                      │  │  LLaMA   │   │ (Vector   │               │
                      │  │   API    │   │    DB)    │               │
                      │  └──────────┘   └───────────┘               │
                      └──────────────────────────────────────────────┘
                                         │
                      ┌──────────────────┼──────────────────┐
                      │           AWS Cloud                 │
                      │  ┌──────────┐   ┌───────────┐      │
                      │  │  Lambda  │◄──│    SQS    │      │
                      │  │Functions │   │  Queues   │      │
                      │  └──────────┘   └───────────┘      │
                      │         Kubernetes Cluster          │
                      └─────────────────────────────────────┘
```

### How It Works

1. **User Query** → The React frontend sends a natural-language query to the FastAPI backend via `POST /api/agent/query`.

2. **ReAct Loop** → The agent enters the Thought → Action → Observation cycle:
   - **Thought**: The LLM (Groq/LLaMA 3.3) analyses the query and decides what to do.
   - **Action**: It invokes a tool (e.g. Pinecone search, web lookup, MCP-connected service).
   - **Observation**: The tool result is fed back into the LLM context.
   - This loops until the agent has enough information to produce a **Final Answer**.

3. **RAG Pipeline** → For knowledge-intensive queries, the agent retrieves relevant document embeddings from Pinecone, injecting them as context for more accurate responses.

4. **MCP Protocol** → External tools and services are connected via the Model Context Protocol, providing a standardised interface for agents to discover and call tools dynamically.

5. **Response** → The final answer is returned as structured JSON to the frontend.

---

## ⚡ Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+ (for frontend)
- A [Groq API key](https://console.groq.com)
- A [Pinecone API key](https://www.pinecone.io)

### 1. Clone the Repository

```bash
git clone https://github.com/JITHENTHIRIYA/ai-agent-orchestration-platform.git
cd ai-agent-orchestration-platform
```

### 2. Set Up the Backend

```bash
cd backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your API keys:
#   GROQ_API_KEY=your_key_here
#   PINECONE_API_KEY=your_key_here
#   PINECONE_INDEX=your_index_name
```

### 3. Run the Backend

```bash
uvicorn main:app --reload
```

The API will be available at **http://localhost:8000**

### 4. Verify It Works

```bash
# Health check
curl http://localhost:8000/health

# Query the agent
curl -X POST http://localhost:8000/api/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 2+2?"}'
```

### 5. API Documentation

FastAPI auto-generates interactive docs:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📁 Project Structure

```
ai-agent-orchestration-platform/
├── backend/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── base_agent.py      # ReAct agent with Groq/LLaMA 3.3
│   ├── tools/
│   │   └── __init__.py        # Custom tool definitions
│   ├── main.py                # FastAPI app + API endpoints
│   ├── config.py              # Environment variable management
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example           # Env var template
│   └── .env                   # Local secrets (git-ignored)
├── frontend/                  # React dashboard (coming soon)
├── .gitignore
└── README.md
```

---

## 🗺 Roadmap

| Week | Focus Area                        | Key Deliverables                                                  |
| ---- | --------------------------------- | ----------------------------------------------------------------- |
| 1    | **Foundation & Agent Setup**      | FastAPI scaffold, ReAct agent, Groq integration, health check API |
| 2    | **RAG Pipeline & Tools**          | Pinecone vector store, document ingestion, retrieval tool          |
| 3    | **MCP Integration**               | Model Context Protocol server, dynamic tool discovery              |
| 4    | **Frontend & Dashboard**          | React UI, agent chat interface, query history                      |
| 5    | **Cloud Deployment**              | AWS Lambda functions, SQS queuing, Kubernetes setup                |
| 6    | **Polish & Production Readiness** | Auth, rate limiting, monitoring, docs, performance tuning          |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Commit with conventional messages (`feat:`, `fix:`, `chore:`)
4. Push and open a Pull Request

---

## 📄 License

This project is developed as part of a portfolio project by [Jithen Thiriyakathirvel](https://github.com/JITHENTHIRIYA).

---

<div align="center">

**Built with ❤️ using LangChain, Groq, and FastAPI**

</div>
