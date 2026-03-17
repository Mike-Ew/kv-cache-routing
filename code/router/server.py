"""FastAPI server — the HTTP entry point for the routing gateway."""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from router.graph import routing_graph
from router import nodes

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

app = FastAPI(title="KV Cache Router")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Accept an OpenAI-compatible chat request and route it through the LangGraph pipeline."""
    body = await request.json()

    result = await routing_graph.ainvoke({"request": body})

    return JSONResponse(content=result["response"])


@app.get("/health")
async def health():
    return {"status": "ok", "policy": nodes._policy_name}
