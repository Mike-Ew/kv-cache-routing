"""Synthetic RAG workload: fixed system prompt + varying user queries.

Tests cross-user shared-prefix locality. Many different "users" all
hit the same long system prompt, but with different questions. If the
router sends them all to the same instance, that instance caches the
system prompt once and reuses it.

Usage:
    python -m workloads.rag_generator \
        --router http://localhost:9000 \
        --n 50 \
        --rate 2.0 \
        --output results/rag_run.jsonl
"""

import json
import time
import asyncio
import argparse
import random
import aiohttp
from pathlib import Path

# A long system prompt simulating a RAG context document (~500 tokens)
SYSTEM_PROMPT = """You are a helpful assistant for Acme Corporation's internal knowledge base.

Context document:
Acme Corporation was founded in 1985 in Austin, Texas by Dr. Sarah Chen and Marcus Williams.
The company initially focused on semiconductor manufacturing but pivoted to cloud infrastructure
services in 2010. Today, Acme operates 12 data centers across North America, Europe, and Asia.

Key products include:
- AcmeCloud: Enterprise cloud computing platform supporting AWS-compatible APIs
- AcmeDB: Distributed database service with automatic sharding and replication
- AcmeML: Machine learning platform for training and deploying models at scale
- AcmeEdge: Edge computing solution for IoT and real-time processing

Financial highlights (FY2025):
- Revenue: $4.2 billion (up 23% YoY)
- Operating margin: 18.5%
- R&D spending: $890 million
- Employees: 15,200 worldwide

Recent developments:
- Opened new data center in Singapore (Q3 2025)
- Launched AcmeML v3 with support for multi-modal models
- Acquired DataStream Inc. for $340 million to strengthen streaming analytics
- Partnered with European Space Agency for satellite data processing
- Published 12 research papers at major ML conferences

Corporate policies:
- All employees receive 20 days PTO plus company holidays
- Remote work policy allows up to 3 days per week from home
- Stock options vest over 4 years with a 1-year cliff
- Parental leave: 16 weeks paid for all parents
- Education stipend: $5,000 per year for professional development

Answer questions based only on the context provided above. If the answer is not in the
context, say "I don't have that information in the provided context."
"""

# Diverse user queries that all share the same system prompt
USER_QUERIES = [
    "When was Acme Corporation founded?",
    "Who are the founders of the company?",
    "How many data centers does Acme operate?",
    "What is AcmeCloud?",
    "What was the revenue in FY2025?",
    "Tell me about the parental leave policy.",
    "What is the remote work policy?",
    "How much does Acme spend on R&D?",
    "What acquisition did Acme make recently?",
    "Where is the newest data center?",
    "How many employees does Acme have?",
    "What is AcmeEdge used for?",
    "What is the education stipend amount?",
    "What is the stock option vesting schedule?",
    "Tell me about AcmeML v3.",
    "What is the operating margin?",
    "What partnership involves satellite data?",
    "How many research papers were published?",
    "What did Acme originally focus on?",
    "When did Acme pivot to cloud services?",
    "What is AcmeDB?",
    "How many PTO days do employees get?",
    "What was the YoY revenue growth?",
    "Where is Acme headquartered?",
    "What does DataStream Inc. specialize in?",
]


async def send_rag_request(
    session: aiohttp.ClientSession,
    router_url: str,
    query: str,
    user_id: str,
    model: str,
) -> dict:
    """Send a single RAG request with the shared system prompt."""
    start = time.monotonic()
    try:
        async with session.post(
            f"{router_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                "max_tokens": 100,
                "user": user_id,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            elapsed = time.monotonic() - start

            metadata = data.get("_routing_metadata", {})
            return {
                "user_id": user_id,
                "query": query,
                "proxy_latency_ms": metadata.get("proxy_latency_ms", round(elapsed * 1000, 1)),
                "instance": metadata.get("instance"),
                "timestamp": time.time(),
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
            }
    except Exception as e:
        return {
            "user_id": user_id,
            "query": query,
            "error": str(e),
            "timestamp": time.time(),
        }


async def run_workload(
    router_url: str,
    n_requests: int,
    rate: float,
    model: str,
    output_path: str,
):
    """Run the synthetic RAG workload."""
    print(f"Sending {n_requests} RAG requests at ~{rate} req/s")
    print(f"System prompt: {len(SYSTEM_PROMPT)} chars, shared across all requests")

    results = []
    async with aiohttp.ClientSession() as session:
        for i in range(n_requests):
            query = random.choice(USER_QUERIES)
            user_id = f"rag_user_{random.randint(0, 99)}"

            result = await send_rag_request(session, router_url, query, user_id, model)
            results.append(result)

            print(f"  [{i+1}/{n_requests}] user={user_id} instance={result.get('instance')} "
                  f"latency={result.get('proxy_latency_ms')}ms")

            # Rate limiting
            if rate > 0 and i < n_requests - 1:
                delay = random.expovariate(rate)
                await asyncio.sleep(delay)

    # Write results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. {len(results)} requests logged to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Synthetic RAG workload")
    parser.add_argument("--router", default="http://localhost:9000")
    parser.add_argument("--n", type=int, default=50, help="Number of requests")
    parser.add_argument("--rate", type=float, default=2.0, help="Requests per second")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="results/rag_run.jsonl")
    args = parser.parse_args()

    asyncio.run(run_workload(
        router_url=args.router,
        n_requests=args.n,
        rate=args.rate,
        model=args.model,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
