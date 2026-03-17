"""Replay multi-turn ShareGPT conversations through the router.

ShareGPT format: each entry has an "id" and "conversations" list.
Each turn is {"from": "human"|"gpt", "value": "..."}.

We replay only the human turns as requests, building up the message
history so the router sees growing prefixes (mimicking real multi-turn chat).

Usage:
    python -m workloads.sharegpt_replayer \
        --router http://localhost:9000 \
        --dataset sharegpt.json \
        --n 20 \
        --rate 1.0 \
        --output results/sharegpt_run.jsonl
"""

import json
import time
import asyncio
import argparse
import random
import aiohttp
from pathlib import Path


async def replay_conversation(
    session: aiohttp.ClientSession,
    router_url: str,
    conversation: dict,
    conv_id: str,
    model: str,
) -> list[dict]:
    """Replay one multi-turn conversation, sending each human turn as a request."""
    messages = []
    results = []

    turns = conversation.get("conversations", [])
    turn_idx = 0

    while turn_idx < len(turns):
        turn = turns[turn_idx]

        if turn["from"] != "human":
            # Skip leading non-human turns (shouldn't happen but be safe)
            turn_idx += 1
            continue

        # Add user message from dataset
        messages.append({"role": "user", "content": turn["value"]})
        turn_idx += 1

        start = time.monotonic()
        try:
            async with session.post(
                f"{router_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": list(messages),  # copy so it grows each turn
                    "max_tokens": 100,
                    "user": conv_id,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
                elapsed = time.monotonic() - start

                metadata = data.get("_routing_metadata", {})
                results.append({
                    "conversation_id": conv_id,
                    "turn": len(results) + 1,
                    "proxy_latency_ms": metadata.get("proxy_latency_ms", round(elapsed * 1000, 1)),
                    "instance": metadata.get("instance"),
                    "timestamp": time.time(),
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
                })

                # Use the dataset's assistant response (not generated) for history
                # This keeps prefix growth deterministic across policies
                if turn_idx < len(turns) and turns[turn_idx]["from"] == "gpt":
                    messages.append({"role": "assistant", "content": turns[turn_idx]["value"]})
                    turn_idx += 1
                elif data.get("choices"):
                    # Fallback: use generated response if dataset has no gpt turn
                    assistant_msg = data["choices"][0].get("message", {}).get("content", "")
                    messages.append({"role": "assistant", "content": assistant_msg})

        except Exception as e:
            results.append({
                "conversation_id": conv_id,
                "turn": len(results) + 1,
                "error": str(e),
                "timestamp": time.time(),
            })
            # Still advance past the gpt turn if present
            if turn_idx < len(turns) and turns[turn_idx]["from"] == "gpt":
                messages.append({"role": "assistant", "content": turns[turn_idx]["value"]})
                turn_idx += 1

    return results


async def run_workload(
    router_url: str,
    dataset_path: str,
    n_conversations: int,
    rate: float,
    model: str,
    output_path: str,
):
    """Run the ShareGPT replay workload."""
    # Load and filter conversations
    with open(dataset_path) as f:
        all_conversations = json.load(f)

    # Filter: 2-10 turns, skip empty
    filtered = [
        c for c in all_conversations
        if 2 <= len(c.get("conversations", [])) <= 20
    ]
    print(f"Loaded {len(all_conversations)} conversations, {len(filtered)} after filtering")

    if n_conversations > len(filtered):
        n_conversations = len(filtered)
    sampled = random.sample(filtered, n_conversations)

    print(f"Replaying {n_conversations} conversations at ~{rate} req/s")

    all_results = []
    async with aiohttp.ClientSession() as session:
        for i, conv in enumerate(sampled):
            conv_id = conv.get("id", str(i))
            results = await replay_conversation(session, router_url, conv, conv_id, model)
            all_results.extend(results)

            # Rate limiting between conversations
            if rate > 0 and i < len(sampled) - 1:
                delay = random.expovariate(rate)
                await asyncio.sleep(delay)

            print(f"  [{i+1}/{n_conversations}] conv={conv_id}, turns={len(results)}")

    # Write results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. {len(all_results)} requests logged to {output_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Replay ShareGPT conversations")
    parser.add_argument("--router", default="http://localhost:9000")
    parser.add_argument("--dataset", required=True, help="Path to ShareGPT JSON file")
    parser.add_argument("--n", type=int, default=20, help="Number of conversations to replay")
    parser.add_argument("--rate", type=float, default=1.0, help="Conversations per second")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="results/sharegpt_run.jsonl")
    args = parser.parse_args()

    asyncio.run(run_workload(
        router_url=args.router,
        dataset_path=args.dataset,
        n_conversations=args.n,
        rate=args.rate,
        model=args.model,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
