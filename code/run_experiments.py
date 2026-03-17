"""Experiment runner with proper research methodology.

Key design decisions:
- Open-loop load generation (send at fixed rate, don't wait for responses)
- Warm-up phase before measurement
- vLLM metrics snapshots (before/after) for ground-truth TTFT and cache stats
- Fixed random seed per experiment for reproducibility across policies
- Multiple trials with different seeds for confidence intervals
- Concurrent requests to create realistic queue pressure

Usage:
    python3 run_experiments.py --exp 1        # Single experiment
    python3 run_experiments.py --exp 1 2      # Multiple experiments
    python3 run_experiments.py --all          # All experiments
    python3 run_experiments.py --all --dry-run
"""

import subprocess
import sys
import time
import signal
import json
import argparse
import asyncio
import os
import random
import aiohttp
import urllib.request
from pathlib import Path
from datetime import datetime

CODE_DIR = Path(__file__).parent
RESULTS_DIR = CODE_DIR / "results"
ROUTER_HOST = "0.0.0.0"
ROUTER_PORT = 9000
ROUTER_URL = f"http://localhost:{ROUTER_PORT}"
FORCE_CLAIM_PORT = False
SHAREGPT_PATH = CODE_DIR / "workloads" / "data" / "sharegpt.json"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

VLLM_INSTANCES = [
    "http://100.106.110.117:8000",
    "http://100.106.110.117:8001",
]

ALL_POLICIES = [
    "round_robin",
    "jsq",
    "p2c",
    "session_affinity",
    "prefix_affinity",
    "load_cache_aware",
]

N_TRIALS = 3
WARMUP_REQUESTS = 20
SEEDS = [42, 123, 7]  # One per trial, fixed for reproducibility


# ---------------------------------------------------------------------------
# vLLM metrics snapshot
# ---------------------------------------------------------------------------

async def scrape_vllm_metrics(instance_url: str) -> dict:
    """Scrape ground-truth metrics from a vLLM instance's /metrics endpoint."""
    import re
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{instance_url}/metrics",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                text = await resp.text()

        def parse(name):
            pattern = rf'^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([\d.eE+\-]+)'
            for line in text.split("\n"):
                m = re.match(pattern, line)
                if m:
                    return float(m.group(1))
            return 0.0

        return {
            "ttft_sum": parse("vllm:time_to_first_token_seconds_sum"),
            "ttft_count": parse("vllm:time_to_first_token_seconds_count"),
            "cache_hits": parse("vllm:prefix_cache_hits_total"),
            "cache_queries": parse("vllm:prefix_cache_queries_total"),
            "requests_waiting": parse("vllm:num_requests_waiting"),
            "requests_running": parse("vllm:num_requests_running"),
        }
    except Exception as e:
        print(f"    WARNING: Could not scrape {instance_url}: {e}")
        return {}


async def snapshot_all_instances() -> list[dict]:
    """Scrape metrics from all vLLM instances."""
    tasks = [scrape_vllm_metrics(url) for url in VLLM_INSTANCES]
    return await asyncio.gather(*tasks)


def compute_deltas(before: list[dict], after: list[dict]) -> dict:
    """Compute per-instance metric deltas between two snapshots."""
    deltas = []
    for b, a in zip(before, after):
        if not b or not a:
            deltas.append({})
            continue
        ttft_count_delta = a["ttft_count"] - b["ttft_count"]
        ttft_sum_delta = a["ttft_sum"] - b["ttft_sum"]
        cache_hits_delta = a["cache_hits"] - b["cache_hits"]
        cache_queries_delta = a["cache_queries"] - b["cache_queries"]
        deltas.append({
            "avg_ttft_ms": round((ttft_sum_delta / ttft_count_delta) * 1000, 2)
                          if ttft_count_delta > 0 else 0,
            "ttft_count": int(ttft_count_delta),
            "cache_hit_rate": round(cache_hits_delta / cache_queries_delta, 4)
                             if cache_queries_delta > 0 else 0,
            "cache_hits": int(cache_hits_delta),
            "cache_queries": int(cache_queries_delta),
        })

    # Aggregate across instances
    total_ttft_count = sum(d.get("ttft_count", 0) for d in deltas)
    total_ttft_sum = sum(
        (a["ttft_sum"] - b["ttft_sum"]) for b, a in zip(before, after) if b and a
    )
    total_hits = sum(d.get("cache_hits", 0) for d in deltas)
    total_queries = sum(d.get("cache_queries", 0) for d in deltas)

    return {
        "per_instance": deltas,
        "aggregate": {
            "avg_ttft_ms": round((total_ttft_sum / total_ttft_count) * 1000, 2)
                          if total_ttft_count > 0 else 0,
            "total_requests": total_ttft_count,
            "cache_hit_rate": round(total_hits / total_queries, 4)
                             if total_queries > 0 else 0,
        },
    }


# ---------------------------------------------------------------------------
# Router management
# ---------------------------------------------------------------------------

def start_router(policy: str, extra_env: dict = None) -> subprocess.Popen:
    """Start the FastAPI router with a given policy."""
    ensure_port_free(ROUTER_PORT, force=FORCE_CLAIM_PORT)

    env = os.environ.copy()
    env["ROUTING_POLICY"] = policy
    if extra_env:
        env.update(extra_env)

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "router.server:app",
         "--host", ROUTER_HOST, "--port", str(ROUTER_PORT),
         "--log-level", "warning"],
        cwd=str(CODE_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(60):
        time.sleep(0.5)
        if proc.poll() is not None:
            raise RuntimeError(
                f"Router process exited early (exit_code={proc.returncode}) for policy={policy}"
            )

        health = get_router_health()
        if not health:
            continue

        if health.get("status") != "ok":
            continue

        active_policy = health.get("policy")
        if active_policy and active_policy != policy:
            stop_router(proc)
            raise RuntimeError(
                f"Policy isolation failure: expected '{policy}', got '{active_policy}' on {ROUTER_URL}"
            )

        listeners = get_listening_pids(ROUTER_PORT)
        if listeners != {proc.pid}:
            stop_router(proc)
            raise RuntimeError(
                f"Router ownership failure on port {ROUTER_PORT}: expected pid {proc.pid}, found {sorted(listeners)}"
            )

        return proc

    stop_router(proc)
    raise RuntimeError(f"Router did not become ready within 30s for policy={policy}")


def stop_router(proc: subprocess.Popen):
    """Gracefully stop the router."""
    if proc.poll() is not None:
        return

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def get_listening_pids(port: int) -> set[int]:
    """Return process IDs currently listening on the given TCP port."""
    cmd = ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # lsof returns 1 when there are no matching processes.
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Failed to query listeners on port {port}: {result.stderr.strip()}")

    pids = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids


def ensure_port_free(port: int, force: bool, timeout_sec: float = 5.0):
    """Ensure the target port is available.

    By default this is non-destructive and raises if the port is occupied.
    Use force=True to terminate existing listeners.
    """
    pids = get_listening_pids(port)
    if not pids:
        return

    if not force:
        raise RuntimeError(
            f"Port {port} is already in use by PID(s) {sorted(pids)}. "
            "Use --router-port to select a different port, or --force-claim-port to terminate listeners."
        )

    print(f"    Releasing port {port} from stale listeners: {sorted(pids)}")
    for pid in pids:
        if pid != os.getpid():
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not get_listening_pids(port):
            return
        time.sleep(0.1)

    pids = get_listening_pids(port)
    for pid in pids:
        if pid != os.getpid():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    time.sleep(0.2)
    remaining = get_listening_pids(port)
    if remaining:
        raise RuntimeError(f"Could not free port {port}; remaining listeners: {sorted(remaining)}")


def get_router_health() -> dict | None:
    """Return router /health payload, or None if unreachable/invalid."""
    try:
        with urllib.request.urlopen(f"{ROUTER_URL}/health", timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def sanity_check_distribution(policy: str, summary: dict):
    """Fail fast when policy behavior indicates broken experiment isolation."""
    dist = summary.get("instance_distribution", {})
    if not dist:
        raise RuntimeError(f"{policy}: no routed instances recorded")

    if policy == "round_robin":
        if len(dist) != len(VLLM_INSTANCES):
            raise RuntimeError(
                f"round_robin sanity failure: expected {len(VLLM_INSTANCES)} active instances, got {dist}"
            )
        counts = list(dist.values())
        if max(counts) - min(counts) > 1:
            raise RuntimeError(f"round_robin imbalance sanity failure: {dist}")

    if policy in ("jsq", "p2c") and len(dist) < 2:
        raise RuntimeError(
            f"{policy} sanity failure: only one instance used ({dist}). "
            "Likely stale router, backend outage, or configuration issue."
        )


# ---------------------------------------------------------------------------
# Open-loop workload sender
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    router_url: str,
    payload: dict,
) -> dict:
    """Send one request and record timing."""
    start = time.monotonic()
    try:
        async with session.post(
            f"{router_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            elapsed = time.monotonic() - start
            metadata = data.get("_routing_metadata", {})
            return {
                "proxy_latency_ms": metadata.get("proxy_latency_ms", round(elapsed * 1000, 1)),
                "instance": metadata.get("instance"),
                "timestamp": time.time(),
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
                "user": payload.get("user", ""),
                "error": None,
            }
    except Exception as e:
        return {
            "proxy_latency_ms": -1,
            "instance": None,
            "timestamp": time.time(),
            "user": payload.get("user", ""),
            "error": str(e),
        }


async def run_open_loop(
    payloads: list[dict],
    router_url: str,
    rate: float,
    concurrency: int = 20,
) -> list[dict]:
    """Send requests at a fixed rate (open-loop), allowing concurrency.

    This creates realistic queue pressure unlike sequential sending.
    """
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_send(session, payload):
        async with sem:
            return await send_request(session, router_url, payload)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, payload in enumerate(payloads):
            task = asyncio.create_task(bounded_send(session, payload))
            tasks.append(task)
            if rate > 0 and i < len(payloads) - 1:
                await asyncio.sleep(1.0 / rate)

        results = await asyncio.gather(*tasks)

    return list(results)


# ---------------------------------------------------------------------------
# Workload builders (deterministic given a seed)
# ---------------------------------------------------------------------------

def build_sharegpt_payloads(n_conversations: int, seed: int) -> list[dict]:
    """Build a deterministic set of ShareGPT request payloads."""
    rng = random.Random(seed)

    with open(SHAREGPT_PATH) as f:
        all_conversations = json.load(f)

    filtered = [
        c for c in all_conversations
        if 2 <= len(c.get("conversations", [])) <= 20
    ]
    sampled = rng.sample(filtered, min(n_conversations, len(filtered)))

    payloads = []
    for conv in sampled:
        conv_id = conv.get("id", "unknown")
        messages = []
        for turn in conv.get("conversations", []):
            if turn["from"] == "human":
                messages.append({"role": "user", "content": turn["value"]})
                payloads.append({
                    "model": MODEL,
                    "messages": list(messages),
                    "max_tokens": 100,
                    "user": conv_id,
                })
            else:
                # Use dataset assistant turns for history (deterministic)
                messages.append({"role": "assistant", "content": turn["value"]})

    return payloads


def build_rag_payloads(n_requests: int, seed: int) -> list[dict]:
    """Build a deterministic set of RAG request payloads."""
    from workloads.rag_generator import SYSTEM_PROMPT, USER_QUERIES

    rng = random.Random(seed)
    payloads = []
    for i in range(n_requests):
        query = rng.choice(USER_QUERIES)
        user_id = f"rag_user_{rng.randint(0, 99)}"
        payloads.append({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "max_tokens": 100,
            "user": user_id,
        })
    return payloads


# ---------------------------------------------------------------------------
# Single experiment run (one policy, one workload, one trial)
# ---------------------------------------------------------------------------

async def run_single(
    policy: str,
    payloads: list[dict],
    warmup_payloads: list[dict],
    rate: float,
    label: str,
) -> dict:
    """Run one trial: warm-up, snapshot metrics, measure, snapshot again."""
    # Warm-up: send some requests to populate caches
    if warmup_payloads:
        print(f"    Warm-up: {len(warmup_payloads)} requests...")
        await run_open_loop(warmup_payloads, ROUTER_URL, rate=rate)
        await asyncio.sleep(1)  # Let queues drain

    # Snapshot metrics BEFORE measurement
    metrics_before = await snapshot_all_instances()

    # Measurement phase
    print(f"    Measuring: {len(payloads)} requests at {rate} req/s...")
    start_time = time.monotonic()
    results = await run_open_loop(payloads, ROUTER_URL, rate=rate)
    wall_time = time.monotonic() - start_time

    await asyncio.sleep(1)  # Let in-flight requests finish

    # Snapshot metrics AFTER measurement
    metrics_after = await snapshot_all_instances()
    deltas = compute_deltas(metrics_before, metrics_after)

    # Compute result summary
    latencies = [r["proxy_latency_ms"] for r in results if r["proxy_latency_ms"] > 0]
    instances = [r["instance"] for r in results if r["instance"] is not None]
    from collections import Counter
    inst_dist = Counter(instances)
    errors = sum(1 for r in results if r["error"])

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    summary = {
        "policy": policy,
        "label": label,
        "n_requests": len(payloads),
        "n_errors": errors,
        "wall_time_s": round(wall_time, 2),
        "throughput_rps": round(len(payloads) / wall_time, 2),
        "instance_distribution": dict(inst_dist),
        "load_imbalance_ratio": round(max(inst_dist.values()) / max(1, min(inst_dist.values())), 2)
                                if inst_dist else 0,
        "latency_ms": {
            "p50": round(sorted_lat[n // 2], 1) if n > 0 else 0,
            "p95": round(sorted_lat[int(n * 0.95)], 1) if n > 0 else 0,
            "p99": round(sorted_lat[int(n * 0.99)], 1) if n > 0 else 0,
            "mean": round(sum(sorted_lat) / n, 1) if n > 0 else 0,
        },
        "vllm_metrics": deltas,
    }
    return {"summary": summary, "raw_results": results}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def experiment_1(dry_run=False):
    """Exp 1: Policy comparison on ShareGPT multi-turn chat.

    50 conversations (~150 requests), 3 trials per policy, open-loop at 5 req/s.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Policy Comparison — ShareGPT Multi-Turn")
    print("=" * 60)

    n_conversations = 50
    rate = 5.0  # req/s — high enough to build queues

    for policy in ALL_POLICIES:
        print(f"\n  Policy: {policy}")
        exp_dir = RESULTS_DIR / "exp1" / policy

        if dry_run:
            print(f"    [DRY RUN] {N_TRIALS} trials × ~150 requests at {rate} req/s")
            continue

        exp_dir.mkdir(parents=True, exist_ok=True)

        for trial in range(N_TRIALS):
            seed = SEEDS[trial]
            print(f"\n    Trial {trial + 1}/{N_TRIALS} (seed={seed})")

            # Build deterministic payloads
            payloads = build_sharegpt_payloads(n_conversations, seed=seed)
            warmup = build_sharegpt_payloads(10, seed=seed + 1000)
            print(f"    Built {len(payloads)} request payloads")

            # Start fresh router for each trial
            router = start_router(policy)
            try:
                result = asyncio.run(run_single(
                    policy, payloads, warmup, rate, f"sharegpt_trial{trial}",
                ))

                # Save results
                with open(exp_dir / f"trial_{trial}_summary.json", "w") as f:
                    json.dump(result["summary"], f, indent=2)
                with open(exp_dir / f"trial_{trial}_raw.jsonl", "w") as f:
                    for r in result["raw_results"]:
                        f.write(json.dumps(r) + "\n")

                s = result["summary"]
                sanity_check_distribution(policy, s)
                print(f"    Results: P50={s['latency_ms']['p50']}ms "
                      f"P95={s['latency_ms']['p95']}ms "
                      f"cache_hit={s['vllm_metrics']['aggregate']['cache_hit_rate']:.2%} "
                      f"dist={s['instance_distribution']}")
            finally:
                stop_router(router)
                time.sleep(2)

    print("\nExperiment 1 complete.")


def experiment_2(dry_run=False):
    """Exp 2: Policy comparison on RAG workload (shared system prompts).

    100 requests per policy, 3 trials, open-loop at 5 req/s.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Policy Comparison — Synthetic RAG")
    print("=" * 60)

    n_requests = 100
    rate = 5.0

    for policy in ALL_POLICIES:
        print(f"\n  Policy: {policy}")
        exp_dir = RESULTS_DIR / "exp2" / policy

        if dry_run:
            print(f"    [DRY RUN] {N_TRIALS} trials × {n_requests} requests at {rate} req/s")
            continue

        exp_dir.mkdir(parents=True, exist_ok=True)

        for trial in range(N_TRIALS):
            seed = SEEDS[trial]
            print(f"\n    Trial {trial + 1}/{N_TRIALS} (seed={seed})")

            payloads = build_rag_payloads(n_requests, seed=seed)
            warmup = build_rag_payloads(WARMUP_REQUESTS, seed=seed + 1000)

            router = start_router(policy)
            try:
                result = asyncio.run(run_single(
                    policy, payloads, warmup, rate, f"rag_trial{trial}",
                ))

                with open(exp_dir / f"trial_{trial}_summary.json", "w") as f:
                    json.dump(result["summary"], f, indent=2)
                with open(exp_dir / f"trial_{trial}_raw.jsonl", "w") as f:
                    for r in result["raw_results"]:
                        f.write(json.dumps(r) + "\n")

                s = result["summary"]
                sanity_check_distribution(policy, s)
                print(f"    Results: P50={s['latency_ms']['p50']}ms "
                      f"P95={s['latency_ms']['p95']}ms "
                      f"cache_hit={s['vllm_metrics']['aggregate']['cache_hit_rate']:.2%} "
                      f"dist={s['instance_distribution']}")
            finally:
                stop_router(router)
                time.sleep(2)

    print("\nExperiment 2 complete.")


def experiment_3(dry_run=False):
    """Exp 3: Alpha/Beta sensitivity sweep for LoadCacheAwareScoring.

    6×6 grid, both workloads, single trial per combo (36 combos × 2 workloads).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Alpha/Beta Sensitivity Sweep")
    print("=" * 60)

    sweep_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    seed = SEEDS[0]
    rate = 5.0

    # Pre-build payloads (same for all combos — fair comparison)
    sharegpt_payloads = build_sharegpt_payloads(30, seed=seed)
    rag_payloads = build_rag_payloads(50, seed=seed)
    warmup_rag = build_rag_payloads(10, seed=seed + 1000)

    exp_dir = RESULTS_DIR / "exp3"

    for alpha in sweep_values:
        for beta in sweep_values:
            tag = f"a{alpha:.1f}_b{beta:.1f}"
            print(f"\n  alpha={alpha}, beta={beta}")

            if dry_run:
                print(f"    [DRY RUN] ShareGPT + RAG")
                continue

            exp_dir.mkdir(parents=True, exist_ok=True)
            extra_env = {"LCA_ALPHA": str(alpha), "LCA_BETA": str(beta)}

            router = start_router("load_cache_aware", extra_env=extra_env)
            try:
                # RAG workload
                result_rag = asyncio.run(run_single(
                    f"lca_{tag}", rag_payloads, warmup_rag, rate, f"rag_{tag}",
                ))
                with open(exp_dir / f"rag_{tag}.json", "w") as f:
                    json.dump(result_rag["summary"], f, indent=2)

                # ShareGPT workload
                result_sgpt = asyncio.run(run_single(
                    f"lca_{tag}", sharegpt_payloads, [], rate, f"sharegpt_{tag}",
                ))
                with open(exp_dir / f"sharegpt_{tag}.json", "w") as f:
                    json.dump(result_sgpt["summary"], f, indent=2)

                rs = result_rag["summary"]
                print(f"    RAG: P50={rs['latency_ms']['p50']}ms "
                      f"cache={rs['vllm_metrics']['aggregate']['cache_hit_rate']:.2%} "
                      f"dist={rs['instance_distribution']}")
            finally:
                stop_router(router)
                time.sleep(1)

    print("\nExperiment 3 complete.")


def experiment_4(dry_run=False):
    """Exp 4: Directory staleness sensitivity.

    Vary cache TTL and measure cache hit rate under load_cache_aware.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Cache Directory Staleness")
    print("=" * 60)

    intervals = [0.0, 0.1, 0.5, 1.0, 5.0, 30.0]
    seed = SEEDS[0]
    rate = 5.0

    sharegpt_payloads = build_sharegpt_payloads(30, seed=seed)
    warmup = build_sharegpt_payloads(10, seed=seed + 1000)
    exp_dir = RESULTS_DIR / "exp4"

    for interval in intervals:
        print(f"\n  Staleness: {interval}s")

        if dry_run:
            print(f"    [DRY RUN] {len(sharegpt_payloads)} requests")
            continue

        exp_dir.mkdir(parents=True, exist_ok=True)
        extra_env = {"CACHE_STALENESS_SEC": str(interval)}

        router = start_router("load_cache_aware", extra_env=extra_env)
        try:
            result = asyncio.run(run_single(
                f"staleness_{interval}s", sharegpt_payloads, warmup, rate,
                f"staleness_{interval}",
            ))
            with open(exp_dir / f"staleness_{interval:.1f}s.json", "w") as f:
                json.dump(result["summary"], f, indent=2)

            s = result["summary"]
            print(f"    P50={s['latency_ms']['p50']}ms "
                  f"cache={s['vllm_metrics']['aggregate']['cache_hit_rate']:.2%} "
                  f"dist={s['instance_distribution']}")
        finally:
            stop_router(router)
            time.sleep(1)

    print("\nExperiment 4 complete.")


def experiment_5(dry_run=False):
    """Exp 5: Scaling simulation (N=2, 4, 8, 16).

    Requires simulator module (built from calibration data).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Scaling Simulation")
    print("=" * 60)
    print("  NOTE: Requires simulator module. Skipping for now.")
    print("  Will be built after calibration data from Exp 1-2.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global ROUTER_PORT, ROUTER_URL, FORCE_CLAIM_PORT

    parser = argparse.ArgumentParser(description="Run KV Cache Routing experiments")
    parser.add_argument("--exp", nargs="+", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--router-port",
        type=int,
        default=9000,
        help="Port for experiment-managed router (default: 9000)",
    )
    parser.add_argument(
        "--force-claim-port",
        action="store_true",
        help="Terminate existing listeners on --router-port before starting experiments",
    )
    args = parser.parse_args()

    if not args.all and not args.exp:
        parser.print_help()
        sys.exit(1)

    experiments = {
        1: experiment_1,
        2: experiment_2,
        3: experiment_3,
        4: experiment_4,
        5: experiment_5,
    }

    to_run = list(experiments.keys()) if args.all else args.exp

    ROUTER_PORT = args.router_port
    ROUTER_URL = f"http://localhost:{ROUTER_PORT}"
    FORCE_CLAIM_PORT = args.force_claim_port

    print(f"{'=' * 60}")
    print(f"KV Cache Routing — Experiment Runner")
    print(f"{'=' * 60}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Experiments: {to_run}")
    print(f"Model: {MODEL}")
    print(f"Trials per config: {N_TRIALS}")
    print(f"Seeds: {SEEDS}")
    print(f"Router: {ROUTER_URL}")
    print(f"Force claim port: {FORCE_CLAIM_PORT}")
    print(f"vLLM instances: {VLLM_INSTANCES}")
    if not SHAREGPT_PATH.exists():
        print(f"WARNING: ShareGPT dataset not found at {SHAREGPT_PATH}")

    for exp_num in sorted(to_run):
        experiments[exp_num](dry_run=args.dry_run)

    print(f"\n{'=' * 60}")
    print(f"ALL DONE. Results: {RESULTS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
