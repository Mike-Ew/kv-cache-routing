# Research Notes

## 1. PagedAttention / vLLM (SOSP 2023)

- **File**: `papers/paged_attention.pdf`
- **Citation**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.
- **Key idea**: Uses OS-style virtual memory paging for KV cache — blocks are non-contiguous, enabling near-zero waste and efficient sharing.
- **What we borrow**: vLLM as our serving backend; its prefix caching (APC) is the cache we route to.
- **What we do differently**: We operate at the gateway level without modifying the engine.
- **Useful figures/metrics**: KV cache utilization, memory fragmentation reduction.

## 2. Mooncake (FAST 2025)

- **File**: `papers/mooncake.pdf`
- **Citation**: Qin et al., "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving," FAST 2025.
- **Key idea**: Disaggregates KV cache from compute — stores caches in a distributed pool, enabling cross-instance cache sharing.
- **What we borrow**: Motivation for cache-aware routing; their analysis of prefill cost reduction.
- **What we do differently**: We route to existing per-instance caches rather than disaggregating them.
- **Useful figures/metrics**: Prefill latency vs. cache hit rate tradeoffs.

## 3. CachedAttention (USENIX ATC 2024)

- **File**: `papers/cached_attention.pdf`
- **Citation**: Gao et al., "Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention," USENIX ATC 2024.
- **Key idea**: Reuses KV caches across multi-turn conversations by storing them between requests.
- **What we borrow**: Multi-turn conversation as a primary workload; their analysis of cache benefit in multi-turn scenarios.
- **What we do differently**: We focus on routing decisions rather than cache storage mechanisms.
- **Useful figures/metrics**: TTFT improvement from cache reuse in conversation settings.

## 4. DualMap (arXiv 2025)

- **File**: `papers/dualmap.pdf`
- **Citation**: Zhu et al., "DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving," arXiv 2025.
- **Key idea**: Uses two mapping layers (cache affinity map + load balance map) to achieve both goals simultaneously.
- **What we borrow**: The cache-vs-load tradeoff framing; their taxonomy of routing strategies.
- **What we do differently**: We evaluate simpler gateway-level policies without engine-level integration.
- **Useful figures/metrics**: Cache hit rate vs. load imbalance Pareto curves.

## 5. SGLang (NeurIPS 2024)

- **File**: `papers/sglang.pdf`
- **Citation**: Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," NeurIPS 2024.
- **Key idea**: RadixAttention — uses a radix tree for prefix caching, enabling automatic prefix sharing across requests.
- **What we borrow**: Motivation for prefix-aware routing; their analysis of prefix sharing patterns.
- **What we do differently**: We use vLLM's simpler APC rather than RadixAttention, and operate at gateway level.
- **Useful figures/metrics**: Prefix sharing rates in real workloads.

## 6. Splitwise (ISCA 2024)

- **Citation**: Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024.
- **Key idea**: Separates prefill and decode phases onto different hardware, optimizing each independently.
- **What we borrow**: Understanding of prefill cost as a dominant factor in LLM serving latency.
- **What we do differently**: We keep prefill and decode co-located; optimize through cache-aware routing instead.
- **Useful figures/metrics**: Prefill vs. decode latency breakdown.

## 7. DistServe (OSDI 2024)

- **Citation**: Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving," OSDI 2024.
- **Key idea**: Disaggregates prefill and decoding to different GPU pools for goodput optimization.
- **What we borrow**: Their analysis of how prefill interference affects tail latency.
- **What we do differently**: We address prefill cost through cache reuse rather than disaggregation.
- **Useful figures/metrics**: Goodput vs. latency SLA curves.

## 8. Consistent Hashing (STOC 1997)

- **Citation**: Karger et al., "Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web," STOC 1997.
- **Key idea**: Hash-based assignment of keys to servers that minimizes reassignments when servers change.
- **What we borrow**: Session affinity hashing as a baseline routing mechanism.
- **What we do differently**: We extend beyond simple hashing to cache-content-aware scoring.
- **Useful figures/metrics**: Load distribution uniformity.

## 9. Power of Two Choices (IEEE TPDS 2001)

- **Citation**: Mitzenmacher, "The Power of Two Choices in Randomized Load Balancing," IEEE TPDS 2001.
- **Key idea**: Sampling just 2 random servers and picking the less loaded one achieves exponential improvement over random assignment.
- **What we borrow**: P2C as one of our load-only baselines.
- **What we do differently**: We show that load-only P2C misses cache locality opportunities.
- **Useful figures/metrics**: Queue length distribution analysis.

## 10. Federated Learning / Communication-Efficient Learning (AISTATS 2017)

- **File**: `papers/Communication - Efficient Learning of Deep Networks from Decentralized Data.pdf`
- **Citation**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017.
- **Key idea**: Training models across decentralized data sources while keeping data local.
- **What we borrow**: Concepts of decentralized decision-making and local state management in distributed systems.
- **What we do differently**: We apply decentralized state awareness (cache state) to routing rather than training.
- **Useful figures/metrics**: Communication cost reduction in distributed settings.

## 11. Preble — Efficient Distributed Prompt Scheduling (2024)

- **Citation**: Preble, "Efficient Distributed Prompt Scheduling for LLM Serving," 2024.
- **Key idea**: Session/prefix-aware prompt scheduling that considers cached state when distributing requests.
- **What we borrow**: Prior art for cache-aware routing motivation.
- **What we do differently**: We provide an apples-to-apples comparison at the gateway level without custom scheduler modifications.
- **Useful figures/metrics**: Prompt reuse rates and scheduling overhead.
