import os
import json
from pathlib import Path
import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity

from FlagEmbedding import BGEM3FlagModel


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _map_operator_to_architecture(op_name: str) -> str:
    name = op_name.lower()
    if "scaled_dot_product" in name or "softmax" in name:
        return "Self-Attention (scores/probabilities)"
    if any(token in name for token in ("addmm", "matmul", "mm", "bmm", "linear")):
        return "Linear projections / FFN matmuls"
    if any(token in name for token in ("gelu", "relu", "silu")):
        return "FFN activation"
    if "layer_norm" in name or "rms_norm" in name:
        return "Normalization"
    if "embedding" in name or "index_select" in name:
        return "Token embedding / lookup"
    if name.endswith("::add") or "add." in name or "::add_" in name:
        return "Residual connections / bias add"
    if any(
        token in name
        for token in (
            "view",
            "reshape",
            "transpose",
            "permute",
            "expand",
            "contiguous",
            "as_strided",
            "slice",
            "cat",
            "split",
            "unsqueeze",
            "squeeze",
        )
    ):
        return "Tensor layout/shape ops"
    return "Other / auxiliary"


def _write_ops_architecture_report(events, output_path: Path):
    lines = [
        "# Operator-to-architecture mapping",
        "",
        "| Operator | Calls | Self CPU (ms) | Architecture block |",
        "|---|---:|---:|---|",
    ]
    for event in events:
        lines.append(
            f"| `{event['name']}` | {event['calls']} | {event['self_cpu_time_ms']:.4f} | "
            f"{_map_operator_to_architecture(event['name'])} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_profiler_outputs(prof):
    output_dir = Path(os.getenv("M3_PROFILER_DIR", "/tmp"))
    prefix = os.getenv("M3_PROFILER_PREFIX", "m3_single_device")
    top_n = int(os.getenv("M3_PROFILER_TOP_N", "30"))
    export_trace = _env_flag("M3_PROFILER_TRACE", default=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=200)
    table_path = output_dir / f"{prefix}.table.txt"
    table_path.write_text(table + "\n", encoding="utf-8")

    events = []
    for evt in prof.key_averages():
        events.append(
            {
                "name": evt.key,
                "calls": int(evt.count),
                "self_cpu_time_ms": float(evt.self_cpu_time_total) / 1000.0,
                "cpu_total_time_ms": float(evt.cpu_time_total) / 1000.0,
                "cpu_mem_bytes": int(getattr(evt, "cpu_memory_usage", 0)),
                "self_cpu_mem_bytes": int(getattr(evt, "self_cpu_memory_usage", 0)),
            }
        )
    events.sort(key=lambda x: x["self_cpu_time_ms"], reverse=True)
    payload = {
        "top_events": events[:top_n],
        "self_cpu_time_total_ms": sum(e["self_cpu_time_ms"] for e in events),
    }
    summary_path = output_dir / f"{prefix}.summary.json"
    ops_map_path = Path(os.getenv("M3_PROFILER_OPS_MAP", str(output_dir / f"{prefix}.ops_architecture.md")))
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_ops_architecture_report(events, ops_map_path)
    print(f"PROFILER_TABLE {table_path}", flush=True)
    print(f"PROFILER_SUMMARY {summary_path}", flush=True)
    print(f"PROFILER_OPS_MAP {ops_map_path}", flush=True)

    if export_trace:
        trace_path = output_dir / f"{prefix}.trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"PROFILER_TRACE {trace_path}", flush=True)


def register_linear_hooks(model):
    execution_state = {'index': 0}
    active_hooks = {}
    handles = []

    def tensor_summary(value):
        if isinstance(value, torch.Tensor):
            return f"shape={tuple(value.shape)}, dtype={value.dtype}"
        return type(value).__name__

    def create_hooks(name):
        def pre_hook(module, inputs):
            execution_state['index'] += 1
            hook_id = execution_state['index']
            active_hooks[id(module)] = hook_id
            input_summary = tensor_summary(inputs[0]) if inputs else 'no-input'
            print(
                f"HOOK_BEGIN {hook_id}: {name} | input={input_summary}",
                flush=True,
            )

        def post_hook(module, inputs, output):
            hook_id = active_hooks.pop(id(module))
            output_summary = tensor_summary(output)
            print(
                f"HOOK_END {hook_id}: {name} | output={output_summary}",
                flush=True,
            )

        return pre_hook, post_hook

    for name, module in model.model.named_modules():
        if isinstance(module, nn.Linear):
            pre_hook, post_hook = create_hooks(name)
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(post_hook))

    return handles


def test_m3_single_device():
    precision = os.getenv('M3_PRECISION', 'bf16').lower()
    precision_flags = {
        'fp32': (False, False),
        'bf16': (False, True),
        'fp16': (True, False),
    }
    if precision not in precision_flags:
        raise ValueError(
            f"Unsupported M3_PRECISION={precision!r}. "
            "Use fp32, bf16, or fp16."
        )

    use_fp16, use_bf16 = precision_flags[precision]
    print(f"PRECISION_BEGIN {precision}", flush=True)
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
        devices="cpu",   # if you don't have a GPU, you can use "cpu"
        pooling_method='cls',
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        cache_dir=os.getenv('HF_HUB_CACHE', None),
    )

    hook_handles = register_linear_hooks(model)
    
    queries = ["What is BGE M3?"]
    passages = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."
    ]
    
    def run_inference():
        print("PHASE_BEGIN query", flush=True)
        query_embeddings = model.encode_queries(
            queries,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        print("PHASE_END query", flush=True)

        print("PHASE_BEGIN passage", flush=True)
        passage_embeddings = model.encode_corpus(
            passages,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        print("PHASE_END passage", flush=True)
        return query_embeddings, passage_embeddings

    if _env_flag("M3_TORCH_PROFILE", default=False):
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            queries_embeddings, passages_embeddings = run_inference()
        write_profiler_outputs(prof)
    else:
        queries_embeddings, passages_embeddings = run_inference()
    
    dense_scores = queries_embeddings["dense_vecs"] @ passages_embeddings["dense_vecs"].T
    sparse_scores = model.compute_lexical_matching_score(
        queries_embeddings["lexical_weights"],
        passages_embeddings["lexical_weights"],
    )

    print("Dense shape:", queries_embeddings["dense_vecs"].shape)
    print("Sparse entries:", len(queries_embeddings["lexical_weights"]))
    print("ColBERT sequences:", len(queries_embeddings["colbert_vecs"]))
    print("First ColBERT shape:", queries_embeddings["colbert_vecs"][0].shape)
    print("Dense score:", dense_scores[0, 0])
    print("Sparse score:", sparse_scores[0, 0])

    for handle in hook_handles:
        handle.remove()

    print(f"PRECISION_END {precision}", flush=True)


if __name__ == '__main__':
    test_m3_single_device()
