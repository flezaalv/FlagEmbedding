import os
import torch
from torch import nn

from FlagEmbedding import BGEM3FlagModel


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
    
    print("PHASE_BEGIN query", flush=True)
    queries_embeddings = model.encode_queries(
        queries,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    print("PHASE_END query", flush=True)

    print("PHASE_BEGIN passage", flush=True)
    passages_embeddings = model.encode_corpus(
        passages,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    print("PHASE_END passage", flush=True)
    
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
