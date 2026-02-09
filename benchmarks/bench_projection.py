"""
Benchmark: DistanceProjection effect on SWNN parameter count and forward pass time.

Demonstrates that projection decouples knn_k from hidden layer structure:
different knn_k values with same embed_dim produce identical hidden layers.

Usage:
    python benchmarks/bench_projection.py
"""

import time

import torch

from gnnwr.networks import SWNN


KNN_K_VALUES = [50, 100, 200, 500, 1000]
EMBED_DIM = 128
OUTSIZE = 7
BATCH_SIZE = 64
WARMUP = 3
REPEATS = 10


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def time_forward(model, x, warmup=WARMUP, repeats=REPEATS):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x)
        return (time.perf_counter() - t0) / repeats


def main():
    print("=" * 80)
    print("DistanceProjection Benchmark")
    print(f"embed_dim={EMBED_DIM}, outsize={OUTSIZE}, batch_size={BATCH_SIZE}")
    print("=" * 80)

    print(f"\n### Without projection (knn_k directly feeds into hidden layers)\n")
    print(f"| {'knn_k':>6} | {'params':>10} | {'hidden layers':>20} | {'forward (ms)':>12} |")
    print(f"|{'-'*8}|{'-'*12}|{'-'*22}|{'-'*14}|")

    for k in KNN_K_VALUES:
        swnn = SWNN(insize=k, outsize=OUTSIZE)
        x = torch.randn(BATCH_SIZE, k)
        params = count_params(swnn)
        t = time_forward(swnn, x) * 1000
        layers = swnn.dense_layer
        print(f"| {k:>6} | {params:>10,} | {str(layers):>20} | {t:>10.2f}  |")

    print(f"\n### With projection (knn_k → embed_dim={EMBED_DIM} → hidden layers)\n")
    print(f"| {'knn_k':>6} | {'params':>10} | {'hidden layers':>20} | {'forward (ms)':>12} | {'proj params':>11} |")
    print(f"|{'-'*8}|{'-'*12}|{'-'*22}|{'-'*14}|{'-'*13}|")

    ref_layers = None
    for k in KNN_K_VALUES:
        swnn = SWNN(insize=k, outsize=OUTSIZE, embed_dim=EMBED_DIM)
        x = torch.randn(BATCH_SIZE, k)
        params = count_params(swnn)
        t = time_forward(swnn, x) * 1000
        layers = swnn.dense_layer
        proj_params = count_params(swnn.projection) if swnn.projection else 0

        if ref_layers is None:
            ref_layers = layers
        layers_match = "same" if layers == ref_layers else "DIFFERENT"

        print(f"| {k:>6} | {params:>10,} | {str(layers):>20} | {t:>10.2f}  | {proj_params:>9,}  |")

    # Verify decoupling
    print(f"\n### Decoupling verification\n")
    all_layers = []
    for k in KNN_K_VALUES:
        swnn = SWNN(insize=k, outsize=OUTSIZE, embed_dim=EMBED_DIM)
        all_layers.append(swnn.dense_layer)

    # Filter: only projected ones (knn_k > embed_dim)
    projected = [(k, l) for k, l in zip(KNN_K_VALUES, all_layers) if k > EMBED_DIM]
    if projected:
        first_layers = projected[0][1]
        all_same = all(l == first_layers for _, l in projected)
        print(f"Hidden layers for projected knn_k values ({[k for k,_ in projected]}):")
        print(f"  {first_layers}")
        print(f"  All identical: {all_same}")
    else:
        print("No knn_k values exceed embed_dim, no projection needed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
