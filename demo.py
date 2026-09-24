import torch

from mrhhgnn import (
    HHGNN,
    MODALITIES,
    NUM_RELATIONS,
    add_self_loops,
)

T = 64
D = 256
D_R = 256
NUM_HEADS = 4
NUM_LAYERS = 2
K_S = 13


def random_adjacency(T: int, K: int, device: str) -> dict:
    scores = torch.rand(T, T, device=device)
    scores.fill_diagonal_(-1.0)
    idx = scores.topk(K, dim=1).indices
    adj_s = torch.zeros(T, T, device=device)
    adj_s.scatter_(1, idx, 1.0)
    adj_s = torch.clamp(adj_s + adj_s.T, max=1.0)

    adj_t = torch.zeros(T, T, device=device)
    for i in range(T - 1):
        adj_t[i, i + 1] = 1.0
        adj_t[i + 1, i] = 1.0

    return {
        "spatial": {m: add_self_loops(adj_s) for m in MODALITIES},
        "temporal": {m: add_self_loops(adj_t) for m in MODALITIES},
    }


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    embeddings = {m: torch.randn(T, D, device=device) for m in MODALITIES}
    relation_all = torch.randn(NUM_RELATIONS, D_R, device=device,
                               requires_grad=True)
    adjacency = random_adjacency(T, K_S, device)

    model = HHGNN(d=D, d_r=D_R, num_heads=NUM_HEADS,
                  num_layers=NUM_LAYERS).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {n_params:,}")
    for i, layer in enumerate(model.layers):
        sub = [("node-level (spatial)", layer.node_agg_spatial),
               ("node-level (temporal)", layer.node_agg_temporal),
               ("path-level", layer.path_agg),
               ("cross-domain", layer.cross_modal_agg)]
        print(f"  layer {i + 1}")
        for name, mod in sub:
            print(f"    {name:<24s} {sum(p.numel() for p in mod.parameters()):>10,}")

    model.train()
    z_hat = model(embeddings, adjacency, relation_all)

    print(f"\ninput   " + "  ".join(f"{m} {tuple(embeddings[m].shape)}"
                                    for m in MODALITIES))
    print(f"        relation embeddings {tuple(relation_all.shape)}")
    print(f"        adjacency  spatial {tuple(adjacency['spatial']['time'].shape)}"
          f"  temporal {tuple(adjacency['temporal']['time'].shape)}")
    print("output  " + "  ".join(f"{m} {tuple(z_hat[m].shape)}" for m in MODALITIES))

    loss = sum(z_hat[m].pow(2).mean() for m in MODALITIES)
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters without a gradient: {missing}"
    grad_norm = torch.norm(
        torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
    )
    print(f"\nglobal gradient norm = {grad_norm.item():.4f}")
    print("relation embedding gradient norm = "
          f"{relation_all.grad.norm().item():.4f}")
    print("\nevery parameter received a gradient - forward and backward pass OK")


if __name__ == "__main__":
    main()
