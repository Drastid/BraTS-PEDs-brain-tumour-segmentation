"""
tests/test_models3d.py
========================
Smoke test per src/models3d.py (roadmap §9): verifica che le tre architetture
3D costruiscano correttamente e producano output della shape attesa su un
input fittizio [B, 4, 128, 128, 128] -> [B, 5, 128, 128, 128].

Esecuzione locale su CPU (no CUDA) — solo verifica strutturale, non prestazionale.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models3d import ARCH_NAMES, build_model_3d, load_pretrained_3d

ROI = (128, 128, 128)
IN_CHANNELS = 4
NUM_CLASSES = 5


@pytest.mark.parametrize("arch", ARCH_NAMES)
def test_build_and_forward(arch: str) -> None:
    model = build_model_3d(arch, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, roi=ROI)
    model.eval()

    x = torch.randn(1, IN_CHANNELS, *ROI)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, NUM_CLASSES, *ROI), (
        f"[{arch}] output shape {tuple(y.shape)} != atteso "
        f"{(1, NUM_CLASSES, *ROI)}"
    )


def test_invalid_arch_raises() -> None:
    with pytest.raises(ValueError):
        build_model_3d("not-a-real-arch")


# ---------------------------------------------------------------------------
# load_pretrained_3d — checkpoint sintetici che replicano i pattern di naming
# reali trovati in weights/model_swinvit.pt (SSL NVIDIA) e
# weights/model_best_fold_0.pth (HuggingFace/BrainSegFounder). I file veri
# sono grandi binari esclusi da git (.gitignore): qui si ricostruisce solo la
# struttura di chiavi/shape rilevante, cosi' il test resta eseguibile senza
# quei file e protegge la logica di remap da regressioni.
# ---------------------------------------------------------------------------


def test_load_pretrained_nvidia_ssl_style_checkpoint(tmp_path) -> None:
    """Checkpoint SSL NVIDIA reale: encoder-only, prefisso 'module.', naming
    MLP legacy 'fc1'/'fc2' (rinominati in MONAI 1.6 a 'linear1'/'linear2'), e
    patch_embed a 1 canale di input (pretraining single-modality) mentre qui
    servono 4 modalita' — quel layer deve restare shape-mismatch e quindi
    NON caricato, il resto dell'encoder swinViT si'."""
    model = build_model_3d("segformer", in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model_sd = model.state_dict()

    fake_sd = {}
    for k, v in model_sd.items():
        if not k.startswith("swinViT."):
            continue  # checkpoint SSL non contiene decoder/head
        inner = k[len("swinViT."):]
        inner = inner.replace(".mlp.linear1.", ".mlp.fc1.").replace(".mlp.linear2.", ".mlp.fc2.")
        ckpt_key = "module." + inner
        if k == "swinViT.patch_embed.proj.weight":
            fake_sd[ckpt_key] = torch.randn(v.shape[0], 1, *v.shape[2:])  # 1 canale, non 4
        else:
            fake_sd[ckpt_key] = torch.rand_like(v.float()).to(v.dtype)

    ckpt_path = tmp_path / "fake_swinvit.pt"
    torch.save({"epoch": 100, "state_dict": fake_sd}, ckpt_path)

    model, unmatched = load_pretrained_3d(model, str(ckpt_path), verbose=False)
    loaded_sd = model.state_dict()

    swinvit_keys = [k for k in model_sd if k.startswith("swinViT.") and k != "swinViT.patch_embed.proj.weight"]
    for k in swinvit_keys:
        fake_key = "module." + k[len("swinViT."):].replace(".mlp.linear1.", ".mlp.fc1.").replace(".mlp.linear2.", ".mlp.fc2.")
        assert torch.equal(loaded_sd[k], fake_sd[fake_key]), f"{k} non caricato correttamente"

    # Il layer a canali incompatibili (1 nel ckpt vs 4 nel modello) resta con
    # l'init originale: la sua shape e' quella del modello (4 canali), non
    # quella incompatibile del checkpoint (1 canale) — prova che NON e' stato
    # sovrascritto dal tensore shape-mismatched.
    assert loaded_sd["swinViT.patch_embed.proj.weight"].shape == model_sd["swinViT.patch_embed.proj.weight"].shape

    # unmatched_param_names deve contenere ESATTAMENTE i parametri fuori
    # dall'encoder swinViT (decoder+head, mai nel checkpoint SSL) PIU'
    # patch_embed.proj.weight (unico shape mismatch: il bias, shape (48,),
    # non dipende dal numero di canali di input e combacia regolarmente).
    expected_unmatched = {k for k in model_sd if not k.startswith("swinViT.")}
    expected_unmatched.add("swinViT.patch_embed.proj.weight")
    assert set(unmatched) == expected_unmatched


def test_load_pretrained_huggingface_style_checkpoint(tmp_path) -> None:
    """Checkpoint HuggingFace/BrainSegFounder reale: encoder+decoder completi,
    prefisso 'net.', head finale a 3 canali (TC/WT/ET adulti) mentre qui
    servono 5 classi pediatriche — solo la head deve restare esclusa, tutto
    il resto (encoder+decoder) deve caricare."""
    model = build_model_3d("segformer", in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model_sd = model.state_dict()

    fake_sd = {}
    for k, v in model_sd.items():
        ckpt_key = "net." + k
        if k == "out.conv.conv.weight":
            fake_sd[ckpt_key] = torch.randn(3, *v.shape[1:])  # 3 classi adulte, non 5
        elif k == "out.conv.conv.bias":
            fake_sd[ckpt_key] = torch.randn(3)
        else:
            fake_sd[ckpt_key] = torch.rand_like(v.float()).to(v.dtype)

    ckpt_path = tmp_path / "fake_best_fold_0.pth"
    torch.save(
        {"epoch": 99, "state_dict": fake_sd, "best_acc": __import__("numpy").float32(0.87)},
        ckpt_path,
    )

    model, unmatched = load_pretrained_3d(model, str(ckpt_path), verbose=False)
    loaded_sd = model.state_dict()

    for k in model_sd:
        if k in ("out.conv.conv.weight", "out.conv.conv.bias"):
            continue
        assert torch.equal(loaded_sd[k], fake_sd["net." + k]), f"{k} non caricato correttamente"

    # La head resta con l'init originale del costruttore (shape mismatch 3 vs 5)
    assert loaded_sd["out.conv.conv.weight"].shape == (NUM_CLASSES, 48, 1, 1, 1)

    # unmatched_param_names deve contenere ESATTAMENTE la head strutturale
    # (out.conv.conv.{weight,bias}) — qui il mismatch coincide col modulo
    # "head" gia' riconosciuto da src.optim_3d, quindi extra_head_param_names
    # e' ridondante ma corretto per questo checkpoint.
    assert set(unmatched) == {"out.conv.conv.weight", "out.conv.conv.bias"}


if __name__ == "__main__":
    for arch in ARCH_NAMES:
        print(f"--- {arch} ---")
        test_build_and_forward(arch)
        print(f"  OK: output shape corretta.")
    test_invalid_arch_raises()
    print("Tutti i test superati.")
