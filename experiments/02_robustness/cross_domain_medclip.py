"""
BiomedCLIP control experiment.
Domain-specialized CLIP (medical pretraining) vs general CLIP (LAION).
Addresses shared pretraining bias concern.
"""

import json, sys, time, torch, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import medmnist
from medmnist import INFO

SRC = Path(__file__).resolve().parent
PROJ = SRC.parent.parent
OUT = PROJ / "results" / "02_robustness" / "cross_domain_medclip"
OUT.mkdir(parents=True, exist_ok=True)

MED_PROMPTS = [
    "this is a photo of {}",
]

CLASS_NAMES = {
    "pathmnist": [
        "adipose tissue", "background", "debris", "lymphocytes",
        "mucus", "smooth muscle", "normal colon mucosa",
        "cancer-associated stroma", "tumor epithelium",
    ],
    "bloodmnist": [
        "basophil", "eosinophil", "erythroblast", "immature granulocyte",
        "lymphocyte", "monocyte", "neutrophil", "platelet",
    ],
    "tissuemnist": [
        "collecting duct", "connecting tubule",
        "distal convoluted tubule", "glomerular endothelial cells",
        "interstitial endothelial cells", "leukocytes",
        "podocytes", "proximal tubule segments",
    ],
}

# MLLM mean per-class accuracy (averaged across Qwen2-VL-7B + LLaVA-1.5-7B)
MLLM_MEAN = {
    "pathmnist": [46.0, 46.0, 0.0, 0.0, 0.0, 0.0, 26.0, 20.0, 0.0],
    "bloodmnist": [10.0, 90.0, 4.0, 0.0, 23.0, 0.0, 6.0, 1.0],
    "tissuemnist": [31.0, 3.0, 2.0, 0.0, 0.0, 0.0, 41.0, 6.0],
}

# General CLIP (LAION ViT-L/14) per-class AnchorScore
GENERAL_ANCHOR = {
    "pathmnist": [1.12, 0.47, 0.59, 0.0, 5.22, 20.78, 33.47, 0.0, 57.26],
    "bloodmnist": [44.26, 0.48, 1.61, 2.42, 9.05, 2.11, 48.5, 0.0],
    "tissuemnist": [15.0, 7.5, 3.5, 0.0, 0.0, 47.0, 26.5, 0.0],
}


def load_data(ds_name):
    info = INFO[ds_name]
    DataClass = getattr(medmnist, info["python_class"])
    data = DataClass(split="test", download=True,
                     root=str(Path.home() / ".medmnist"))
    imgs = np.asarray(data.imgs)
    labels = np.asarray(data.labels).flatten()
    return imgs, labels, len(info["label"])


@torch.no_grad()
def compute_anchor_biomed(model, tokenizer, preprocess, device,
                          imgs, labels, n_classes, class_names):
    context_length = 256
    prompt = MED_PROMPTS[0]
    texts = tokenizer(
        [prompt.format(c) for c in class_names],
        context_length=context_length
    ).to(device)

    model.eval()
    text_features = model.encode_text(texts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct = np.zeros(n_classes, dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.float32)
    n = len(imgs)

    for i in range(0, n, 64):
        batch_imgs = imgs[i:i+64]
        batch_labels = labels[i:i+64]

        imgs_pil = []
        for arr in batch_imgs:
            from PIL import Image
            pil = Image.fromarray(arr).convert("RGB")
            imgs_pil.append(preprocess(pil))

        images = torch.stack(imgs_pil).to(device)

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = (100.0 * image_features) @ text_features.T
        preds = logits.argmax(dim=1).cpu().numpy()

        for c in range(n_classes):
            mask = batch_labels == c
            counts[c] += mask.sum()
            correct[c] += (preds[mask] == c).sum()

    acc = np.where(counts > 0, correct / counts * 100, 0.0)
    return acc.tolist()


def main():
    # Set HF_ENDPOINT if using a mirror (e.g., https://hf-mirror.com)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading BiomedCLIP from hf-hub...")
    from open_clip import create_model_from_pretrained, get_tokenizer
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model = model.to(device)
    print("BiomedCLIP loaded.")

    results = {}
    all_biomed = []
    all_general = []
    all_mllm = []

    for ds_name in ["pathmnist", "bloodmnist", "tissuemnist"]:
        print(f"\n=== {ds_name} ===")
        imgs, labels, n_classes = load_data(ds_name)
        class_names = CLASS_NAMES[ds_name]
        print(f"  Samples: {len(imgs)}, Classes: {n_classes}")

        t0 = time.time()
        biomed_acc = compute_anchor_biomed(
            model, tokenizer, preprocess, device,
            imgs, labels, n_classes, class_names
        )
        elapsed = time.time() - t0
        print(f"  BiomedCLIP Anchor: {np.mean(biomed_acc):.1f}% ({elapsed:.0f}s)")

        results[ds_name] = {
            "biomedclip_anchor": biomed_acc,
            "general_clip_anchor": GENERAL_ANCHOR[ds_name],
            "mllm_mean_acc": MLLM_MEAN[ds_name],
            "class_names": class_names,
        }

        valid = [i for i, v in enumerate(MLLM_MEAN[ds_name]) if not (v == 0 and all(
            x == 0 for x in [biomed_acc[i], GENERAL_ANCHOR[ds_name][i]]))]

        if len(valid) >= 4:
            r_bm, p_bm = spearmanr(
                [biomed_acc[i] for i in valid],
                [MLLM_MEAN[ds_name][i] for i in valid]
            )
            r_gn, p_gn = spearmanr(
                [GENERAL_ANCHOR[ds_name][i] for i in valid],
                [MLLM_MEAN[ds_name][i] for i in valid]
            )
        else:
            r_bm, p_bm = 0.0, 1.0
            r_gn, p_gn = 0.0, 1.0

        results[ds_name]["biomedclip_rho"] = round(r_bm, 3)
        results[ds_name]["biomedclip_p"] = round(p_bm, 4)
        results[ds_name]["general_clip_rho"] = round(r_gn, 3)
        results[ds_name]["general_clip_p"] = round(p_gn, 4)

        all_biomed.extend(biomed_acc)
        all_general.extend(GENERAL_ANCHOR[ds_name])
        all_mllm.extend(MLLM_MEAN[ds_name])

        print(f"  BiomedCLIP ρ={r_bm:.3f} p={p_bm:.4f}")
        print(f"  General CLIP ρ={r_gn:.3f} p={p_gn:.4f}")

    # Pooled (all 25 classes)
    r_pool_bm, p_pool_bm = spearmanr(all_biomed, all_mllm)
    r_pool_gn, p_pool_gn = spearmanr(all_general, all_mllm)
    results["pooled_medical"] = {
        "biomedclip_rho": round(r_pool_bm, 3),
        "biomedclip_p": round(p_pool_bm, 4),
        "general_clip_rho": round(r_pool_gn, 3),
        "general_clip_p": round(p_pool_gn, 4),
        "n_classes": len(all_mllm),
    }
    print(f"\n=== Pooled (n={len(all_mllm)}) ===")
    print(f"  BiomedCLIP ρ={r_pool_bm:.3f} p={p_pool_bm:.4f}")
    print(f"  General CLIP ρ={r_pool_gn:.3f} p={p_pool_gn:.4f}")

    out_path = OUT / "cross_domain_medclip_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    import os
    main()
