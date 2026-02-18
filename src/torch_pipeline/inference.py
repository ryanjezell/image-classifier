import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
from PIL import Image

from src.torch_pipeline.artifacts import load_inference_artifacts
from src.torch_pipeline.model_factory import create_model
from src.torch_pipeline.transforms import build_inference_transform


class TorchImageClassifier:
    def __init__(self, model_path: str, cpu_only: bool = False):
        _, model_state_path, classes, inference_cfg = load_inference_artifacts(model_path)
        self.classes = classes
        self.device = torch.device("cpu" if cpu_only or not torch.cuda.is_available() else "cuda")

        self.model = create_model(
            architecture=inference_cfg["architecture"],
            num_classes=len(classes),
            pretrained=False,
        ).to(self.device)
        state_dict = torch.load(model_state_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        norm = inference_cfg.get("normalization", {})
        self.transform = build_inference_transform(
            image_size=int(inference_cfg["image_size"]),
            mean=norm.get("mean"),
            std=norm.get("std"),
        )
        logging.info("[CLASSIFIER] Ready — %d classes", len(self.classes))

    @torch.inference_mode()
    def predict(self, image_path: Union[str, Path]) -> Dict:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: '{path}'")

        image = Image.open(path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        idx = int(probs.argmax().item())

        probs_dict = dict(
            sorted(
                {cls: round(float(p), 6) for cls, p in zip(self.classes, probs)}.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )
        label = self.classes[idx]
        return {
            "image_path": str(path),
            "label": label,
            "confidence": round(float(probs[idx]), 6),
            "all_probs": probs_dict,
            "top_3": list(probs_dict.items())[:3],
        }

    def predict_batch(self, paths: List[Union[str, Path]], verbose: bool = True) -> List[Dict]:
        results = []
        for i, p in enumerate(paths):
            if verbose and i % 10 == 0:
                logging.info("[BATCH] %d/%d", i + 1, len(paths))
            try:
                results.append(self.predict(p))
            except Exception as e:
                logging.error("[BATCH] Failed on '%s': %s", p, e)
                results.append({"image_path": str(p), "error": str(e), "label": None, "confidence": None, "all_probs": None})
        return results

    def predict_folder(self, folder: Union[str, Path]) -> List[Dict]:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: '{folder}'")
        files = sorted(
            f for f in folder.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        )
        if not files:
            raise ValueError(f"No images found in '{folder}'")
        return self.predict_batch(files)

    def print_result(self, r: Dict) -> None:
        if "error" in r:
            print(f"\n❌  {r['image_path']}  →  ERROR: {r['error']}")
            return
        print(f"\n{'─'*52}")
        print(f"  Image  : {r['image_path']}")
        print(f"  Result : {r['label'].upper()}  ({r['confidence']*100:.2f}%)")
        print()
        for cls, prob in r['all_probs'].items():
            bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
            tag = "  ◄" if cls == r['label'] else ""
            print(f"  {cls:>14}: {bar} {prob*100:5.1f}%{tag}")
        print(f"{'─'*52}")
