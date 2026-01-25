from typing import Optional, Any
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: Optional[dict[str, nn.Module]] = None,
    ):
        self._pretrained_args: Optional[dict[str, Any]] = None
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        from pathlib import Path

        # Prefer local directory loads (supports both direct exports and nested snapshot layouts).
        p = Path(path)
        resolved_path = path
        config_file: str
        is_hub_source = False

        if p.exists() and p.is_dir():
            direct = p / "pipeline.json"
            if direct.exists():
                config_file = str(direct)
                resolved_path = str(p)
            else:
                # Try to locate a nested pipeline.json (e.g., downloaded into a subfolder or cache snapshot).
                candidates = [c for c in p.rglob("pipeline.json") if c.is_file()]
                if not candidates:
                    raise FileNotFoundError(
                        f"No pipeline.json found under local directory: {p}. "
                        f"Download the model files into that folder first (Launcher option 'Download / Update Models')."
                    )

                def _is_candidate_locally_valid(candidate: Path) -> bool:
                    """Return True if candidate's referenced ckpt files exist under candidate.parent."""
                    try:
                        data = json.loads(candidate.read_text(encoding="utf-8"))
                        model_paths = data.get("args", {}).get("models", {})
                        if not isinstance(model_paths, dict) or not model_paths:
                            return False
                        base_dir = candidate.parent
                        for v in model_paths.values():
                            # models.from_pretrained expects "{path}.json" and "{path}.safetensors"
                            ckpt_base = base_dir / str(v)
                            if not (ckpt_base.with_suffix(".json").exists() and ckpt_base.with_suffix(".safetensors").exists()):
                                return False
                        return True
                    except Exception:
                        return False

                def _score(c: Path) -> tuple[int, int, int, str]:
                    # Prefer candidates that are self-consistent with local files.
                    valid = 0 if _is_candidate_locally_valid(c) else 1
                    # Prefer typical HF snapshot locations if present.
                    prefer_snapshot = 0 if "snapshots" in c.parts else 1
                    # Prefer shallower paths.
                    rel_parts = len(c.relative_to(p).parts)
                    return (valid, prefer_snapshot, rel_parts, str(c))

                best = sorted(candidates, key=_score)[0]
                config_file = str(best)
                resolved_path = str(best.parent)
        else:
            # Treat as Hugging Face Hub repo id only if it looks like one.
            if "/" not in path:
                raise ValueError(
                    f"from_pretrained expected a local folder (containing pipeline.json) or a Hub repo id like 'owner/repo'. "
                    f"Got: {path!r}"
                )
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")
            resolved_path = path
            is_hub_source = True

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {
            k: models.from_pretrained(f"{resolved_path}/{v}" if is_hub_source else os.path.join(resolved_path, v))
            for k, v in args['models'].items()
        }

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        # Jan 2025 memory optimizations: we'll move different models between CPU and GPU.
        # Our models are kept on CPU, but a model that is active is always loaded into GPU.
        # return 'cuda' if there is at least 1 model on CUDA. 
        # Only return 'cpu' if everything is on CPU:
        for model in self.models.values():
            if hasattr(model, 'device') and model.device.type == 'cuda':
                return torch.device('cuda')
            if hasattr(model, 'parameters'):
                try:
                    if next(model.parameters()).device.type == 'cuda':
                        return torch.device('cuda')
                except StopIteration:
                    continue # Handle models with no parameters.
        return torch.device('cpu')# If we get here, no models were on cuda.

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
         self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
