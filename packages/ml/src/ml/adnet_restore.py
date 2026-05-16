from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ADNetConfig:
    enabled: bool = False
    weights_path: str | None = None
    repo_dir: str | None = None
    device: str = "cpu"


class ADNetRestorer:
    """Lazy EG-Restormer wrapper for already-localized QR crops.

    ADNet is optional. Missing repo, weights or deep-learning dependencies are
    reported through ``status`` and never break the QR pipeline.
    """

    def __init__(self, config: ADNetConfig | None = None) -> None:
        self.config = config or ADNetConfig()
        self._model: Any | None = None
        self._device = "cpu"
        self._error = ""
        self._loaded = False

    @property
    def status(self) -> dict[str, str | bool]:
        return {
            "enabled": self.config.enabled,
            "loaded": self._model is not None,
            "device": self._device,
            "error": self._error,
            "weights_path": str(self._weights_path() or ""),
            "repo_dir": str(self._repo_dir() or ""),
        }

    def restore(self, image_bgr: Any) -> Any | None:
        if not self.config.enabled:
            return None
        model = self._load_model()
        if model is None:
            return None
        try:
            from .media import import_cv2, import_numpy

            cv2 = import_cv2()
            np = import_numpy()
            import torch  # type: ignore
            import torch.nn.functional as functional  # type: ignore
        except Exception as exc:
            self._error = str(exc)
            return None

        try:
            if len(image_bgr.shape) == 2:
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(np.float32(image_rgb) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self._device)
            )
            height, width = tensor.shape[2], tensor.shape[3]
            padded_height = ((height + 7) // 8) * 8
            padded_width = ((width + 7) // 8) * 8
            tensor = functional.pad(
                tensor,
                (0, padded_width - width, 0, padded_height - height),
                "reflect",
            )
            with torch.no_grad():
                output = model(tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            output = output[:, :, :height, :width]
            restored = (
                torch.clamp(output, 0, 1)
                .detach()
                .cpu()
                .permute(0, 2, 3, 1)
                .squeeze(0)
                .numpy()
            )
            return cv2.cvtColor((restored * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            self._error = str(exc)
            return None

    def _load_model(self) -> Any | None:
        if self._loaded:
            return self._model
        self._loaded = True
        if not self.config.enabled:
            return None
        repo_dir = self._repo_dir()
        weights_path = self._weights_path()
        if repo_dir is None or not repo_dir.exists():
            self._error = f"ADNet repo not found: {repo_dir or ''}"
            return None
        if weights_path is None or not weights_path.exists():
            self._error = f"ADNet weights not found: {weights_path or ''}"
            return None
        yaml_path = repo_dir / "options" / "train" / "train_egrestormer_qrdataset.yml"
        if not yaml_path.exists():
            self._error = f"ADNet yaml config not found: {yaml_path}"
            return None
        try:
            import torch  # type: ignore
            import yaml  # type: ignore

            if str(repo_dir) not in sys.path:
                sys.path.insert(0, str(repo_dir))
            from basicsr.archs.eg_restormer_arch import EGRestormer  # type: ignore
        except Exception as exc:
            self._error = str(exc)
            return None

        try:
            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            network_config = dict(config["network_g"])
            network_config.pop("type", None)
            model = EGRestormer(**network_config)
            checkpoint = torch.load(str(weights_path), map_location="cpu")
            model.load_state_dict(checkpoint.get("params", checkpoint))
            self._device = self._select_device(torch)
            self._model = model.to(self._device).eval()
            self._error = ""
        except Exception as exc:
            self._model = None
            self._error = str(exc)
        return self._model

    def _repo_dir(self) -> Path | None:
        value = self.config.repo_dir or os.getenv("ADNET_REPO_DIR")
        if value:
            return Path(value)
        root = Path(__file__).resolve().parents[4]
        return root / "experiments" / "adnet_repo"

    def _weights_path(self) -> Path | None:
        value = self.config.weights_path or os.getenv("ADNET_WEIGHTS_PATH")
        return Path(value) if value else None

    def _select_device(self, torch_module: Any) -> str:
        requested = self.config.device
        if requested.startswith("cuda") and torch_module.cuda.is_available():
            return requested
        return "cpu"
