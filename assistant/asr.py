"""NVIDIA Parakeet TDT ASR wrapper."""

import asyncio
import importlib.util
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"


def _fix_nemo_case() -> None:
    """On macOS the NeMo package is installed as 'NEMO/' (uppercase) but
    Python's importer is case-sensitive and can't find 'nemo' (lowercase).
    We locate the NEMO directory, pre-register a 'nemo' module stub in
    sys.modules so that the absolute imports inside NEMO/__init__.py resolve,
    then execute the real __init__.py in that stub's namespace."""
    if "nemo" in sys.modules:
        return

    site_pkgs = next(
        (p for p in sys.path if "site-packages" in p and os.path.isdir(p)), None
    )
    if site_pkgs is None:
        return

    nemo_dir = os.path.join(site_pkgs, "NEMO")
    nemo_init = os.path.join(nemo_dir, "__init__.py")
    if not os.path.isfile(nemo_init):
        return

    spec = importlib.util.spec_from_file_location(
        "nemo",
        nemo_init,
        submodule_search_locations=[nemo_dir],
    )
    nemo_mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so that 'from nemo.package_info import …' inside
    # __init__.py can resolve against this same module object.
    sys.modules["nemo"] = nemo_mod
    sys.modules["NEMO"] = nemo_mod  # alias in case anything imports NEMO
    try:
        spec.loader.exec_module(nemo_mod)
    except Exception:
        # If init fails, leave the stub; subpackage imports still work.
        pass


class ASRModel:
    """Loads Parakeet via NeMo and transcribes numpy audio arrays."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        _fix_nemo_case()
        import nemo.collections.asr as nemo_asr  # lazy import (heavy)

        print(f"[ASR] Loading {self.model_name} ...")
        self._model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        self._model.eval()
        print("[ASR] Model ready.")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """Transcribe float32 mono 16 kHz audio. Returns plain text."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, audio, sample_rate)
            results = self._model.transcribe([f.name])
        if not results:
            return ""
        result = results[0]
        # NeMo returns either a string or a dataclass with .text
        return result.text if hasattr(result, "text") else str(result)

    async def transcribe_async(
        self, audio: np.ndarray, sample_rate: int = 16_000
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.transcribe, audio, sample_rate)
