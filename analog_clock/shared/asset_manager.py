"""
Asset manager: downloads and discovers background textures.

Consolidated from the two identical asset_loader.py / AssetManager copies.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm

from analog_clock.shared.config import BaseConfig

logger = logging.getLogger(__name__)


class AssetManager:
    """Downloads Kaggle / Picsum backgrounds and provides random access."""

    def __init__(self, config: BaseConfig | None = None) -> None:
        self.config = config or BaseConfig()
        self.root_dir = Path(self.config.BACKGROUNDS_DIR)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_assets(self) -> None:
        logger.info("Asset directory: %s", self.root_dir)

        if self.config.DOWNLOAD_KAGGLE:
            self._check_and_download_backgrounds()

        images = self.get_all_images()
        logger.info("Background count: %d", len(images))

        if len(images) < 10 and self.config.DOWNLOAD_PICSUM:
            logger.warning("Backgrounds missing — downloading fallback textures")
            self._download_picsum(self.config.NUM_TEXTURES_TO_DOWNLOAD)

    def get_all_images(self) -> List[Path]:
        files: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(self.root_dir.rglob(ext))
        return files

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_and_download_backgrounds(self) -> None:
        if self.root_dir.exists() and os.listdir(self.root_dir):
            return
        logger.info("Attempting Kaggle CLI download …")
        try:
            subprocess.run(
                [
                    "kaggle", "datasets", "download", "-d",
                    self.config.KAGGLE_DATASET,
                    "-p", str(self.root_dir.resolve()), "--unzip",
                ],
                check=True,
            )
            logger.info("Textures downloaded.")
        except Exception as exc:
            logger.warning("Kaggle download failed: %s. Using fallback.", exc)

    def _download_picsum(self, count: int) -> None:
        picsum_dir = self.root_dir / "picsum"
        picsum_dir.mkdir(exist_ok=True)
        for i in tqdm(range(count), desc="Downloading textures"):
            try:
                url = f"https://picsum.photos/1024/1024?random={i + 2000}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    (picsum_dir / f"tex_{i}.jpg").write_bytes(resp.content)
            except Exception as exc:
                logger.debug("Picsum download %d failed: %s", i, exc)
