"""upload the orthographic dnn priming dataset to hugging face hub."""

import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi


class DatasetUploader:
    """assembles and uploads the dataset to hugging face."""

    def __init__(self, repo_id: str, src_root: Path = Path(__file__).resolve().parent.parent / "src"):
        self.repo_id = repo_id
        self.src = src_root
        self.staging = Path(__file__).resolve().parent / "staging"
        self.api = HfApi()

    def assemble(self):
        """copy prime data and metadata into staging directory."""
        prime_src = self.src / "data" / "prime_data"
        prime_dst = self.staging / "prime_data"
        metadata_dst = self.staging / "metadata"

        assert prime_src.exists(), f"prime data not found at {prime_src} - run generate_data.py first"

        if self.staging.exists():
            shutil.rmtree(self.staging)

        shutil.copytree(prime_src, prime_dst)
        metadata_dst.mkdir(parents=True)

        for name in ["2014-prime-types.txt", "2014-targets.txt", "2014-prime-data.json"]:
            shutil.copy2(self.src / "assets" / name, metadata_dst / name)

        norm_path = self.src / "data" / "normalization_stats.json"
        if norm_path.exists():
            shutil.copy2(norm_path, metadata_dst / "normalization-stats.json")

        shutil.copy2(Path(__file__).resolve().parent / "README.md", self.staging / "README.md")

        n_images = sum(1 for _ in prime_dst.rglob("*.png"))
        print(f"staged {n_images} images + metadata in {self.staging}")

    def upload(self):
        """push staging directory to hugging face hub."""
        self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", exist_ok=True)
        self.api.upload_large_folder(
            folder_path=str(self.staging),
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        print(f"uploaded to https://huggingface.co/datasets/{self.repo_id}")

    def cleanup(self):
        """remove staging directory."""
        if self.staging.exists():
            shutil.rmtree(self.staging)
            print("staging directory removed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python upload_to_hf.py <your-username/orthographic-dnn-priming>")
        sys.exit(1)

    uploader = DatasetUploader(repo_id=sys.argv[1])
    uploader.assemble()
    uploader.upload()
    uploader.cleanup()
