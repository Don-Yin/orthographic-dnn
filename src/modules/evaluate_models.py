import json
import logging
from collections import defaultdict
from itertools import product
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder as torch_image_folder
from tqdm import tqdm

from config import DIR_ASSETS, DIR_DATA, MODEL_REGISTRY, PATH_NORM_STATS, PATH_TARGETS, PATH_PRIME_TYPES, load_model
from utils.data_generate.read_corpus import read_corpus
from utils.data_load.device_control import device_allocator
from utils.data_load.normalize import add_compute_stats
from utils.evaluate.evaluate_model import Evaluate, build_image_index


class BatchEvaluate:
    """evaluate all model-word-prime combinations and save cosine similarities."""

    def __init__(self, prime_data_folder: Path = DIR_DATA / "prime_data"):
        self.prime_data_folder = prime_data_folder
        self.path_output = DIR_ASSETS / "model_output" / "random_strings.json"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.read_prime_data()
        self.read_words_and_primes()
        self.read_selected_models()
        self.make_combinations()
        self.main()
        self.save()

    def read_words_and_primes(self):
        """load target words and prime type lists."""
        self.targets_2014 = read_corpus(PATH_TARGETS)
        self.prime_types_2014 = read_corpus(PATH_PRIME_TYPES)

    def read_selected_models(self):
        """read model names from the registry."""
        self.selected_network_names = list(MODEL_REGISTRY.keys())

    def read_prime_data(self):
        """load prime image dataset with normalization."""
        norm_path = PATH_NORM_STATS
        normalize_stats = json.loads(norm_path.read_text()) if norm_path.exists() else None
        self.prime_data = add_compute_stats(torch_image_folder)(
            root=str(self.prime_data_folder),
            stats=normalize_stats,
        )
        self.image_index = build_image_index(self.prime_data)

    def make_combinations(self):
        """build or reload the list of evaluation combinations."""
        if not self.path_output.exists():
            raw = list(
                product(
                    self.selected_network_names,
                    self.targets_2014,
                    ["ID"],
                    self.prime_types_2014,
                )
            )
            self.combinations = [{"model": c[0], "word": c[1], "primes": (c[2], c[3])} for c in raw]
            self.save()
        else:
            self.combinations = json.loads(self.path_output.read_text())

    def main(self):
        """process all combinations, loading one model at a time to save memory."""
        combos_by_model = defaultdict(list)
        for i, combo in enumerate(self.combinations):
            combos_by_model[combo["model"]].append(i)

        for model_name, indices in combos_by_model.items():
            logging.info("loading model: %s", model_name)
            model = load_model(model_name)
            model.eval()
            device_allocator(model)

            for i in tqdm(indices, desc=model_name):
                combo = self.combinations[i]
                if "cosine_similarities" not in combo:
                    combo["cosine_similarities"] = Evaluate(
                        model=model,
                        word=combo["word"],
                        prime_types=combo["primes"],
                        which_layer="all",
                        prime_data=self.prime_data,
                        image_index=self.image_index,
                        device=self.device,
                    ).compute_similarity_layer_wise()

                    if i % 128 == 0:
                        self.save()
                        logging.info("saved: %d", i)

            del model
            torch.cuda.empty_cache()

    def save(self):
        """write combinations to json output file."""
        self.path_output.parent.mkdir(parents=True, exist_ok=True)
        self.path_output.write_text(json.dumps(self.combinations))


if __name__ == "__main__":
    BatchEvaluate()
