"""orchestration functions for train and prime data generation pipelines."""

import json
from itertools import product
from multiprocessing import Pool, cpu_count

from numpy import arange
from torchvision.datasets import ImageFolder as torch_image_folder
from tqdm import tqdm

from config import DIR_DATA, DIR_FONTS, PATH_CORPUS, PATH_PRIME_DATA, PATH_PRIME_TYPES, PATH_RANDOM_STRINGS, PATH_TARGETS
from utils.data_generate.main import CreateData, generate_single_image
from utils.data_generate.read_corpus import read_corpus
from utils.data_load.normalize import add_compute_stats


def get_position_from_type(position_correction: bool, prime_type: str, target: str, content: str) -> tuple:
    """return positional offset for a prime type, correcting for length differences."""
    if not position_correction:
        return (0, 0)

    match prime_type:
        case "SUB3" if content == target[:3]:
            return (-0.415, 0)
        case "SUB3":
            return (0.415, 0)
        case "DL-1F":
            return (-0.3, 0)
        case "DL-2M":
            return (-0.34, 0)
        case "IL-1M":
            return (0.345, 0)
        case "IL-2M":
            return (0.3, 0)
        case "IL-1I":
            return (-0.3, 0)
        case "IL-1F":
            return (0.3, 0)
        case _:
            return (0, 0)


def _build_combination_dict(c: tuple) -> dict:
    """convert a raw combination tuple into a labelled dict for CreateData."""
    return {
        "word": c[0],
        "font": c[1],
        "size": c[2],
        "position": c[3],
        "index": c[4],
        "name_save": "-".join([c[0].lower(), c[1], str(c[2]), str(c[4]), ".png"]),
    }


def _print_train_summary(n_total: int, coeff_train: float, coeff_valid: float,
                         corpus: list, fonts: list, sizes: list, positions: list, config: dict):
    """print summary statistics for train data generation."""
    print(f"\n--------------------------------")
    print(f"total images: {n_total}")
    print(f"images in train set: {int(n_total * coeff_train)}")
    print(f"images in valid set: {int(n_total * coeff_valid)}")
    print(f"--------------------------------")
    print(f"words: {len(corpus)}")
    print(f"fonts: {len(fonts)}")
    print(f"level of sizes: {len(sizes)}")
    print(f"random positions (1 == all centred): {len(positions)}")
    print(f"--------------------------------")
    print(f"letter rotation angle sd: {config['standard_deviation_rotation']}")
    print(f"letter translation coefficient: {config['coefficient_translation']}")
    print(f"space between letters coefficient: {config['coefficient_space']}")
    print(f"font variation range: {config['variance_font']}\n")


def init_create_train_data(dummy: bool = False, random: bool = False):
    """create train data images with font/size/position variations."""
    if random:
        corpus = json.loads(PATH_RANDOM_STRINGS.read_text())
    else:
        corpus = read_corpus(PATH_CORPUS)

    fonts = [f.name for f in DIR_FONTS.iterdir() if f.suffix == ".ttf"]
    sizes = list(arange(18, 20, 2)) if dummy else list(arange(18, 38, 2))
    positions = [(0, 0)]
    multiplier = 1 if dummy else 8

    configuration_variation = {
        "standard_deviation_rotation": 8.0,
        "coefficient_translation": 0.04,
        "coefficient_space": 0.5,
        "variance_font": 3,
    }

    raw_combos = list(product(corpus, fonts, sizes, positions)) * multiplier
    indexed_combos = [combo + (i,) for i, combo in enumerate(raw_combos)]
    combinations = [_build_combination_dict(c) for c in indexed_combos]

    generated_files = {file.name for file in (DIR_DATA / "data_all").rglob("*.png")} if (DIR_DATA / "data_all").exists() else set()
    if generated_files:
        combinations = [c for c in combinations if c["name_save"] not in generated_files]

    create = CreateData(mode="main")

    _print_train_summary(
        len(combinations), create.coefficient_data_train, create.coefficient_data_valid,
        corpus, fonts, sizes, positions, configuration_variation,
    )

    args = [(c, "main", configuration_variation) for c in combinations]
    chunksize = max(1, len(args) // (cpu_count() * 4))
    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(generate_single_image, args, chunksize=chunksize), total=len(args)):
            pass

    create = CreateData(mode="main")
    create.split_images_to_folders()

    stats = add_compute_stats(torch_image_folder)(root=str(DIR_DATA / "data_train")).stats
    stats = {"mean": list(stats["mean"]), "std": list(stats["std"])}
    filename = "normalization_stats_random.json" if random else "normalization_stats.json"
    (DIR_DATA / filename).write_text(json.dumps(stats))


def _lookup_prime_content(prime_data: list[dict], target_id: str, prime_type: str) -> str:
    """find the prime content for a given target and prime type."""
    return next(item[prime_type] for item in prime_data if item["ID"] == target_id)


def _build_prime_dict(target: str, prime: str, prime_data: list[dict], position_correction: bool) -> dict:
    """build a configuration dict for a single prime image."""
    content = _lookup_prime_content(prime_data, target, prime)
    return {
        "target": target,
        "prime": prime,
        "content": content,
        "font": "arial.ttf",
        "size": 22,
        "position": get_position_from_type(
            position_correction=position_correction,
            prime_type=prime,
            target=target,
            content=content,
        ),
        "index": 0,
        "name_save": f"{prime}.png",
    }


def init_create_prime_data(position_correction: bool = False):
    """create prime stimulus images with no variation, centered placement."""
    targets = read_corpus(PATH_TARGETS)
    primes = read_corpus(PATH_PRIME_TYPES)
    prime_data = json.loads(PATH_PRIME_DATA.read_text())

    combinations = [
        _build_prime_dict(t, p, prime_data, position_correction)
        for t, p in product(targets, primes)
    ]

    print(f"\n--------------------------------")
    print(f"total images: {len(combinations)}")
    print(f"--------------------------------")
    print(f"words: {len(targets)}")
    print(f"--------------------------------\n")

    configuration_variation = {
        "standard_deviation_rotation": 0,
        "coefficient_translation": 0,
        "coefficient_space": 0.5,
        "variance_font": 0,
    }

    args = [(c, "prime", configuration_variation) for c in combinations]
    chunksize = max(1, len(args) // (cpu_count() * 4))
    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(generate_single_image, args, chunksize=chunksize), total=len(args)):
            pass
