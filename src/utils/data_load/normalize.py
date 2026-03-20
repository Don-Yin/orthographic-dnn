import pickle
from pathlib import Path
from time import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as tf
from PIL import ImageStat
from sty import fg, rs


def _iter_pil_images(data_loader):
    """yields individual PIL images from a batched data loader."""
    to_pil = tf.ToPILImage()
    for batch_data, _ in data_loader:
        yield from (to_pil(batch_data[b]) for b in range(batch_data.shape[0]))


def compute_mean_and_std_from_dataset(dataset, dataset_path=None, max_iteration=100, data_loader=None, verbose=True):
    """computes channel-wise mean and std from a dataset by iterating over images."""
    if max_iteration < 30:
        print("max iteration in compute mean and std for dataset is lower than 30! this could create unrepresentative stats!") if verbose else None
    start = time()
    stats = {}
    transform_save = dataset.transform
    if data_loader is None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    statistics = None
    count = 0
    for img in _iter_pil_images(data_loader):
        if count % 10 == 9 and verbose:
            print(f"{count}/{max_iteration}, m: {np.array(statistics.mean)/255}, std: {np.array(statistics.stddev)/255}")
        count += 1
        statistics = Stats(img) if statistics is None else statistics + Stats(img)
        if count >= max_iteration:
            break

    stats["time_one_iter"] = (time() - start) / max_iteration
    stats["mean"] = np.array(statistics.mean) / 255
    stats["std"] = np.array(statistics.stddev) / 255
    stats["iter"] = max_iteration
    print((fg.cyan + "mean: {}; std: {}, time: {}" + rs.fg).format(stats["mean"], stats["std"], stats["time_one_iter"])) if verbose else None
    if dataset_path is not None:
        print(f"saving in {dataset_path}")
        with open(dataset_path, "wb") as f:
            pickle.dump(stats, f)

    dataset.transform = transform_save
    return stats


class Stats(ImageStat.Stat):
    """accumulates image statistics across multiple images."""

    def __add__(self, other):
        return Stats(list(map(np.add, self.h, other.h)))


def add_compute_stats(obj_class):
    """returns a dataset class that auto-computes normalization stats."""

    class ComputeStatsUpdateTransform(obj_class):
        """dataset wrapper that computes and applies normalization transforms."""

        def __init__(
            self,
            name_generator="dataset",
            add_PIL_transforms=None,
            add_tensor_transforms=None,
            num_image_calculate_mean_std=70,
            stats=None,
            save_stats_file=None,
            **kwargs,
        ):
            """initializes dataset with optional normalization stats."""
            self.verbose = True
            print(fg.yellow + "\n**creating dataset [" + fg.cyan + f"{name_generator}" + fg.yellow + "]**" + rs.fg)
            super().__init__(**kwargs)
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            if add_PIL_transforms is None:
                add_PIL_transforms = []
            if add_tensor_transforms is None:
                add_tensor_transforms = []

            self.transform = torchvision.transforms.Compose([*add_PIL_transforms, torchvision.transforms.ToTensor(), *add_tensor_transforms])

            self.name_generator = name_generator
            self.additional_transform = add_PIL_transforms
            self.num_image_calculate_mean_std = num_image_calculate_mean_std
            self.num_classes = len(self.classes)

            match stats:
                case dict():
                    self.stats = stats
                    print(fg.red + "using precomputed stats: " + fg.cyan + f"mean = {self.stats['mean']}, std = {self.stats['std']}" + rs.fg)
                case str() if Path(stats).is_file():
                    with open(Path(stats), "rb") as f:
                        self.stats = pickle.load(f)
                    print(
                        fg.red
                        + f"using stats from file [{Path(stats).name}]: "
                        + fg.cyan
                        + f"mean = {self.stats['mean']}, std = {self.stats['std']}"
                        + rs.fg
                    )
                    if stats == save_stats_file:
                        save_stats_file = None
                case str():
                    print(fg.red + f"file [{Path(stats).name}] not found, stats will be computed." + rs.fg)
                    self.stats = self.call_compute_stats()
                case _:
                    self.stats = self.call_compute_stats()

            if save_stats_file is not None:
                print(f"stats saved in {save_stats_file}")
                Path(save_stats_file).parent.mkdir(parents=True, exist_ok=True)
                with open(save_stats_file, "wb") as f:
                    pickle.dump(self.stats, f)

            normalize = torchvision.transforms.Normalize(mean=self.stats["mean"], std=self.stats["std"])
            self.transform.transforms += [normalize]
            print(f"map class_name -> labels: {self.class_to_idx}\n{len(self)} samples.") if self.verbose else None

        def call_compute_stats(self):
            """computes normalization stats from the current dataset."""
            return compute_mean_and_std_from_dataset(self, None, max_iteration=self.num_image_calculate_mean_std, verbose=self.verbose)

    return ComputeStatsUpdateTransform


if __name__ == "__main__":
    from config import DIR_DATA
    from subset import MyImageFolder

    data_train = MyImageFolder(root=str(DIR_DATA / "data_all"), name_classes=["and"])
    print(data_train)
