from pathlib import Path

import numpy as np
from sty import ef, fg, rs
from torchvision.datasets import ImageFolder


class MyImageFolder(ImageFolder):
    """selectively loads training data for specified class names."""

    def __init__(self, name_classes=None, verbose=True, *args, **kwargs):
        print(fg.red + ef.inverse + "ROOT:  " + kwargs["root"] + rs.inverse + rs.fg)
        self.name_classes = np.sort(name_classes)
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def find_classes(self, directory: str):
        """finds classes matching name_classes in the given directory."""
        if self.name_classes is None:
            return super().find_classes(directory)
        classes = [d.name for d in Path(directory).iterdir() if d.is_dir() and (d.name in self.name_classes)]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
