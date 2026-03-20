import json
from pathlib import Path

import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader as data_loader
from torchvision.datasets import ImageFolder as torch_image_folder

from config import DIR_DATA, DIR_PARAMS, PATH_CORPUS, PATH_NORM_STATS
from utils.data_load.device_control import DeviceDataLoader, device_allocator
from utils.data_load.normalize import add_compute_stats
from utils.data_load.subset import MyImageFolder
from utils.train.converging_average import ExpMovingAverage


class Train:
    """trains a single model with early stopping and periodic checkpointing."""

    def __init__(self, hyperparameters: dict):
        self.hyperparameters = hyperparameters
        self.path_data: Path = DIR_DATA / "data_train"

    def fit(self):
        """runs the full training loop across epochs."""
        self.load_data()
        self.set_batch_loaders()
        self.set_model()

        optimizer = self.hyperparameters["function_optimizer"](
            params=self.model.parameters(),
            lr=self.hyperparameters["rate_learning"],
            weight_decay=self.hyperparameters["weight_decay"],
        )

        moving_average = ExpMovingAverage(0)

        for epoch in range(self.hyperparameters["num_epochs"]):
            print(f"epoch: {epoch}")
            self.model.train()
            for batch_count, batch in enumerate(self.data_train_batch_loader):
                loss = self.get_train_loss(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if (batch_count + 1) % self.hyperparameters["batch_report_every"] == 0:
                    print(f"reporting batch: {batch_count + 1}")
                    test_loss_and_accuracy = self.evaluate_test_set()
                    moving_average(test_loss_and_accuracy["batch_loss"])

                    if moving_average.avg < self.hyperparameters["min_moving_average_threshold"]:
                        self.save_params()
                        print(f"training converged: {self.hyperparameters['model_tag']}")
                        return

                if (batch_count + 1) % self.hyperparameters["batch_save_every"] == 0:
                    self.save_params()
                    print(f"params saved: {self.hyperparameters['model_tag']}")

    def read_list_corpus(self):
        """reads the word corpus from the assets file."""
        corpus: str = PATH_CORPUS.read_text()
        return [w for w in corpus.split("\n") if w != ""]

    def load_data(self, whether_normalize: bool = True, subset: tuple | None = None):
        """loads training data with optional normalization and subsetting."""
        normalization_stats = json.loads(PATH_NORM_STATS.read_text()) if PATH_NORM_STATS.exists() else None

        if subset:
            corpus: list[str] = self.read_list_corpus()
            classes = corpus[subset[0] : subset[1]]
            base_class = add_compute_stats(MyImageFolder) if whether_normalize else MyImageFolder
            kwargs = {"root": str(self.path_data), "name_classes": classes}
            if whether_normalize:
                kwargs["stats"] = normalization_stats
            self.data = base_class(**kwargs)
        else:
            self.data = add_compute_stats(torch_image_folder)(root=str(self.path_data), stats=normalization_stats)

    def set_batch_loaders(self):
        """creates train and test data loaders with proper split sizes."""
        total = len(self.data)
        train_size = int(total * 0.8)
        test_size = total - train_size
        data_train, data_test = random_split(self.data, [train_size, test_size])

        num_workers = 0

        self.data_train_batch_loader = DeviceDataLoader(data_loader(data_train, self.hyperparameters["size_batch"], shuffle=True, num_workers=num_workers, pin_memory=True))

        self.data_test_batch_loader = DeviceDataLoader(data_loader(data_test, self.hyperparameters["size_batch"] * 2, shuffle=True, num_workers=num_workers, pin_memory=True))

    def set_model(self):
        """assigns the model and moves it to the compute device."""
        self.model = self.hyperparameters["model"]
        device_allocator(self.model)

    def save_params(self):
        """saves model state dict to the params directory."""
        DIR_PARAMS.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), f=DIR_PARAMS / f"{self.hyperparameters['model_tag']}.pth")

    def get_train_loss(self, batch):
        """computes forward pass loss for a training batch."""
        images, labels = batch
        output = self.model(images)
        loss = self.hyperparameters["function_loss"](output, labels)
        return loss

    def evaluate_test_set(self):
        """evaluates loss and accuracy on the full test set without gradients."""
        self.model.eval()
        total_loss = 0.0
        total_correct_top1 = 0
        total_correct_top5 = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.data_test_batch_loader:
                images, labels = batch
                outputs = self.model(images)
                loss = self.hyperparameters["function_loss"](outputs, labels)
                _, predictions_top_1 = torch.max(outputs, dim=1)
                _, predictions_top_5 = torch.topk(outputs, k=5, dim=1)
                predictions_top_5 = predictions_top_5.t()

                batch_size = len(labels)
                total_loss += loss.item() * batch_size
                total_correct_top1 += torch.sum(predictions_top_1 == labels).item()
                total_correct_top5 += torch.sum(torch.eq(predictions_top_5, labels).t()).item()
                total_samples += batch_size

        self.model.train()

        return {
            "batch_loss": torch.tensor(total_loss / total_samples),
            "batch_accuracy_top_1": torch.tensor(total_correct_top1 / total_samples),
            "batch_accuracy_top_5": torch.tensor(total_correct_top5 / total_samples),
        }


if __name__ == "__main__":
    pass
