from pathlib import Path

from utils.evaluate.cosine_similarity import compute_cosine_similarity


def build_image_index(prime_data) -> dict[tuple[str, str], int]:
    """build a lookup dict mapping (word, prime_type) to dataset index."""
    index = {}
    for idx, (path_str, _label) in enumerate(prime_data.imgs):
        p = Path(path_str).with_suffix("")
        word = p.parent.name
        prime_type = p.name
        index[(word, prime_type)] = idx
    return index


class Evaluate:
    """compute cosine similarity between prime activations for a word pair."""

    def __init__(self, model, word: str, prime_types: tuple[str], which_layer: str,
                 prime_data, image_index: dict, device: str = "cpu"):
        self.model = model
        self.word = word
        self.prime_types = prime_types
        self.which_layer = which_layer
        self.prime_data = prime_data
        self.image_index = image_index
        self.device = device
        self.activation = {}
        self._get_image_tensor()

    def _get_image_tensor(self):
        """load and prepare input tensors for the two prime types."""
        idx_0 = self.image_index[(self.word, self.prime_types[0])]
        idx_1 = self.image_index[(self.word, self.prime_types[1])]
        self.tensors = (
            self.prime_data[idx_0][0].unsqueeze(0).to(self.device).detach(),
            self.prime_data[idx_1][0].unsqueeze(0).to(self.device).detach(),
        )

    def compute_similarity_layer_wise(self):
        """compute similarity at the requested layer(s) and return result."""
        match self.which_layer:
            case "classification":
                return self._similarity_classification()
            case "penultimate":
                return self._similarity_penultimate()
            case "all":
                return self._similarity_all()
            case _:
                raise ValueError(f"unknown layer mode: {self.which_layer}")

    def _similarity_classification(self):
        """compute cosine similarity on classification outputs."""
        outputs = (self.model(self.tensors[0]), self.model(self.tensors[1]))
        return compute_cosine_similarity(outputs)

    def _similarity_penultimate(self):
        """compute cosine similarity on penultimate layer activations."""
        all_layers = self._group_all_layers()
        names, hooks = self._register_hooks(all_layers)
        self.model(self.tensors[0])
        output_0 = self.activation[names[-2]].flatten().unsqueeze(0)
        self.model(self.tensors[1])
        output_1 = self.activation[names[-2]].flatten().unsqueeze(0)
        self._remove_hooks(hooks)
        return compute_cosine_similarity((output_0, output_1))

    def _similarity_all(self):
        """compute cosine similarity at every layer."""
        all_layers = self._group_all_layers()
        names, hooks = self._register_hooks(all_layers)
        self.model(self.tensors[0])
        activations_0 = [self.activation[n].flatten().unsqueeze(0) for n in names]
        self.model(self.tensors[1])
        activations_1 = [self.activation[n].flatten().unsqueeze(0) for n in names]
        self._remove_hooks(hooks)
        return [
            compute_cosine_similarity((activations_0[i], activations_1[i]))
            for i in range(len(names))
        ]

    def _register_hooks(self, all_layers) -> tuple[list[str], list]:
        """register forward hooks on all leaf layers and return names and handles."""
        self.activation = {}
        names = []
        handles = []
        for idx, layer in enumerate(all_layers):
            name = f"{idx}: {str(layer).split('(')[0]}"
            names.append(name)
            handles.append(layer.register_forward_hook(self._get_activation(name)))
        return names, handles

    @staticmethod
    def _remove_hooks(handles):
        """remove all registered forward hook handles."""
        for h in handles:
            h.remove()

    def _get_activation(self, name):
        """return a hook function that captures layer output."""
        def hook(_model, _input, output):
            self.activation[name] = output.detach()
        return hook

    def _group_all_layers(self):
        """flatten model hierarchy into a list of leaf layers."""
        all_layers = []

        def recursive_group(model):
            for layer in model.children():
                if not list(layer.children()):
                    all_layers.append(layer)
                else:
                    recursive_group(layer)

        recursive_group(self.model)
        return all_layers
