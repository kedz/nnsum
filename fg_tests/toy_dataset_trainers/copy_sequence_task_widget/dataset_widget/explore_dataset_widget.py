import ipywidgets as widgets

from nnsum.datasets import CopyDataset


class ExploreDatasetWidget(object):
    def __init__(self, dataset_params, part="test"):

        self._dataset_params = dataset_params
        self._datasets = self._create_datasets(dataset_params)

        self._w_selector = widgets.Dropdown(options=["train", "test"],
                                            value=part, description="Part:")
        self._w_selector.observe(self._part_change, names=["value"])
        self._w_example = widgets.IntSlider(min=1, 
                                            max=dataset_params["test_size"],
                                            value=1,
                                            description="Example:")
        self._w_example.observe(self._example_change, names=["value"])
        

        self._w_source = widgets.Label()
        self._w_target = widgets.Label()

        self._w_main = widgets.VBox([
            widgets.HBox([self._w_selector, self._w_example]),
            widgets.Label("Source"),
            self._w_source,
            widgets.Label("Target"),
            self._w_target,
        ])

        self._show_example(0)

    def __call__(self):
        return self._w_main

    def _create_datasets(self, dataset_params):
        train_data = CopyDataset(dataset_params["vocab_size"], 
                                 dataset_params["min_length"],
                                 dataset_params["max_length"],
                                 dataset_params["train_size"],
                                 random_seed=dataset_params["train_seed"])
        test_data = CopyDataset(dataset_params["vocab_size"], 
                                dataset_params["min_length"],
                                dataset_params["max_length"],
                                dataset_params["test_size"],
                                random_seed=dataset_params["test_seed"])

        return {"train": train_data, "test": test_data}

    @property
    def part(self):
        return self._w_selector.value

    def set_source(self, source):
        self._w_source.value = source

    def set_target(self, target):
        self._w_target.value = target

    def _show_example(self, index):
        ex = self._datasets[self.part][index]
        self.set_source(" ".join(ex["source"]["tokens"]))
        self.set_target(" ".join(ex["target"]["tokens"]))

    def _example_change(self, change):
        self._show_example(change["new"] - 1)

    def _part_change(self, change):
        dataset = self._datasets[change["new"]]
        new_size = len(dataset)
        self._w_example.max = new_size
        self._w_example.value = 1
        self._show_example(0)
