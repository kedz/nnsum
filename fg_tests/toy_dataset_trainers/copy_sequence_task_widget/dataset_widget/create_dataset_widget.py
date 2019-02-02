import ipywidgets as widgets


class CreateDatasetWidget(object):
    def __init__(self):

        self._w_vocab_size = widgets.BoundedIntText(
            value=25, min=1, max=200000, step=1, description='Vocab Size:')
        self._w_len = widgets.IntRangeSlider(
            value=[5,10], min=1, max=200, step=1, description='Length:')
        self._w_train_size = widgets.BoundedIntText(
            value=2048, min=1, max=20000, step=1, description='Train Size:')
        self._w_test_size = widgets.BoundedIntText(
            value=100, min=1, max=20000, step=1, description='Test Size:')
        self._w_train_seed = widgets.BoundedIntText(
            value=84729745, min=1, max=999999999, step=1, 
            description='Train Seed:')
        self._w_test_seed = widgets.BoundedIntText(
            value=94627924, min=1, max=999999999, step=1, 
            description='Test Seed:')
        self._w_dataset_name = widgets.Text(
            value="dataset-1", description="Name:")
        self._w_create_button = widgets.Button(description="Create Dataset")
        self._w_create_button.on_click(self.create_button_click)

        self._w_main = widgets.VBox([
            widgets.HBox([self._w_vocab_size, self._w_len]),
            widgets.HBox([self._w_train_size, self._w_test_size]),
            widgets.HBox([self._w_train_seed, self._w_test_seed]),
            widgets.HBox([self._w_create_button, self._w_dataset_name]),
        ])

        self._callbacks = []

    def __call__(self):
        return self._w_main

    @property
    def vocab_size(self):
        return self._w_vocab_size.value

    @vocab_size.setter
    def vocab_size(self, new_size):
        self._w_vocab_size.value = new_size

    @property
    def min_length(self):
        return self._w_len.value[0]

    @min_length.setter
    def min_length(self, new_len):
        self._w_len.value = [new_len, self._w_len.value[1]]

    @property
    def max_length(self):
        return self._w_len.value[1]

    @max_length.setter
    def max_length(self, new_len):
        self._w_len.value = [self._w_len.value[0], new_len]

    @property
    def train_size(self):
        return self._w_train_size.value

    @train_size.setter
    def train_size(self, new_size):
        self._w_train_size.value = new_size

    @property
    def test_size(self):
        return self._w_test_size.value

    @test_size.setter
    def test_size(self, new_size):
        self._w_test_size.value = new_size

    @property
    def train_seed(self):
        return self._w_train_seed.value

    @train_seed.setter
    def train_seed(self, new_seed):
        self._w_train_seed.value = new_seed

    @property
    def test_seed(self):
        return self._w_test_seed.value

    @test_seed.setter
    def test_seed(self, new_seed):
        self._w_test_seed.value = new_seed

    @property
    def dataset_name(self):
        return self._w_dataset_name.value.strip()
    
    @dataset_name.setter
    def dataset_name(self, new_name):
        self._w_dataset_name.value = new_name
    
    def register_callback(self, callback):
        self._callbacks.append(callback)

    def create_button_click(self, button):
        params = {
            "name": self.dataset_name, "vocab_size": self.vocab_size, 
            "min_length": self.min_length, "max_length": self.max_length, 
            "train_size": self.train_size, "test_size": self.test_size,
            "train_seed": self.train_seed, "test_seed": self.test_seed,
        }
        for callback in self._callbacks:
            callback(params)
