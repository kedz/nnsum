from torch.utils.data import Dataset


class AlignedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert isinstance(dataset1, Dataset)
        assert isinstance(dataset2, Dataset)
        assert len(dataset1) == len(dataset2)

        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, index):
        return {"source": self.dataset1[index], "target": self.dataset2[index]}
