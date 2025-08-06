import pandas as pd
import torch
from torch.utils.data import Dataset

class DiabetesDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.diabetes_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.diabetes_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.diabetes_frame.iloc[idx, 0]
        features = self.diabetes_frame.iloc[idx, 1:]
        result = { "label": torch.tensor([label], dtype=torch.float32), "features": torch.tensor(features.values, dtype=torch.float32) }

        if self.transform:
            result = self.transform(result)

        return result

if __name__ == "__main__":
    data = DiabetesDataset("data/all.csv")
    print(data[50000])
