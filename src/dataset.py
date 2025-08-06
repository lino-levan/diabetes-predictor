import pandas as pd
import torch
from torch.utils.data import Dataset

class DiabetesDataset(Dataset):
    def __init__(self, csv_file, transform=None, scaler=None, fit_scaler=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            scaler: Optional StandardScaler for feature normalization
            fit_scaler: Whether to fit the scaler on this data
        """
        self.diabetes_frame = pd.read_csv(csv_file)
        self.transform = transform

        # Separate features and labels
        self.labels = self.diabetes_frame.iloc[:, 0].values
        self.features = self.diabetes_frame.iloc[:, 1:].values

        # Apply scaling if scaler is provided
        if scaler is not None:
            if fit_scaler:
                self.features = scaler.fit_transform(self.features)
            else:
                self.features = scaler.transform(self.features)

        self.scaler = scaler

    def __len__(self):
        return len(self.diabetes_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]
        features = self.features[idx]
        result = { "label": torch.tensor([label], dtype=torch.float32), "features": torch.tensor(features, dtype=torch.float32) }

        if self.transform:
            result = self.transform(result)

        return result

if __name__ == "__main__":
    data = DiabetesDataset("data/all.csv")
    print(data[50000])
