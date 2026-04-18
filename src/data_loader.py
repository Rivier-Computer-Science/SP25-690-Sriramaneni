import pandas as pd

class DataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load_data(self):
        """
        Load dataset metadata (audio filenames + emotion labels)
        """
        df = pd.read_csv(self.csv_path)
        return df