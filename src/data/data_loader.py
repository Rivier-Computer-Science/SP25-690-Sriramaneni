import pandas as pd

class DataLoader:
    def __init__(self, csv_path):
        """
        Handles loading of metadata CSV file
        """
        self.csv_path = csv_path

    def load_data(self):
        """
        Loads dataset into pandas dataframe
        """
        df = pd.read_csv(self.csv_path)

        print("Dataset Loaded Successfully!")
        print("Shape:", df.shape)
        print(df.head())

        return df