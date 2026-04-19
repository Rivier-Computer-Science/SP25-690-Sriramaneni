import pandas as pd


class DataLoader:
    def __init__(self, csv_path):
        """Handles loading of a dataset CSV file."""
        self.csv_path = csv_path

    def load_data(self):
        """Load the dataset and perform basic validation."""
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError as exc:
            raise Exception(f"File not found at path: {self.csv_path}") from exc

        print("\nDataset loaded successfully.")
        print("Shape:", df.shape)

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.head())

        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\nMissing values found:")
            print(missing[missing > 0])

            df = df.dropna()
            print("Missing values removed. New shape:", df.shape)

        if "sentiment" not in df.columns:
            raise Exception("'sentiment' column not found in dataset!")

        print("\nSentiment distribution:")
        print(df["sentiment"].value_counts())

        return df
