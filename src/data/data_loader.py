import pandas as pd

class DataLoader:
    def __init__(self, csv_path):
        """
        Handles loading of dataset CSV file
        """
        self.csv_path = csv_path

    def load_data(self):
        """
        Load dataset and perform basic validation
        """

        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            raise Exception(f"File not found at path: {self.csv_path}")

        print("\n✅ Dataset Loaded Successfully!")
        print("Shape:", df.shape)

        # --------------------------
        # Basic Info
        # --------------------------
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.head())

        # --------------------------
        # Check missing values
        # --------------------------
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\n⚠️ Missing values found:")
            print(missing[missing > 0])

            # simple fix → drop missing rows
            df = df.dropna()
            print("Missing values removed. New shape:", df.shape)

        # --------------------------
        # Check sentiment column
        # --------------------------
        if "sentiment" not in df.columns:
            raise Exception("❌ 'sentiment' column not found in dataset!")

        print("\nSentiment distribution:")
        print(df["sentiment"].value_counts())

        return df