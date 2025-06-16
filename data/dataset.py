import pandas as pd
import numpy as np
import json
from scipy import sparse
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional
import random
from pathlib import Path

class RecommendationDataset:

    def __init__(self, data_path: str, implicit_threshold: float = 4.0,
                 negative_sampling_ratio: float = 0.0, validation_split: float = 4.0):
        self.data_path = Path(data_path)
        self.implicit_threshold = implicit_threshold
        self.negative_sampling_ratio = negative_sampling_ratio
        self.validation_split = validation_split

        self.train_data = None
        self.test_data = None
        self.item_metadata = None
        self.id_mappings = None

        self.user_item_matrix = None
        self.train_matrix = None
        self.val_matrix = None

        self.n_users = None
        self.n_items = None
        self.n_interactions = None

    def load_data(self):
        train_file = self.data_path / 'train.csv'
        self.train_data = pd.read_csv(train_file)
        print("Train data loaded")

        test_file = self.data_path / 'test.csv'
        self.test_data = pd.read_csv(test_file)
        print("Test data loaded")

        metadata_file = self.data_path / 'item_metadata.csv'
        self.item_metadata = pd.read_csv(metadata_file)
        print("Item metadata loaded")

        mappings_file = self.data_path / 'id_mappings.json'
        with open(mappings_file, 'r') as f:
            self.id_mappings = json.load(f)
        print("Item mappings loaded")

        self.n_users = self.train_data['user_id'].max() + 1
        self.n_items = self.train_data['item_id'].max() + 1
        self.n_interactions = len(self.train_data)

        print("Datasets info:")
        print(f"   Users: {self.n_users:,}")
        print(f"   Items: {self.n_items:,}")
        print(f"   Interactions: {self.n_interactions:,}")
        print(f"   Sparsity: {(1 - self.n_interactions / (self.n_users * self.n_items)) * 100:.4f}%")

    def create_user_item_matrix(self) -> sparse.csr_matrix:
        if self.user_item_matrix is None:
            self.load_data()

        users = self.train_data['user_id'].values
        item_ids = self.train_data['item_id'].values
        ratings = self.train_data['rating'].values

        self.user_item_matrix = sparse.csr_matrix(
            (ratings, (users, item_ids)),
            shape=(self.n_users, self.n_items)
        )

        print(f"Sparse matrix created: {self.user_item_matrix.shape}")
        print(f"    Non-zero entries: {self.user_item_matrix.nnz:,}")

        return self.user_item_matrix

    def create_implicit_matrix(self) -> sparse.csr_matrix:
        if self.user_item_matrix is None:
            self.create_user_item_matrix()

        print(f"Converting to implicit matrix (temporal={self.implicit_threshold}))")

        #1 jeśli rating >= threshhold
        implicit_matrix = self.user_item_matrix.copy()
        implicit_matrix.data = (implicit_matrix.data >= self.implicit_threshold).astype(np.float32)

        #usuwanie ratingów nie spełniających threshholdu
        implicit_matrix.eliminate_zeros()

        print("Implicit matrix created:")
        print(f"    Positive interactions: {implicit_matrix.nnz:,}")
        print(f"    Conversion rate: {implicit_matrix.nnz / self.n_interactions * 100:.2f}%")

        return implicit_matrix

    def create_train_val_split(self, temporal_split: bool = True) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        if self.train_data is None:
            self.load_data()

        print(f"Creating train and val split (temporal={temporal_split})")

        if temporal_split and 'timestamp' in self.train_data.columns:

            # sortowanie po timestamp
            train_sorted = self.train_data.sort_values('timestamp')

            #Wolne w chuj

            # #dla każdego użytkownika weź ostatnie interactions do validation
            # val_data = []
            # train_data = []
            #
            # for user_id in train_sorted['user_id'].unique():
            #     user_interactions = train_sorted[train_sorted['user_id'] == user_id]
            #     n_val = max(1, int(len(user_interactions) * self.validation_split))
            #
            #     val_data.append(user_interactions.tail(n_val))
            #     train_data.append(user_interactions.head(-n_val))
            #
            # val_df = pd.concat(val_data, ignore_index=True)
            # train_df = pd.concat(train_data, ignore_index=True)

            #Szybsze i czystszy kod, może ten u góry się jeszcze do czegoś przydać potem tho

            split_idx = int(len(train_sorted) * (1 - self.validation_split))
            train_df = train_sorted.iloc[:split_idx]
            val_df = train_sorted.iloc[split_idx:]

        else:
            #random split na wszelki wypadek
            train_df, val_df = train_test_split(
                self.train_data,
                test_size=self.validation_split,
                random_state=42,
                stratify=self.train_data['user_id']
            )

        #tworzenie macierzy
        def create_matrix(df):
            users = df['user_id'].values
            item_ids = df['item_id'].values
            ratings = (df['rating'].values >= self.implicit_threshold).astype(np.float32)
            return sparse.csr_matrix((ratings, (users, item_ids)), shape=(self.n_users, self.n_items))

        self.train_matrix = create_matrix(train_df)
        self.val_matrix = create_matrix(val_df)

        print("Train/VAL split created")
        print(f"    Train interactions: {self.train_matrix.nnz:,}")
        print(f"    Val interactions: {self.val_matrix.nnz:,}")

        return self.train_matrix, self.val_matrix

    def negative_sampling(self, matrix: sparse.csr_matrix, n_negative: int = None) -> List[Tuple[int, int, float]]:
        if n_negative is None:
            n_negative = int(matrix.nnz * self.negative_sampling_ratio)

        print(f"Generating {n_negative} negative samples")

        positive_pairs = set(zip(*matrix.nonzero()))

        negative_samples = []

        attempts = 0
        max_attempts = 10 * n_negative

        while len(negative_samples) < n_negative and attempts < max_attempts:
            user = random.randint(0, self.n_users - 1)
            item = random.randint(0, self.n_items - 1)

            if (user, item) not in positive_pairs:
                negative_samples.append((user, item, 0.0))

            attempts += 1

        rows, cols = matrix.nonzero()
        positive_samples = [(r, c, 1.0) for r, c in zip(rows, cols)]

        print("Generated samples:")
        print(f"    Positive: {len(positive_samples):,}")
        print(f"    Negative: {len(negative_samples):,}")

        all_samples = positive_samples + negative_samples
        random.shuffle(all_samples)

        return all_samples

class RecommendationDatasetTorch(Dataset):
    """Wrapper dla torcha do danych treningowych"""
    def __init__(self, samples: List[Tuple[int, int, float]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_id, item_id, rating = self.samples[idx]
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float32)
        }

class RecommendationDataLoader:
    """
    Interface do ładowania gówna treingowego

    get_train_loader: dataloader dla training setu
    get_val_loader: dataloader dla validation setu
    get_test_loader: Lista idków userów do generowania rekomendacji

    """
    def __init__(self, dataset: RecommendationDataset, batch_size: int = 1024,
                 num_workers: int = 4, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def get_train_loader(self) -> DataLoader:
        if self.dataset.train_matrix is None:
            self.dataset.create_train_val_split()

        train_samples = self.dataset.negative_sampling(self.dataset.train_matrix)
        train_dataset = RecommendationDatasetTorch(train_samples)

        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle, pin_memory=True)

    def get_val_loader(self) -> DataLoader:
        if self.dataset.val_matrix is None:
            self.dataset.create_train_val_split()

        rows, cols = self.dataset.val_matrix.nonzero()
        val_samples = [(r, c, 1.0) for r, c in zip(rows, cols)]
        val_dataset = RecommendationDatasetTorch(val_samples)

        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def get_test_loader(self) -> List[int]:
        if self.dataset.test_data is None:
            self.dataset.load_data()

        return self.dataset.test_data['user_id'].to_list()