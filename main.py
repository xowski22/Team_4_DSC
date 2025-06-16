import torch
import pandas as pd
import sys
from pathlib import Path

sys.path.append("./")

from data.dataset import RecommendationDataset, RecommendationDataLoader
from model.recommendermodel import RecommenderModel
from tranining.traning import quick_train

def main():

    data_path = ("./data")

    embedding_dim = 64
    batch_size = 1024
    epochs = 20
    learning_rate = 0.001
    implicit_threshold = 4.0
    negative_sampling_ratio = 4.0
    validation_split = 0.2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    if torch.cuda.is_available():
        print(f"\tGPU: {torch.cuda.get_device_name(0)}")

    dataset = RecommendationDataset(
        data_path,
        implicit_threshold=implicit_threshold,
        negative_sampling_ratio=negative_sampling_ratio,
        validation_split=validation_split
    )

    dataset.load_data()
    dataset.create_train_val_split(temporal_split=True)

    data_loader = RecommendationDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )

    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_users = data_loader.get_test_loader()

    print(f"Data ready - {len(train_loader)} train batches, {len(val_loader)} val batches")

    model = RecommenderModel(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=embedding_dim
    )

    try:
        trainer, results = quick_train(model, train_loader, val_loader, device, epochs, learning_rate)
    except Exception as e:
        print(f"Training failed: {e}")
        return

    model.eval()
    model = model.to(device)

    predictions = []

    for i, user_id in enumerate(test_users):
        top_items = model.recommend_top_k(user_id, k=10, device=device)
        predictions.append(' '.join(map(str, top_items)))

        if i % 5000 == 0 and i > 0:
            print(f"\tProgress: {i:,}/{len(test_users):,})")

    submission = pd.DataFrame({'user_id': test_users, 'predictions': predictions})
    submission.to_csv("submission.csv", index=False)
    print(f"Submission saved: {submission.shape[0]:,} predictions")

if __name__ == '__main__':
    main()