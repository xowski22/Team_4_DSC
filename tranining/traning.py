import torch
import torch.nn as nn
from typing import Dict, Any
import os

class Trainer:

    def __init__(self, model, device='cuda', lr=0.001, weight_decay=1e-5, save_path='./'):
        self.model = model.to(device)
        self.device = device
        self.save_path = save_path

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []

        print(f"Training on {device}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, train_loader):
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):

            user_ids = batch['user_id'].to(self.device, non_blocking=True)
            item_ids = batch['item_id'].to(self.device, non_blocking=True)
            ratings = batch['rating'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)}, loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    def validate(self, val_loader) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch['user_id'].to(self.device, non_blocking=True)
                item_ids = batch['item_id'].to(self.device, non_blocking=True)
                ratings = batch['rating'].to(self.device, non_blocking=True)

                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings)

                total_loss += loss.item()
                num_batches += 1

                probs = torch.sigmoid(predictions)
                predicted = (probs > 0.5).float()
                correct += (predicted == ratings).sum().item()
                total += ratings.size(0)

        avg_loss = total_loss / num_batches
        accuracy = correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        if metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            best_path = os.path.join(self.save_path, 'model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved. Val loss: {metrics['val_loss']:.4f}")
            return True

        return False

    def train(self, train_loader, val_loader, epochs = 20, verbose=True):
        print(f"Training started for {epochs} epochs")

        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}")

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }

            self.train_history.append(train_metrics['loss'])
            self.val_history.append(val_metrics['loss'])

            if verbose:
                print(f"\tTrain loss: {epoch_metrics['train_loss']:.4f}")
                print(f"\tVal loss: {epoch_metrics['val_loss']:.4f}")
                print(f"\tVal accuracy: {epoch_metrics['val_accuracy']:.4f}")

            self.save_checkpoint(epoch, epoch_metrics)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Training finished. Best val loss {self.best_val_loss:.4f}")

        return{
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
        }

    def load_best_model(self):
        best_path = os.path.join(self.save_path, 'model.pth')

        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
            return checkpoint['metrics']
        else:
            print("No best model found")
            return None

def quick_train(model, train_loader, val_loader, device='cuda', epochs=20, lr=0.001):
    trainer = Trainer(model, device=device, lr=lr)
    results = trainer.train(train_loader, val_loader, epochs=epochs)
    trainer.load_best_model()

    return trainer, results