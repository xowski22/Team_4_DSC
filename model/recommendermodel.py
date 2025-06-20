import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

class RecommenderModel(nn.Module):

    def __init__(self, n_users: int, n_items: int, embedding_dim: int, n_categories: int):
        super(RecommenderModel, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_categories = n_categories

        #Embeddingi
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        self.category_embeddings = nn.Embedding(n_categories, embedding_dim // 4)

        #Bias
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        #inicjalizacja wag
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.category_embeddings.weight, std=0.001)

        self.dropout = nn.Dropout(0.2)

    def forward(self, user_ids, item_ids, category_ids):

        user_emb = self.dropout(self.user_embeddings(user_ids))
        item_emb = self.dropout(self.item_embeddings(item_ids))
        cat_emb = self.category_embeddings(category_ids)

        interaction = (user_emb * item_emb).sum(dim=1)

        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        cat_score = cat_emb.sum(dim=1)

        prediction = interaction + user_b + item_b + self.global_bias + cat_score * 0.1

        return prediction

    def predict_probability(self, user_ids, item_ids, category_ids):

        logits = self.forward(user_ids, item_ids, category_ids)
        return torch.sigmoid(logits)

    def recommend_top_k(self, user_id: int, item_categories, k: int = 10, device = 'cuda') -> List[int]:

        self.eval()
        self.to(device)

        batch_size = 8192
        all_scores = []

        with torch.no_grad():
            for start_idx in range(0, self.n_items, batch_size):
                end_idx = min(start_idx + batch_size, self.n_items)
                batch_items = torch.arange(start_idx, end_idx, device=device)
                batch_users = torch.full((len(batch_items),), user_id, device=device)
                batch_categories = torch.tensor(item_categories[start_idx:end_idx], device=device)

                scores = self.predict_probability(batch_users, batch_items, batch_categories)
                all_scores.append(scores)

        all_scores = torch.cat(all_scores)

        _, top_items = torch.topk(all_scores, k=k)
        return top_items.cpu().tolist()

    def cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:

        vec1_norm = F.normalize(vec1, dim=-1, p=2)
        vec2_norm = F.normalize(vec2, dim=-1, p=2)

        return torch.matmul(vec1_norm, vec2_norm.T if vec2_norm.dim() > 1 else vec2_norm)

    def find_similar_users(self, user_id: int, k: int = 10, device = 'cuda') -> List[Tuple[int, float]]:
        self.eval()
        self.to(device)

        with torch.no_grad():
            target_user_emb = self.user_embeddings(torch.tensor([user_id], device=device))
            all_user_embs = self.user_embeddings.weight

            similarities = self.cosine_similarity(target_user_emb.squeeze(), all_user_embs)
            similarities[user_id] = -1.0

            top_scores, top_users = torch.topk(similarities, k=k)

            similar_users = [
                (user_idx.item(), score.item())
                for score, user_idx in zip(top_scores, top_users)
            ]

        return similar_users

    def find_similar_items(self, item_id: int, k: int = 10, device='cuda') -> List[Tuple[int, float]]:
        self.eval()
        self.to(device)

        with torch.no_grad():
            target_item_emb = self.item_embeddings(torch.tensor([item_id], device=device))
            all_item_embs = self.item_embeddings.weight

            similarities = self.cosine_similarity(target_item_emb.squeeze(), all_item_embs)
            similarities[item_id] = -1.0

            top_scores, top_items = torch.topk(similarities, k=k)

            similar_items = [
                (item_idx.item(), score.item())
                for score, item_idx in zip(top_scores, top_items)
            ]

        return similar_items

    def recommend_by_similarity(self, user_id: int, k: int = 10,
                                similar_users_count: int = 50, device = 'cuda') -> List[int]:
        self.eval()
        self.to(device)

        similar_users = self.find_similar_users(user_id, similar_users_count, device=device)

        item_scores = {}

        for similar_user_id, similarity_score in similar_users:
            user_recs = self.recommend_top_k(user_id, k=20, device=device)

            for item_id, score in user_recs:
                if item_id not in item_scores:
                    item_scores[item_id] = 0.0
                item_scores[item_id] += similarity_score

        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

        return [item_id for item_id, _ in sorted_items[:k]]