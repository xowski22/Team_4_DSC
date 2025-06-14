import torch
import torch.nn as nn
from typing import List, Tuple

class RecommenderModel(nn.Module):

    def __init__(self, n_users: int, n_items: int, embedding_dim: int):
        super(RecommenderModel, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        #Embeddingi
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)

        #Bias
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        #inicjalizacja wag
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):

        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        interaction = (user_emb * item_emb).sum(dim=1)

        user_b = self.user_bias(user_ids)
        item_b = self.item_bias(item_ids)

        prediction = interaction + user_b + item_b + self.global_bias

        return prediction

    def prodict_probability(self, user_ids, item_ids):

        logits = self.forward(user_ids, item_ids)
        return torch.sigmoid(logits)

    def recommend_top_k(self, user_id: int, k: int = 10, device = 'cuda') -> List[int]:

        self.eval()
        self.to(self.device)

        batch_size = 4096
        all_scores = []

        with torch.no_grad():
            for start_idx in range(0, self.n_items, batch_size):
                end_idx = min(start_idx + batch_size, self.n_items)
                batch_items = torch.arange(start_idx, end_idx, device=device)
                batch_users = torch.full((len(batch_items),), user_id, device=device)

                scores = self.predict_proba(batch_items, batch_users)
                all_scores.append(scores)

        all_scores = torch.cat(all_scores)

        _, top_items = torch.topk(all_scores, k=k)
        return top_items.cpu().tolist()