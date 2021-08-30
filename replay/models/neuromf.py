"""
Generalized Matrix Factorization (GMF),
Multi-Layer Perceptron (MLP),
Neural Matrix Factorization (MLP + GMF).
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim
from ignite.engine import Engine
from pyspark.sql import DataFrame
from sklearn.model_selection import train_test_split
from torch import LongTensor, Tensor, nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from replay.models.base_torch_rec import TorchRecommender
from replay.session_handler import State

EMBED_DIM = 128


def xavier_init_(layer: nn.Module):
    """
    Xavier initialization

    :param layer: net layer
    """
    if isinstance(layer, (nn.Embedding, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)

    if isinstance(layer, nn.Linear):
        layer.bias.data.normal_(0.0, 0.001)


class GMF(nn.Module):
    """Generalized Matrix Factorization"""

    def __init__(self, user_count: int, item_count: int, embedding_dim: int):
        """
        :param user_count: number of users
        :param item_count: number of items
        :param embedding_dim: embedding size
        """
        super().__init__()
        self.user_embedding = nn.Embedding(
            num_embeddings=user_count, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=item_count, embedding_dim=embedding_dim
        )
        self.item_biases = nn.Embedding(
            num_embeddings=item_count, embedding_dim=1
        )
        self.user_biases = nn.Embedding(
            num_embeddings=user_count, embedding_dim=1
        )

        xavier_init_(self.user_embedding)
        xavier_init_(self.item_embedding)
        self.user_biases.weight.data.zero_()
        self.item_biases.weight.data.zero_()

    # pylint: disable=arguments-differ
    def forward(self, user: Tensor, item: Tensor) -> Tensor:  # type: ignore
        """
        :param user: user id batch
        :param item: item id batch
        :return: model output
        """
        user_emb = self.user_embedding(user) + self.user_biases(user)
        item_emb = self.item_embedding(item) + self.item_biases(item)
        element_product = torch.mul(user_emb, item_emb)

        return element_product


class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(
        self,
        user_count: int,
        item_count: int,
        embedding_dim: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        """
        :param user_count: number of users
        :param item_count: number of items
        :param embedding_dim: embedding size
        :param hidden_dims: list of hidden dimension sizes
        """
        super().__init__()
        self.user_embedding = nn.Embedding(
            num_embeddings=user_count, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=item_count, embedding_dim=embedding_dim
        )
        self.item_biases = nn.Embedding(
            num_embeddings=item_count, embedding_dim=1
        )
        self.user_biases = nn.Embedding(
            num_embeddings=user_count, embedding_dim=1
        )

        if hidden_dims:
            full_hidden_dims = [2 * embedding_dim] + hidden_dims
            self.hidden_layers = nn.ModuleList(
                [
                    nn.Linear(d_in, d_out)
                    for d_in, d_out in zip(
                        full_hidden_dims[:-1], full_hidden_dims[1:]
                    )
                ]
            )

        else:
            self.hidden_layers = nn.ModuleList()

        self.activation = nn.ReLU()

        xavier_init_(self.user_embedding)
        xavier_init_(self.item_embedding)
        self.user_biases.weight.data.zero_()
        self.item_biases.weight.data.zero_()
        for layer in self.hidden_layers:
            xavier_init_(layer)

    # pylint: disable=arguments-differ
    def forward(self, user: Tensor, item: Tensor) -> Tensor:  # type: ignore
        """
        :param user: user id batch
        :param item: item id batch
        :return: output
        """
        user_emb = self.user_embedding(user) + self.user_biases(user)
        item_emb = self.item_embedding(item) + self.item_biases(item)
        hidden = torch.cat([user_emb, item_emb], dim=-1)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
            hidden = self.activation(hidden)
        return hidden


class NMF(nn.Module):
    """NMF = MLP + GMF"""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        user_count: int,
        item_count: int,
        embedding_gmf_dim: Optional[int] = None,
        embedding_mlp_dim: Optional[int] = None,
        hidden_mlp_dims: Optional[List[int]] = None,
    ):
        """
        :param user_count: number of users
        :param item_count: number of items
        :param embedding_gmf_dim: embedding size for gmf
        :param embedding_mlp_dim: embedding size for mlp
        :param hidden_mlp_dims: list of hidden dimension sizes for mlp
        """
        self.gmf: Optional[GMF] = None
        self.mlp: Optional[MLP] = None

        super().__init__()
        merged_dim = 0
        if embedding_gmf_dim:
            self.gmf = GMF(user_count, item_count, embedding_gmf_dim)
            merged_dim += embedding_gmf_dim

        if embedding_mlp_dim:
            self.mlp = MLP(
                user_count, item_count, embedding_mlp_dim, hidden_mlp_dims
            )
            merged_dim += (
                hidden_mlp_dims[-1]
                if hidden_mlp_dims
                else 2 * embedding_mlp_dim
            )

        self.last_layer = nn.Linear(merged_dim, 1)
        xavier_init_(self.last_layer)

    # pylint: disable=arguments-differ
    def forward(self, user: Tensor, item: Tensor) -> Tensor:  # type: ignore
        """
        :param user: user id batch
        :param item: item id batch
        :return: output
        """
        batch_size = len(user)
        if self.gmf:
            gmf_vector = self.gmf(user, item)
        else:
            gmf_vector = torch.zeros(batch_size, 0).to(user.device)

        if self.mlp:
            mlp_vector = self.mlp(user, item)
        else:
            mlp_vector = torch.zeros(batch_size, 0).to(user.device)

        merged_vector = torch.cat([gmf_vector, mlp_vector], dim=1)
        merged_vector = self.last_layer(merged_vector).squeeze()
        merged_vector = torch.sigmoid(merged_vector)

        return merged_vector


# pylint: disable=too-many-instance-attributes
class NeuroMF(TorchRecommender):
    """
    Neural Matrix Factorization model (NeuMF, NCF).

    In this implementation MLP and GMF modules are optional.
    """

    num_workers: int = 0
    batch_size_users: int = 100000
    trainer: Engine
    val_evaluator: Engine
    patience: int = 3
    n_saved: int = 2
    valid_split_size: float = 0.1
    seed: int = 42
    _search_space = {
        "embedding_gmf_dim": {"type": "int", "args": [EMBED_DIM, EMBED_DIM]},
        "embedding_mlp_dim": {"type": "int", "args": [EMBED_DIM, EMBED_DIM]},
        "learning_rate": {"type": "loguniform", "args": [0.0001, 0.5]},
        "l2_reg": {"type": "loguniform", "args": [1e-9, 5]},
        "gamma": {"type": "uniform", "args": [0.8, 0.99]},
        "count_negative_sample": {"type": "int", "args": [1, 20]},
    }

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 20,
        embedding_gmf_dim: Optional[int] = None,
        embedding_mlp_dim: Optional[int] = None,
        hidden_mlp_dims: Optional[List[int]] = None,
        l2_reg: float = 0,
        gamma: float = 0.99,
        count_negative_sample: int = 1,
    ):
        """
        MLP or GMF model can be ignored if
        its embedding size (embedding_mlp_dim or embedding_gmf_dim) is set to ``None``.
        Default variant is MLP + GMF with embedding size 128.

        :param learning_rate: learning rate
        :param epochs: number of epochs to train model
        :param embedding_gmf_dim: embedding size for gmf
        :param embedding_mlp_dim: embedding size for mlp
        :param hidden_mlp_dims: list of hidden dimension sized for mlp
        :param l2_reg: l2 regularization term
        :param gamma: decrease learning rate by this coefficient per epoch
        :param count_negative_sample: number of negative samples to use
        """
        if not embedding_gmf_dim and not embedding_mlp_dim:
            embedding_gmf_dim, embedding_mlp_dim = EMBED_DIM, EMBED_DIM

        if (embedding_gmf_dim is None or embedding_gmf_dim < 0) and (
            embedding_mlp_dim is None or embedding_mlp_dim < 0
        ):
            raise ValueError(
                "embedding_gmf_dim and embedding_mlp_dim must be positive"
            )

        self.device = State().device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.embedding_gmf_dim = embedding_gmf_dim
        self.embedding_mlp_dim = embedding_mlp_dim
        self.hidden_mlp_dims = hidden_mlp_dims
        self.l2_reg = l2_reg
        self.gamma = gamma
        self.count_negative_sample = count_negative_sample

    def _data_loader(
        self, data: pd.DataFrame, shuffle: bool = True
    ) -> DataLoader:

        user_batch = LongTensor(data["user_idx"].values)  # type: ignore
        item_batch = LongTensor(data["item_idx"].values)  # type: ignore

        dataset = TensorDataset(user_batch, item_batch)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size_users,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return loader

    def _get_neg_batch(self, batch: Tensor) -> Tensor:
        negative_items = torch.randint(
            0,
            self.items_count - 1,
            (batch.shape[0] * self.count_negative_sample,),
        )
        return negative_items

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.model = NMF(
            user_count=self.users_count,
            item_count=self.items_count,
            embedding_gmf_dim=self.embedding_gmf_dim,
            embedding_mlp_dim=self.embedding_mlp_dim,
            hidden_mlp_dims=self.hidden_mlp_dims,
        ).to(self.device)

        self.logger.debug("Create batch")
        tensor_data = log.select("user_idx", "item_idx").toPandas()
        train_tensor_data, valid_tensor_data = train_test_split(
            tensor_data,
            test_size=self.valid_split_size,
            random_state=self.seed,
        )
        train_data_loader = self._data_loader(train_tensor_data)
        val_data_loader = self._data_loader(valid_tensor_data)

        self.logger.debug("Train NeuroMF")
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg / self.batch_size_users,
        )
        lr_scheduler = ExponentialLR(optimizer, gamma=self.gamma)
        nmf_trainer, _ = self._create_trainer_evaluator(
            optimizer,
            val_data_loader,
            lr_scheduler,
            self.patience,
            self.n_saved,
        )

        nmf_trainer.run(train_data_loader, max_epochs=self.epochs)

    # pylint: disable=arguments-differ
    def _loss(self, y_pred, y_true):
        return F.binary_cross_entropy(y_pred, y_true).mean()

    def _batch_pass(self, batch, model):
        user_batch, pos_item_batch = batch
        neg_item_batch = self._get_neg_batch(user_batch)
        pos_relevance = model(
            user_batch.to(self.device), pos_item_batch.to(self.device)
        )
        neg_relevance = model(
            user_batch.repeat([self.count_negative_sample]).to(self.device),
            neg_item_batch.to(self.device),
        )
        y_pred = torch.cat((pos_relevance, neg_relevance), 0)
        y_true_pos = torch.ones_like(pos_item_batch).to(self.device)
        y_true_neg = torch.zeros_like(neg_item_batch).to(self.device)
        y_true = torch.cat((y_true_pos, y_true_neg), 0).float()

        return y_pred, y_true

    @staticmethod
    def _predict_pairs_inner(
        model: nn.Module,
        user_idx: int,
        items_np: np.ndarray,
        cnt: Optional[int] = None,
    ) -> DataFrame:
        model.eval()
        with torch.no_grad():
            user_batch = LongTensor([user_idx] * len(items_np))
            item_batch = LongTensor(items_np)
            user_recs = torch.reshape(
                model(user_batch, item_batch).detach(),
                [
                    -1,
                ],
            )
            if cnt is not None:
                best_item_idx = (
                    torch.argsort(user_recs, descending=True)[:cnt]
                ).numpy()
                user_recs = user_recs[best_item_idx]
                items_np = items_np[best_item_idx]

            return pd.DataFrame(
                {
                    "user_idx": user_recs.shape[0] * [user_idx],
                    "item_idx": items_np,
                    "relevance": user_recs,
                }
            )

    @staticmethod
    def _predict_by_user(
        pandas_df: pd.DataFrame,
        model: nn.Module,
        items_np: np.ndarray,
        k: int,
        item_count: int,
    ) -> pd.DataFrame:
        return NeuroMF._predict_pairs_inner(
            model=model,
            user_idx=pandas_df["user_idx"][0],
            items_np=items_np,
            cnt=min(len(pandas_df) + k, len(items_np)),
        )

    @staticmethod
    def _predict_by_user_pairs(
        pandas_df: pd.DataFrame, model: nn.Module, item_count: int
    ) -> pd.DataFrame:
        return NeuroMF._predict_pairs_inner(
            model=model,
            user_idx=pandas_df["user_idx"][0],
            items_np=np.array(pandas_df["item_idx_to_pred"][0]),
            cnt=None,
        )
