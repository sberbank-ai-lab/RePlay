from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
import torch
from torch import nn

from replay.constants import REC_SCHEMA
from replay.models.base_rec import Recommender
from replay.session_handler import State


class TorchRecommender(Recommender):
    """Base class for neural recommenders"""

    model: Any
    device: torch.device

    def __init__(self):
        self.logger.info(
            "The model is neural network with non-distributed training"
        )
        self.checkpoint_path = State().session.conf.get("spark.local.dir")
        self.device = State().device

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        items_consider_in_pred = items.toPandas()["item_idx"].values
        items_count = self._item_dim
        model = self.model.cpu()
        agg_fn = self._predict_by_user

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            return agg_fn(
                pandas_df, model, items_consider_in_pred, k, items_count
            )[["user_idx", "item_idx", "relevance"]]

        self.logger.debug("Predict started")
        # do not apply map on cold users for MultVAE predict
        join_type = "inner" if self.__str__() == "MultVAE" else "left"
        recs = (
            users.join(log, how=join_type, on="user_idx")
            .select("user_idx", "item_idx")
            .groupby("user_idx")
            .applyInPandas(grouped_map, REC_SCHEMA)
        )
        return recs

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        items_count = self._item_dim
        model = self.model.cpu()
        agg_fn = self._predict_by_user_pairs
        users = pairs.select("user_idx").distinct()

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            return agg_fn(pandas_df, model, items_count)[
                ["user_idx", "item_idx", "relevance"]
            ]

        self.logger.debug("Calculate relevance for user-item pairs")
        user_history = (
            users.join(log, how="inner", on="user_idx")
            .groupBy("user_idx")
            .agg(sf.collect_list("item_idx").alias("item_idx_history"))
        )
        user_pairs = pairs.groupBy("user_idx").agg(
            sf.collect_list("item_idx").alias("item_idx_to_pred")
        )
        full_df = user_pairs.join(user_history, on="user_idx", how="inner")

        recs = full_df.groupby("user_idx").applyInPandas(
            grouped_map, REC_SCHEMA
        )

        return recs

    @staticmethod
    @abstractmethod
    def _predict_by_user(
        pandas_df: pd.DataFrame,
        model: nn.Module,
        items_np: np.ndarray,
        k: int,
        item_count: int,
    ) -> pd.DataFrame:
        """
        Calculate predictions.

        :param pandas_df: DataFrame with user-item interactions ``[user_idx, item_idx]``
        :param model: trained model
        :param items_np: items available for recommendations
        :param k: length of recommendation list
        :param item_count: total number of items
        :return: DataFrame ``[user_idx , item_idx , relevance]``
        """

    @staticmethod
    @abstractmethod
    def _predict_by_user_pairs(
        pandas_df: pd.DataFrame,
        model: nn.Module,
        item_count: int,
    ) -> pd.DataFrame:
        """
        Get relevance for provided pairs

        :param pandas_df: DataFrame with rated items and items that need prediction
            ``[user_idx, item_idx_history, item_idx_to_pred]``
        :param model: trained model
        :param item_count: total number of items
        :return: DataFrame ``[user_idx , item_idx , relevance]``
        """

    def load_model(self, path: str) -> None:
        """
        Load model from file

        :param path: path to model
        :return:
        """
        self.logger.debug("-- Loading model from file")
        self.model.load_state_dict(torch.load(path))

    def _save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
