# pylint: disable=wildcard-import,invalid-name,eval-used,unused-wildcard-import
import os
import json
import shutil

from pyspark.ml.feature import StringIndexerModel, IndexToString
from os.path import exists, join

from replay.models import *
from replay.models.base_rec import BaseRecommender
from replay.session_handler import State


def save(model: BaseRecommender, path: str):
    """
    Save fitted model to disk as a folder

    :param model: Trained recommender
    :param path: destination where model files will be stored
    :return:
    """
    if exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    model._save_model(join(path, "model"))

    init_args = model._init_args
    init_args["_model_name"] = str(model)
    with open(join(path, "init_args.json"), "w") as json_file:
        json.dump(init_args, json_file)

    model.user_indexer.save(join(path, "user_indexer"))
    model.item_indexer.save(join(path, "item_indexer"))
    model.inv_user_indexer.save(join(path, "inv_user_indexer"))
    model.inv_item_indexer.save(join(path, "inv_item_indexer"))

    dataframes = model._dataframes
    df_path = join(path, "dataframes")
    os.makedirs(df_path)
    for name, df in dataframes.items():
        df.rdd.saveAsPickleFile(join(df_path, name))


def load(path: str):
    """
    Load saved model from disk

    :param path: path to model folder
    :return: Restored trained model
    """
    spark = State().session
    with open(join(path, "init_args.json"), "r") as json_file:
        init_args = json.load(json_file)
    name = init_args["_model_name"]
    del init_args["_model_name"]
    model = eval(f"{name}(**{str(init_args)})")

    model.user_indexer = StringIndexerModel.load(join(path, "user_indexer"))
    model.item_indexer = StringIndexerModel.load(join(path, "item_indexer"))
    model.inv_user_indexer = IndexToString.load(join(path, "inv_user_indexer"))
    model.inv_item_indexer = IndexToString.load(join(path, "inv_item_indexer"))

    df_path = join(path, "dataframes")
    dataframes = os.listdir(df_path)
    for name in dataframes:
        pickle_rdd = spark.sparkContext.pickleFile(
            join(df_path, name)
        ).collect()
        df = spark.createDataFrame(pickle_rdd).cache()
        setattr(model, name, df)

    model._load_model(join(path, "model"))
    return model
