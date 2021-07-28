import pandas as pd
import numpy as np
import random
import string

from functools import reduce
from sklearn.preprocessing import RobustScaler


def get_date_features(data):
    df = data.copy()
    df["transaction_created_at"] = pd.to_datetime(df["transaction_created_at"])
    df["transaction_created_at_date"] = df["transaction_created_at"].dt.date
    return df


def get_delta_total_price(data):
    df = get_date_features(data)
    df["delta_total_price"] = (
        df.sort_values(["buyer_id", "transaction_created_at"])
        .groupby(["buyer_id"])["total_price"]
        .diff()
    )
    df["delta_total_price"] = np.abs(df["delta_total_price"])
    df["delta_total_price"] = df["delta_total_price"].fillna(
        df["delta_total_price"].median()
    )
    df["log10_delta_total_price"] = np.log10(df["delta_total_price"] + 1)
    return df[
        [
            "invoice_id",
            "buyer_id",
            "transaction_created_at_date",
            "log10_delta_total_price",
        ]
    ]


def get_diffsec_transaction(data):
    df = get_date_features(data)
    df["diff"] = (
        df.sort_values(["buyer_id", "transaction_created_at"])
        .groupby(["buyer_id"])["transaction_created_at"]
        .diff()
    )
    df["delta_time_trx_in_sec"] = (
        df["diff"].dt.total_seconds().fillna(df["diff"].dt.total_seconds().median())
    )
    df["log10_delta_time_trx_in_sec"] = np.log10(df["delta_time_trx_in_sec"] + 1)
    return df[
        [
            "invoice_id",
            "buyer_id",
            "transaction_created_at_date",
            "log10_delta_time_trx_in_sec",
        ]
    ]


def run_all_features(data):
    df = get_date_features(data)

    df.sort_values(by=["buyer_id", "transaction_created_at"], inplace=True)
    delta_time_trx = get_diffsec_transaction(df)
    delta_total_price = get_delta_total_price(df)
    df["log10_total_price"] = np.log10(df["total_price"] + 1)

    data_frame = [delta_time_trx, delta_total_price]
    df_transformed = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=["invoice_id", "buyer_id", "transaction_created_at_date"],
            how="left",
        ),
        data_frame,
    )

    df_transformed = df[
        ["buyer_id", "transaction_created_at_date", "invoice_id", "log10_total_price"]
    ].merge(
        df_transformed,
        on=["invoice_id", "buyer_id", "transaction_created_at_date"],
        how="left",
    )

    df_transformed.set_index("invoice_id", inplace=True)
    df_transformed.drop("buyer_id", axis=1, inplace=True)

    return df_transformed


# wrap up in function
def get_deviation_from_median(data):
    """
    data : pd.Series
    log10 scale feature for the input of model
    """
    index = data.index
    robust = RobustScaler()
    data = pd.DataFrame(robust.fit_transform(data))

    data.index = index

    distance = data - data.median()
    df = np.power(distance, 2)  # times square
    df = pd.DataFrame(df.sum(axis=1), columns=["sum_median"])  # sum of square
    return df
