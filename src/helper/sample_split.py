# Train-Test Split
import pandas as pd
from datetime import timedelta


def train_test_split(df: pd.DataFrame, test_month: int):
    df_copy = df.copy()
    df_copy = df_copy.sort_values("transaction_created_at_date")
    df_copy["transaction_created_at_date"] = pd.to_datetime(
        df_copy["transaction_created_at_date"]
    )

    max_date = max(df_copy["transaction_created_at_date"])
    test = max_date - timedelta(days=test_month * 30)

    train = df_copy[df_copy["transaction_created_at_date"] < test.strftime("%Y-%m-%d")]
    test = df_copy[
        (df_copy["transaction_created_at_date"] >= test.strftime("%Y-%m-%d"))
    ]

    train.drop(columns=["transaction_created_at_date"], inplace=True)
    test.drop(columns=["transaction_created_at_date"], inplace=True)

    return train, test
