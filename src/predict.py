import pandas as pd
import numpy as np
import joblib
import os
import datetime

from .helper.enum_param import ModelStrategy
from .helper.feature_engineering import run_all_features, get_deviation_from_median


def predict_data(df: pd.DataFrame, model_type: ModelStrategy = ModelStrategy.LOGREGCV):
    print("Start run feature engineering ...")
    df_transformed = run_all_features(df)
    log10_cols = [col for col in df_transformed.columns if "log10" in col]
    df_sum_median = get_deviation_from_median(df_transformed[log10_cols])
    df_final = pd.concat(
        [
            df_sum_median,
            df_transformed[["transaction_created_at_date"]],
        ],
        axis=1,
    )
    print("Done run feature engineering!")

    print("Start predict the data ...")
    model_destination = (
        os.path.join(
            "trained_models",
            f"{model_type.value}_model_v1",
        )
        + ".pkl"
    )
    clf_model = joblib.load(model_destination)

    df_prediction = df_final.copy()
    df_prediction["anomaly_proba"] = clf_model.predict_proba(
        np.array(df_prediction["sum_median"]).reshape(-1, 1)
    )[:, 1]
    df_prediction["anomaly_class"] = clf_model.predict_proba(
        np.array(df_prediction["sum_median"]).reshape(-1, 1)
    ).argmax(1)

    df_result = pd.concat(
        [
            df.set_index("invoice_id"),
            df_prediction[["anomaly_proba", "anomaly_class"]],
        ],
        axis=1,
    )
    print("Done predict the data ...")

    return df_result


def run_predict_pipeline(
    filename: str = typer.Option(...),
    model_type: ModelStrategy = ModelStrategy.LOGREGCV,
):

    print("Start import predict data ...")
    df_predict = pd.read_csv(f"./data/{filename}.csv")
    print("Done import predict data ...")

    df_result = predict_data(df=df_predict, model_type=model_type)

    print("Start saving the result ...")
    prediction_destination = (
        os.path.join(
            "data",
            "result",
            f"result_{model_type}" + "_anomaly_detection" + f"_{datetime.date.today()}",
        )
        + ".csv"
    )

    df_result.to_csv(prediction_destination)
    print("Done saving the result!")
