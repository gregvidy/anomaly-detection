import pandas as pd
import numpy as np
import os
import pickle
import typer

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from .helper.anomaly_injection import inject_artifical_anomaly
from .helper.feature_engineering import run_all_features, get_deviation_from_median
from .helper.sample_split import train_test_split
from .helper.models import LINEAR_MODEL, ENSEMBLE_MODEL, CALIBRATED_CLF
from .helper.enum_param import ModelStrategy

app = typer.Typer()


def train_model(
    df: pd.DataFrame,
    anomaly_frac_size: int = 7,
    anomaly_size_per_frac: int = 50,
    model_type: ModelStrategy = ModelStrategy.LOGREGCV,
):

    print("Start run feature engineering ...")
    df_transformed = run_all_features(df)
    print("Done run feature engineering!")

    # anomaly injection
    print("Start anomaly injection ...")
    df_with_anomaly = inject_artifical_anomaly(
        df=df_transformed,
        frac_slice=anomaly_frac_size,
        anomaly_size=anomaly_size_per_frac,
    )
    log10_cols = [col for col in df_with_anomaly.columns if "log10" in col]
    df_sum_median = get_deviation_from_median(df_with_anomaly[log10_cols])
    df_final = pd.concat(
        [
            df_sum_median,
            df_with_anomaly[["is_artificial", "transaction_created_at_date"]],
        ],
        axis=1,
    )
    print("Done anomaly injection!")

    # splitting data
    print("Start splitting data ...")
    train, test = train_test_split(df=df_final, test_month=2)
    X_train = train.drop(["is_artificial"], axis=1)
    y_train = train.is_artificial
    X_test = test.drop(["is_artificial"], axis=1)
    y_test = test.is_artificial
    print("Done splitting data!")

    if model_type == ModelStrategy.LOGREGCV.value:
        clf_model = LINEAR_MODEL[model_type]

    elif model_type == ModelStrategy.ENSEMBLE_RF.value:
        clf_model = ENSEMBLE_MODEL[model_type]

    elif model_type == ModelStrategy.CALIBRATED_CLF_RF.value:
        clf_model = CALIBRATED_CLF[model_type]

    # train
    print("Start training data ...")
    clf_model.fit(X_train, y_train)
    y_pred_proba = clf_model.predict_proba(X_test)[:, 1]
    y_pred_class = clf_model.predict_proba(X_test).argmax(1)

    print("------------------------------")
    print("Model Performance:")
    print("ROC_AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Recall:", recall_score(y_test, y_pred_class))
    print("Precision:", precision_score(y_test, y_pred_class))
    print("f1_score:", f1_score(y_test, y_pred_class, average="macro"))
    print("------------------------------")
    print("Done training data!")

    return clf_model


@app.command()
def run_model_pipeline(
    filename: str = typer.Option(...),
    model_type: ModelStrategy = ModelStrategy.LOGREGCV,
    anomaly_frac_size: int = 7,
    anomaly_size_per_frac: int = 50,
):

    # ingest/import data
    print("Start import data from local ...")
    df_anomaly = pd.read_csv(f"./data/{filename}.csv")
    print("Done import data from local!")

    anomaly_model = train_model(
        df=df_anomaly,
        anomaly_frac_size=anomaly_frac_size,
        anomaly_size_per_frac=anomaly_size_per_frac,
        model_type=model_type,
    )

    # set model destination
    print("Start dump the model ...")
    model_destination = os.path.join("trained_models")
    model_path = model_type + "_model_v1.pkl"
    model_destination = os.path.join(model_destination, model_path)
    print("Finish dump the model!")

    pickle.dump(anomaly_model, open(model_destination, "wb"))
