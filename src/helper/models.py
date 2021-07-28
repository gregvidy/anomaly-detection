from sklearn import linear_model, ensemble, calibration


LINEAR_MODEL = {
    "logreg_cv": linear_model.LogisticRegressionCV(
        max_iter=300,
        cv=3,
        class_weight="balanced",
        scoring="roc_auc",
        random_state=123,
        solver="liblinear",
        n_jobs=-1,
    )
}

ENSEMBLE_MODEL = {
    "ensemble_rf": ensemble.RandomForestClassifier(
        n_estimators=1000, max_depth=6, n_jobs=-1, verbose=2, random_state=123
    ),
}

CALIBRATED_CLF = {
    "calibrated_clf_rf": calibration.CalibratedClassifierCV(
        ENSEMBLE_MODEL["ensemble_rf"], method="isotonic", cv=3
    ),
}
