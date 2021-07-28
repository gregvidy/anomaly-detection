from enum import Enum


class ModelStrategy(str, Enum):
    LOGREGCV = "logreg_cv"
    ENSEMBLE_RF = "ensemble_rf"
    ENSEMBLE_XGBOOST = "ensemble_xgboost"
    CALIBRATED_CLF_RF = "calibrated_clf_rf"
    CALIBRATED_CLF_XGBOOST = "calibrated_clf_xgboost"
