ANOMALY_FRAC_SIZE:= 7
ANOMALY_SIZE_PER_FRAC:= 50

# train model
train-model-logreg:
	anomaly-detection train-model run-model-pipeline \
	--filename anomaly_sample_train \
	--model-type logreg_cv \
	--anomaly-frac-size ${ANOMALY_FRAC_SIZE} \
	--anomaly-size-per-frac ${ANOMALY_SIZE_PER_FRAC}

train-model-ensemble-rf:
	anomaly-detection train-model run-model-pipeline \
	--filename anomaly_sample_train \
	--model-type ensemble_rf \
	--anomaly-frac-size ${ANOMALY_FRAC_SIZE} \
	--anomaly-size-per-frac ${ANOMALY_SIZE_PER_FRAC}

# predict data
predict-model-logreg:
	anomaly-detection predict-data run-predict-pipeline \
	--filename anomaly_sample_predict \
	--model-type logreg_cv

predict-model-ensemble-rf:
	anomaly-detection predict-data run-predict-pipeline \
	--filename anomaly_sample_predict \
	--model-type ensemble_rf