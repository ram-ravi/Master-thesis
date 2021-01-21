import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_regression
from sklearn.linear_model import ElasticNetCV, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=64)

# Average CV score on the training set was: -0.5333033007184602
exported_pipeline = make_pipeline(
    StandardScaler(),
    MaxAbsScaler(),
    PCA(iterated_power=1, svd_solver="randomized"),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.75, tol=0.1)),
    PCA(iterated_power=2, svd_solver="randomized"),
    VarianceThreshold(threshold=0.005),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.35000000000000003, tol=0.1)),
    SelectPercentile(score_func=f_regression, percentile=69),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.001, max_depth=1, min_child_weight=8, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.2)),
    StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=0.1, fit_intercept=False, l1_ratio=0.25, learning_rate="invscaling", loss="squared_loss", penalty="elasticnet", power_t=10.0)),
    ElasticNetCV(l1_ratio=0.8, tol=0.1)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 64)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
