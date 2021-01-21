import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.linear_model import ElasticNetCV, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=64)

# Average CV score on the training set was: -57.69479623463843
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.25, tol=0.1)),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.001, max_depth=6, min_child_weight=10, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.35000000000000003)),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=SGDRegressor(alpha=0.0, eta0=0.01, fit_intercept=False, l1_ratio=0.5, learning_rate="invscaling", loss="epsilon_insensitive", penalty="elasticnet", power_t=10.0)),
    SelectFwe(score_func=f_regression, alpha=0.026000000000000002),
    RandomForestRegressor(bootstrap=False, max_features=0.35000000000000003, min_samples_leaf=18, min_samples_split=18, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 64)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
