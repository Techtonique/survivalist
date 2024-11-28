from sklearn.linear_model import LinearRegression
from ..ensemble import PIComponentwiseGenGradientBoostingSurvivalAnalysis

__all__ = ["PISurvivalCustom"]


class PISurvivalCustom(PIComponentwiseGenGradientBoostingSurvivalAnalysis):
    """Generic Gradient boosting with any base learner (with prediction intervals).

    Parameters
    ----------
    regr : object, default: LinearRegression()
        The base learner that will be used for the regression model.
        The default is LinearRegression.

    loss : {'coxph', 'squared', 'ipcwls'}, optional, default: 'coxph'
        loss function to be optimized. 'coxph' refers to partial likelihood loss
        of Cox's proportional hazards model. The loss 'squared' minimizes a
        squared regression loss that ignores predictions beyond the time of censoring,
        and 'ipcwls' refers to inverse-probability of censoring weighted least squares error.

    level : int, default: 95
        Confidence level for the prediction intervals.

    type_pi: str, optional, default: 'scp'
        Type of prediction intervals. The value must be one of 'scp'
        (Split Conformal Prediction), 'bootstrap', 'kde', 'normal', 'ecdf',
        'permutation', 'smooth-bootstrap'

    random_state : int seed, RandomState instance, or None, default: None
        The seed of the pseudo random number generator to use when
        shuffling the data.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while.
        Values must be in the range `[0, inf)`.

    Attributes
    ----------
    estimators_ : list of base learners
        The collection of fitted sub-estimators.

    train_score_ : ndarray, shape = (n_estimators,)
        The i-th score ``train_score_[i]`` is the loss of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the loss on the training data.

    oob_scores_ : ndarray of shape (n_estimators,)
        The full history of the loss values on the out-of-bag
        samples. Only available if ``subsample < 1.0``.

    oob_score_ : float
        The last value of the loss on the out-of-bag samples. It is
        the same as ``oob_scores_[-1]``. Only available if ``subsample < 1.0``.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    unique_times_ : array of shape = (n_unique_times,)
        Unique time points.

    """

    def __init__(
        self,
        *,
        regr=LinearRegression(),
        loss="coxph",
        level=95,
        type_pi="scp",
        n_replications=250,
        random_state=None,
        verbose=0,
    ):
        self.regr = regr
        self._baseline_model = self.regr
        self.loss = loss
        self.n_estimators = 1  # this
        self.learning_rate = 1.0  # this too
        self.subsample = 1.0
        self.level = level
        self.type_pi = type_pi
        self.n_replications = n_replications
        self.random_state = random_state
        self.verbose = verbose
        self.show_progress = False
        self.warm_start = False
        self.dropout_rate = 0
        self.alpha_ = 1 - level / 100
