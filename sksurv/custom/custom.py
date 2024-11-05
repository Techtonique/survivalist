from sklearn.linear_model import LinearRegression
from ..ensemble import ComponentwiseGenGradientBoostingSurvivalAnalysis

__all__ = ["SurvivalCustom"]

class SurvivalCustom(ComponentwiseGenGradientBoostingSurvivalAnalysis):
    """Generic Gradient boosting with any base learner.
    
    Parameters
    ----------
    loss : {'coxph', 'squared', 'ipcwls'}, optional, default: 'coxph'
        loss function to be optimized. 'coxph' refers to partial likelihood loss
        of Cox's proportional hazards model. The loss 'squared' minimizes a
        squared regression loss that ignores predictions beyond the time of censoring,
        and 'ipcwls' refers to inverse-probability of censoring weighted least squares error.

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

    oob_improvement_ : ndarray, shape = (n_estimators,)
        The improvement in loss on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.
        Only available if ``subsample < 1.0``.

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

    References
    ----------
    .. [1] Hothorn, T., Bühlmann, P., Dudoit, S., Molinaro, A., van der Laan, M. J.,
           "Survival ensembles", Biostatistics, 7(3), 355-73, 2006
    """

    def __init__(
        self,
        *,
        regr=LinearRegression(),
        loss="coxph",
        random_state=None,
        verbose=0,
    ):
        self.regr = regr 
        self.loss = loss
        self.n_estimators = 1
        self.learning_rate = 1.0
        self.subsample = 1.0
        self.random_state = random_state
        self.verbose = verbose
        self.show_progress = False 
        self.warm_start=False
        self.dropout_rate=0
