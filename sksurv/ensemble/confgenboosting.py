import numpy as np
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from .genboosting import ComponentwiseGenGradientBoostingSurvivalAnalysis
from .survival_loss import LOSS_FUNCTIONS

__all__ = ["PIComponentwiseGenGradientBoostingSurvivalAnalysis"]

class PIComponentwiseGenGradientBoostingSurvivalAnalysis(ComponentwiseGenGradientBoostingSurvivalAnalysis):
    """Generic Gradient boosting with any base learner with prediction intervals.
    
    Parameters
    ----------
    loss : {'coxph', 'squared', 'ipcwls'}, optional, default: 'coxph'
        loss function to be optimized. 'coxph' refers to partial likelihood loss
        of Cox's proportional hazards model. The loss 'squared' minimizes a
        squared regression loss that ignores predictions beyond the time of censoring,
        and 'ipcwls' refers to inverse-probability of censoring weighted least squares error.

    learning_rate : float, optional, default: 0.1
        learning rate shrinks the contribution of each base learner by `learning_rate`.
        There is a trade-off between `learning_rate` and `n_estimators`.
        Values must be in the range `[0.0, inf)`.

    n_estimators : int, default: 100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range `[1, inf)`.

    subsample : float, optional, default: 1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.
        Values must be in the range `(0.0, 1.0]`.
    
    level : int, optional, default: 95
        Confidence level for the prediction intervals. The value must be in the range `(0, 100)`.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution.

    dropout_rate : float, optional, default: 0.0
        If larger than zero, the residuals at each iteration are only computed
        from a random subset of base learners. The value corresponds to the
        percentage of base learners that are dropped. In each iteration,
        at least one base learner is dropped. This is an alternative regularization
        to shrinkage, i.e., setting `learning_rate < 1.0`.
        Values must be in the range `[0.0, 1.0)`.

    random_state : int seed, RandomState instance, or None, default: None
        The seed of the pseudo random number generator to use when
        shuffling the data.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while.
        Values must be in the range `[0, inf)`.
    
    show_progress : bool, default: True
        If set, show a progress bar for the fitting process.

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
    
    calibrated_risk_scores_ : array, shape = (n_calib_samples,)
        Predicted risk scores on the calibration set.
    
    calibrated_residuals_ : array, shape = (n_calib_samples,)
        Residuals of the calibrated risk scores on the calibration set.
    
    alpha_ : float
        Risk level for the prediction intervals.
    
    quantiles_ : float
        Quantile of absolute calibrated residuals.

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
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        level=95,
        warm_start=False,
        dropout_rate=0,
        random_state=None,
        verbose=0,
        show_progress=True,
    ):
        self.regr = regr 
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.level = level
        self.alpha_ = 1 - level / 100
        self.warm_start = warm_start
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.verbose = verbose
        self.show_progress = show_progress
        self.baseline_model = None
        self.calibrated_risk_scores_ = None
        self.calibrated_residuals_ = None
        self.quantiles_ = None
        super().__init__(regr=self.regr,
                        loss=self.loss,
                        learning_rate=self.learning_rate,
                        n_estimators=self.n_estimators,
                        subsample=self.subsample,
                        warm_start=self.warm_start,
                        dropout_rate=self.dropout_rate,
                        random_state=self.random_state,
                        verbose=self.verbose,
                        show_progress=self.show_progress)

    def fit(self, X, y, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        sample_weight : array-like, shape = (n_samples,), optional
            Weights given to each sample. If omitted, all samples have weight 1.

        Returns
        -------
        self
        """        
        self.loss_obj = LOSS_FUNCTIONS[self.loss]()
        X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.5, 
                                                              random_state=self.random_state) 
        self.X_train_ = X_train  
        self.obj_train = ComponentwiseGenGradientBoostingSurvivalAnalysis(regr=self.regr,
                                                                loss=self.loss,
                                                                learning_rate=self.learning_rate,
                                                                n_estimators=self.n_estimators,
                                                                subsample=self.subsample,
                                                                warm_start=self.warm_start,
                                                                dropout_rate=self.dropout_rate,
                                                                random_state=self.random_state,
                                                                verbose=self.verbose,
                                                                show_progress=self.show_progress)
        self.obj_train.fit(X_train, y_train, sample_weight=sample_weight)
        self.calibrated_risk_scores_ = self.obj_train.predict(X_calib)         
        risk_scores_calib = self.obj_train.fit(X_calib, y_calib, sample_weight=sample_weight).predict(X_calib)
        self.calibrated_residuals_ = np.abs(self.calibrated_risk_scores_ - risk_scores_calib)
        self.quantiles_ = np.quantile(self.calibrated_residuals_, 1 - self.alpha_)        
        return self

    def predict(self, X, **kwargs):
        """Predict risk scores.

        If `loss='coxph'`, predictions can be interpreted as log hazard ratio
        corresponding to the linear predictor of a Cox proportional hazards
        model. If `loss='squared'` or `loss='ipcwls'`, predictions are the
        time to event.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        Returns
        -------
        risk_score : array, shape = (n_samples,)
            Predicted risk scores.
        """
        assert self.quantiles_ is not None, "Model not fitted yet."
        preds = self.obj_train.predict(X, **kwargs)
        DescribeResult = namedtuple("DescribeResult", ["mean", "lower", "upper"])
        return DescribeResult(preds, preds - self.quantiles_, preds + self.quantiles_)

    def predict_cumulative_hazard_function(self, X, return_array=False, **kwargs):
        """Predict cumulative hazard function.
        """
        assert self.quantiles_ is not None, "Model not fitted yet."
        preds = self.predict(X, **kwargs)
        result = namedtuple("DescribeResult", 
                                    ["mean", "lower", "upper"])
        result.mean = self.obj_train.predict_cumulative_hazard_function(preds.mean, 
                                                                        return_array=return_array, 
                                                                        **kwargs)
        result.lower = self.obj_train.predict_cumulative_hazard_function(preds.lower, 
                                                                         return_array=return_array, 
                                                                         **kwargs)
        result.upper = self.obj_train.predict_cumulative_hazard_function(preds.upper, 
                                                                         return_array=return_array, 
                                                                         **kwargs)
        return result

    def predict_survival_function(self, X, return_array=False, **kwargs):
        """Predict survival function.
        """
        assert self.quantiles_ is not None, "Model not fitted yet."
        preds = self.predict(X, **kwargs)
        result = namedtuple("DescribeResult", 
                                    ["mean", "lower", "upper"])
        result.mean = self._predict_survival_function(self.obj_train.get_baseline_model(), 
                                        preds.mean,
                                        return_array, **kwargs)
        result.lower = self._predict_survival_function(self.obj_train.get_baseline_model(), 
                                         preds.upper,
                                         return_array, **kwargs)
        result.upper = self._predict_survival_function(self.obj_train.get_baseline_model(),
                                            preds.lower,
                                            return_array, **kwargs)
        return result


        
        

