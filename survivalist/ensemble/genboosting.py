import numbers

import numpy as np
from collections import namedtuple
from sklearn.base import BaseEstimator
from sklearn.ensemble._base import BaseEnsemble
from sklearn.ensemble._gb import VerboseReporter
from sklearn.ensemble._gradient_boosting import _random_sample_mask
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from tqdm import tqdm

from ..base import SurvivalAnalysisMixin
from ..linear_model.coxph import BreslowEstimator
from ..util import check_array_survival
from .survival_loss import (
    LOSS_FUNCTIONS,
    CensoredSquaredLoss,
    CoxPH,
    IPCWLeastSquaresError,
)

__all__ = ["ComponentwiseGenGradientBoostingSurvivalAnalysis"]


def _sample_binomial_plus_one(p, size, random_state):
    drop_model = random_state.binomial(1, p=p, size=size)
    n_dropped = np.sum(drop_model)
    if n_dropped == 0:
        idx = random_state.randint(0, size)
        drop_model[idx] = 1
        n_dropped = 1
    return drop_model, n_dropped


class _ComponentwiseBaseLearner(BaseEstimator):
    def __init__(self, component, regr):
        self.component = component
        self.regr = regr

    def fit(self, X, y, sample_weight):
        try:
            self.regr.fit(X, y, sample_weight=sample_weight)
        except Exception as e:
            self.regr.fit(X, y)
        return self.regr

    def predict(self, X, **kwargs):
        return self.regr.predict(X[:, self.component], **kwargs)


def _fit_stage_componentwise(
    X, residuals, sample_weight, regr, **fit_params
):  # pylint: disable=unused-argument
    """Fit component-wise weighted least squares model"""
    n_features = X.shape[1]

    base_learners = []
    error = np.empty(n_features)
    for component in range(n_features):
        learner = _ComponentwiseBaseLearner(component, regr=regr).fit(
            X, residuals, sample_weight
        )
        l_pred = learner.predict(X)
        error[component] = squared_norm(residuals - l_pred)
        base_learners.append(learner)

    # TODO: could use bottleneck.nanargmin for speed
    return base_learners[np.nanargmin(error)]


class ComponentwiseGenGradientBoostingSurvivalAnalysis(
    BaseEnsemble, SurvivalAnalysisMixin
):
    """Generic Gradient boosting with any base learner.

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

    References
    ----------
    .. [1] Hothorn, T., Bühlmann, P., Dudoit, S., Molinaro, A., van der Laan, M. J.,
           "Survival ensembles", Biostatistics, 7(3), 355-73, 2006
    """

    _parameter_constraints = {
        "loss": [StrOptions(frozenset(LOSS_FUNCTIONS.keys()))],
        "learning_rate": [Interval(numbers.Real, 0.0, None, closed="left")],
        "n_estimators": [Interval(numbers.Integral, 1, None, closed="left")],
        "subsample": [Interval(numbers.Real, 0.0, 1.0, closed="right")],
        "warm_start": ["boolean"],
        "dropout_rate": [Interval(numbers.Real, 0.0, 1.0, closed="left")],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        *,
        regr=LinearRegression(),
        loss="coxph",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
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
        self.warm_start = warm_start
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.verbose = verbose
        self.show_progress = show_progress
        self.quantiles_ = None

    @property
    def _predict_risk_score(self):
        return isinstance(self.loss_obj, CoxPH)

    def _is_fitted(self):
        return len(getattr(self, "estimators_", [])) > 0

    def _init_state(self):
        self.estimators_ = np.empty(self.n_estimators, dtype=object)

        self.train_score_ = np.zeros(self.n_estimators, dtype=np.float64)
        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros(
                self.n_estimators, dtype=np.float64
            )
            self.oob_scores_ = np.zeros(self.n_estimators, dtype=np.float64)
            self.oob_score_ = np.nan

        if self.dropout_rate > 0:
            self._scale = np.ones(int(self.n_estimators), dtype=float)

    def _resize_state(self):
        """Add additional ``n_estimators`` entries to all attributes."""
        # self.n_estimators is the number of additional est to fit
        total_n_estimators = self.n_estimators

        self.estimators_ = np.resize(self.estimators_, total_n_estimators)
        self.train_score_ = np.resize(self.train_score_, total_n_estimators)
        if self.subsample < 1 or hasattr(self, "oob_improvement_"):
            # if do oob resize arrays or create new if not available
            if hasattr(self, "oob_improvement_"):
                self.oob_improvement_ = np.resize(
                    self.oob_improvement_, total_n_estimators
                )
                self.oob_scores_ = np.resize(
                    self.oob_scores_, total_n_estimators
                )
                self.oob_score_ = np.nan
            else:
                self.oob_improvement_ = np.zeros(
                    total_n_estimators, dtype=np.float64
                )
                self.oob_scores_ = np.zeros(
                    (total_n_estimators,), dtype=np.float64
                )
                self.oob_score_ = np.nan

        if self.dropout_rate > 0:
            if not hasattr(self, "_scale"):
                raise ValueError(
                    "fitting with warm_start=True and dropout_rate > 0 is only "
                    "supported if the previous fit used dropout_rate > 0 too"
                )

            self._scale = np.resize(self._scale, total_n_estimators)
            self._scale[self.n_estimators_:] = 1

    def _clear_state(self):
        """Clear the state of the gradient boosting model."""
        if hasattr(self, "estimators_"):
            self.estimators_ = np.empty(0, dtype=object)
        if hasattr(self, "train_score_"):
            del self.train_score_
        if hasattr(self, "oob_improvement_"):
            del self.oob_improvement_
        if hasattr(self, "oob_scores_"):
            del self.oob_scores_
        if hasattr(self, "oob_score_"):
            del self.oob_score_
        if hasattr(self, "_rng"):
            del self._rng
        if hasattr(self, "_scale"):
            del self._scale

    def _update_with_dropout(self, i, X, raw_predictions, scale, random_state):
        # select base learners to be dropped for next iteration
        drop_model, n_dropped = _sample_binomial_plus_one(
            self.dropout_rate, i + 1, random_state
        )

        # adjust scaling factor of tree that is going to be trained in next iteration
        scale[i + 1] = 1.0 / (n_dropped + 1.0)

        raw_predictions[:] = 0
        for m in range(i + 1):
            if drop_model[m] == 1:
                # adjust scaling factor of dropped trees
                scale[m] *= n_dropped / (n_dropped + 1.0)
            else:
                # pseudoresponse of next iteration (without contribution of dropped trees)
                raw_predictions += (
                    self.learning_rate
                    * scale[m]
                    * self.estimators_[m].predict(X)
                )

    def _fit(
        self,
        X,
        event,
        time,
        y_pred,
        sample_weight,
        random_state,
        begin_at_stage=0,
    ):  # noqa: C901
        n_samples = X.shape[0]
        # account for intercept
        y = np.fromiter(
            zip(event, time), dtype=[("event", bool), ("time", np.float64)]
        )

        do_oob = self.subsample < 1.0
        if do_oob:
            n_inbag = max(1, int(self.subsample * n_samples))

        do_dropout = self.dropout_rate > 0
        if do_dropout:
            scale = self._scale

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, 0)

        # perform boosting iterations
        i = begin_at_stage
        if self.show_progress:
            iterator = tqdm(range(begin_at_stage, int(self.n_estimators)))
        else:
            iterator = range(begin_at_stage, int(self.n_estimators))
        for i in iterator:
            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(
                    n_samples, n_inbag, random_state
                )
                subsample_weight = sample_weight * sample_mask.astype(
                    np.float64
                )

                # OOB score before adding this stage
                y_oob_masked = y[~sample_mask]
                sample_weight_oob_masked = sample_weight[~sample_mask]
                if i == 0:  # store the initial loss to compute the OOB score
                    initial_loss = self.loss_obj(
                        y_true=y_oob_masked,
                        raw_prediction=y_pred[~sample_mask],
                        sample_weight=sample_weight_oob_masked,
                    )
            else:
                subsample_weight = sample_weight

            residuals = self.loss_obj.gradient(
                y, y_pred, sample_weight=sample_weight
            )

            best_learner = _fit_stage_componentwise(
                X, residuals, subsample_weight, regr=self.regr
            )
            self.estimators_[i] = best_learner

            if do_dropout and i < len(scale) - 1:
                self._update_with_dropout(i, X, y_pred, scale, random_state)
            else:
                y_pred += self.learning_rate * best_learner.predict(X)

            # track loss
            if do_oob:
                self.train_score_[i] = self.loss_obj(
                    y_true=y[sample_mask],
                    raw_prediction=y_pred[sample_mask],
                    sample_weight=sample_weight[sample_mask],
                )
                self.oob_scores_[i] = self.loss_obj(
                    y_true=y_oob_masked,
                    raw_prediction=y_pred[~sample_mask],
                    sample_weight=sample_weight_oob_masked,
                )
                previous_loss = (
                    initial_loss if i == 0 else self.oob_scores_[i - 1]
                )
                self.oob_improvement_[i] = previous_loss - self.oob_scores_[i]
                self.oob_score_ = self.oob_scores_[-1]
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = self.loss_obj(
                    y_true=y, raw_prediction=y_pred, sample_weight=sample_weight
                )

            if self.verbose > 0:
                verbose_reporter.update(i, self)

        return i + 1

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
        self._validate_params()

        if not self.warm_start:
            self._clear_state()

        X = self._validate_data(X, ensure_min_samples=2)
        event, time = check_array_survival(X, y)

        sample_weight = _check_sample_weight(sample_weight, X)

        n_samples = X.shape[0]
        Xi = np.column_stack((np.ones(n_samples), X))

        self.loss_obj = LOSS_FUNCTIONS[self.loss]()
        if isinstance(
            self.loss_obj, (CensoredSquaredLoss, IPCWLeastSquaresError)
        ):
            time = np.log(time)

        if not self._is_fitted():
            self._init_state()

            y_pred = np.zeros(n_samples, dtype=np.float64)

            begin_at_stage = 0

            self._rng = check_random_state(self.random_state)
        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError(
                    "n_estimators=%d must be larger or equal to "
                    "estimators_.shape[0]=%d when "
                    "warm_start==True"
                    % (self.n_estimators, self.estimators_.shape[0])
                )
            begin_at_stage = self.estimators_.shape[0]
            y_pred = self._raw_predict(Xi)
            self._resize_state()

            # apply dropout to last stage of previous fit
            if hasattr(self, "_scale") and self.dropout_rate > 0:
                # pylint: disable-next=access-member-before-definition
                self._update_with_dropout(
                    self.n_estimators_ - 1, Xi, y_pred, self._scale, self._rng
                )

        self.n_estimators_ = self._fit(
            Xi, event, time, y_pred, sample_weight, self._rng, begin_at_stage
        )

        self.set_baseline_model(X, event, time)
        try:
            self.X_ = self.regr.X_
        except AttributeError:
            pass
        return self

    def set_baseline_model(self, X, event, time):
        if isinstance(self.loss_obj, CoxPH):
            risk_scores = self._predict(X)
            self.baseline_model = BreslowEstimator().fit(
                risk_scores, event, time
            )
        else:
            self.baseline_model = None

    def _raw_predict(self, X, **kwargs):
        if ("return_pi" in kwargs) or ("return_std" in kwargs):
            pred = {
                "mean": np.zeros(
                    X.shape[0], dtype=float
                ),  # 4 is mean, std, lower, upper
                "lower": np.zeros(X.shape[0], dtype=float),
                "upper": np.zeros(X.shape[0], dtype=float),
            }
            for estimator in self.estimators_:
                preds = estimator.predict(X, **kwargs)
                if "return_std" in kwargs:
                    pred["mean"] += self.learning_rate * preds.mean
                    pred["lower"] += self.learning_rate * preds.upper
                    pred["upper"] += self.learning_rate * preds.lower
            DescribeResult = namedtuple(
                "DescribeResult", ["mean", "lower", "upper"]
            )
            return DescribeResult(pred["mean"], pred["lower"], pred["upper"])
        pred = np.zeros(X.shape[0], dtype=float)
        for estimator in self.estimators_:
            pred += self.learning_rate * estimator.predict(X, **kwargs)
        return pred

    def _predict(self, X, **kwargs):
        # account for intercept
        Xi = np.column_stack((np.ones(X.shape[0]), X))
        pred = self._raw_predict(Xi, **kwargs)
        if len(np.asarray(pred).shape) == 1:
            return self.loss_obj._scale_raw_prediction(pred)
        if ("return_std" in kwargs) or ("return_pi" in kwargs):
            DescribeResult = namedtuple(
                "DescribeResult", ["mean", "lower", "upper"]
            )
            res = [
                self.loss_obj._scale_raw_prediction(p)
                for p in (pred[0], pred[1], pred[2])
            ]
            return DescribeResult(res[0], res[1], res[2])

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
        check_is_fitted(self, "estimators_")
        X = self._validate_data(X, reset=False)
        return self._predict(X, **kwargs)

    def get_baseline_model(self):
        if self.baseline_model is None:
            raise ValueError(
                "`fit` must be called with the loss option set to 'coxph'."
            )
        return self.baseline_model

    def predict_cumulative_hazard_function(
        self, X, return_array=False, **kwargs
    ):
        """Predict cumulative hazard function.

        Only available if :meth:`fit` has been called with `loss = "coxph"`.

        The cumulative hazard function for an individual
        with feature vector :math:`x` is defined as

        .. math::

            H(t \\mid x) = \\exp(f(x)) H_0(t) ,

        where :math:`f(\\cdot)` is the additive ensemble of base learners,
        and :math:`H_0(t)` is the baseline hazard function,
        estimated by Breslow's estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        return_array : boolean, default: False
            If set, return an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of
            :class:`survivalist.functions.StepFunction`.

        Returns
        -------
        cum_hazard : ndarray
            If `return_array` is set, an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of length `n_samples`
            of :class:`survivalist.functions.StepFunction` instances will be returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from survivalist.datasets import load_whas500
        >>> from survivalist.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

        Load the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = ComponentwiseGradientBoostingSurvivalAnalysis(loss="coxph").fit(X, y)

        Estimate the cumulative hazard function for the first 10 samples.

        >>> chf_funcs = estimator.predict_cumulative_hazard_function(X.iloc[:10])

        Plot the estimated cumulative hazard functions.

        >>> for fn in chf_funcs:
        ...     plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        if self.quantiles_ is not None:
            preds = self.predict(X, **kwargs)
            DescribeResult = namedtuple(
                "DescribeResult", ["mean", "lower", "upper"]
            )
            res = (
                self._predict_cumulative_hazard_function(
                    self.get_baseline_model(), preds.mean, return_array
                ),
                self._predict_cumulative_hazard_function(
                    self.get_baseline_model(), preds.lower, return_array
                ),
                self._predict_cumulative_hazard_function(
                    self.get_baseline_model(), preds.upper, return_array
                ),
            )
            return DescribeResult(res[0], res[1], res[2])
        return self._predict_cumulative_hazard_function(
            self.get_baseline_model(), self.predict(X, **kwargs), return_array
        )

    def predict_survival_function(self, X, return_array=False, **kwargs):
        """Predict survival function.

        Only available if :meth:`fit` has been called with `loss = "coxph"`.

        The survival function for an individual
        with feature vector :math:`x` is defined as

        .. math::

            S(t \\mid x) = S_0(t)^{\\exp(f(x)} ,

        where :math:`f(\\cdot)` is the additive ensemble of base learners,
        and :math:`S_0(t)` is the baseline survival function,
        estimated by Breslow's estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix.

        return_array : boolean, default: False
            If set, return an array with the probability
            of survival for each `self.unique_times_`,
            otherwise an array of :class:`survivalist.functions.StepFunction`.

        Returns
        -------
        survival : ndarray
            If `return_array` is set, an array with the probability of
            survival for each `self.unique_times_`, otherwise an array of
            length `n_samples` of :class:`survivalist.functions.StepFunction`
            instances will be returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from survivalist.datasets import load_whas500
        >>> from survivalist.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

        Load the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = ComponentwiseGradientBoostingSurvivalAnalysis(loss="coxph").fit(X, y)

        Estimate the survival function for the first 10 samples.

        >>> surv_funcs = estimator.predict_survival_function(X.iloc[:10])

        Plot the estimated survival functions.

        >>> for fn in surv_funcs:
        ...     plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        if ("return_pi" in kwargs) or ("return_std" in kwargs):
            preds = self.predict(X, **kwargs)
            DescribeResult = namedtuple(
                "DescribeResult", ["mean", "lower", "upper"]
            )
            res = (
                self._predict_survival_function(
                    self.get_baseline_model(),
                    preds.mean,
                    return_array,
                    **kwargs,
                ),
                self._predict_survival_function(
                    self.get_baseline_model(),
                    preds.lower,
                    return_array,
                    **kwargs,
                ),
                self._predict_survival_function(
                    self.get_baseline_model(),
                    preds.upper,
                    return_array,
                    **kwargs,
                ),
            )
            return DescribeResult(res[0], res[1], res[2])
        return self._predict_survival_function(
            self.get_baseline_model(),
            self.predict(X, **kwargs),
            return_array,
            **kwargs,
        )

    @property
    def unique_times_(self):
        return self.get_baseline_model().unique_times_

    @property
    def feature_importances_(self):
        imp = np.empty(self.n_features_in_ + 1, dtype=object)
        for i in range(imp.shape[0]):
            imp[i] = []

        for k, estimator in enumerate(self.estimators_):
            imp[estimator.component].append(k + 1)

        def _importance(x):
            if len(x) > 0:
                return np.min(x)
            return np.nan

        ret = np.array([_importance(x) for x in imp])
        return ret

    def _make_estimator(self, append=True, random_state=None):
        # we don't need _make_estimator
        raise NotImplementedError()
