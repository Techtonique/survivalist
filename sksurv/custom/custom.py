from math import ceil
from numbers import Integral, Real

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.tree._utils import _any_isnan_axis0
from sklearn.utils.validation import (
    _assert_all_finite_element_wise,
    assert_all_finite,
    check_is_fitted,
    check_random_state,
)

from ..base import SurvivalAnalysisMixin
from ..functions import StepFunction
from ..util import check_array_survival
from ._criterion import LogrankCriterion, get_unique_times

__all__ = ["ExtraSurvivalCustom", "SurvivalCustom"]

DTYPE = _tree.DTYPE


def _array_to_step_function(x, array):
    n_samples = array.shape[0]
    funcs = np.empty(n_samples, dtype=np.object_)
    for i in range(n_samples):
        funcs[i] = StepFunction(x=x, y=array[i])
    return funcs


class SurvivalCustom(BaseEstimator, SurvivalAnalysisMixin):
    """A survival custom model.

    The quality of a split is measured by the
    log-rank splitting rule.

    If ``splitter='best'``, fit and predict methods support
    missing values. See :ref:`custom_missing_value_support` for details.

    See [1]_, [2]_ and [3]_ for further description.

    Parameters
    ----------

    regr: object
        Base learner 

    random_state : int, RandomState instance or None, optional, default: None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behavior
        during fitting, ``random_state`` has to be fixed to an integer.

    low_memory : boolean, default: False
        If set, ``predict`` computations use reduced memory but ``predict_cumulative_hazard_function``
        and ``predict_survival_function`` are not implemented.

    Attributes
    ----------
    unique_times_ : array of shape = (n_unique_times,)
        Unique time points.

    max_features_ : int,
        The inferred value of max_features.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    custom_ : Custom object
        The underlying Custom object. Please refer to
        ``help(sklearn.tree._tree.Custom)`` for attributes of Custom object.

    See also
    --------
    sksurv.ensemble.RandomSurvivalForest
        An ensemble of SurvivalTrees.

    References
    ----------
    .. [1] Leblanc, M., & Crowley, J. (1993). Survival Trees by Goodness of Split.
           Journal of the American Statistical Association, 88(422), 457–467.

    .. [2] Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008).
           Random survival forests. The Annals of Applied Statistics, 2(3), 841–860.

    .. [3] Ishwaran, H., Kogalur, U. B. (2007). Random survival forests for R.
           R News, 7(2), 25–31. https://cran.r-project.org/doc/Rnews/Rnews_2007-2.pdf.
    """

    _parameter_constraints = {
        "splitter": [StrOptions({"best", "random"})],
        "max_depth": [Interval(Integral, 1, None, closed="left"), None],
        "min_samples_split": [
            Interval(Integral, 2, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="neither"),
        ],
        "min_samples_leaf": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 0.5, closed="right"),
        ],
        "min_weight_fraction_leaf": [Interval(Real, 0.0, 0.5, closed="both")],
        "max_features": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="right"),
            StrOptions({"sqrt", "log2"}),
            None,
        ],
        "random_state": ["random_state"],
        "max_leaf_nodes": [Interval(Integral, 2, None, closed="left"), None],
        "low_memory": ["boolean"],
    }

    criterion = "logrank"

    def __init__(
        self,
        *,
        regr,
        random_state=None,
        low_memory=False,
    ):
        self.custom_ = regr 
        self.random_state = random_state
        self.low_memory = low_memory

    def _more_tags(self):
        allow_nan = self.splitter == "best"
        return {"allow_nan": allow_nan}

    def _support_missing_values(self, X):
        return not issparse(X) and self._get_tags()["allow_nan"]

    def _compute_missing_values_in_feature_mask(self, X, estimator_name=None):
        """Return boolean mask denoting if there are missing values for each feature.

        This method also ensures that X is finite.

        Parameter
        ---------
        X : array-like of shape (n_samples, n_features), dtype=DOUBLE
            Input data.

        estimator_name : str or None, default=None
            Name to use when raising an error. Defaults to the class name.

        Returns
        -------
        missing_values_in_feature_mask : ndarray of shape (n_features,), or None
            Missing value mask. If missing values are not supported or there
            are no missing values, return None.
        """
        estimator_name = estimator_name or self.__class__.__name__
        common_kwargs = dict(estimator_name=estimator_name, input_name="X")

        if not self._support_missing_values(X):
            assert_all_finite(X, **common_kwargs)
            return None

        with np.errstate(over="ignore"):
            overall_sum = np.sum(X)

        if not np.isfinite(overall_sum):
            # Raise a ValueError in case of the presence of an infinite element.
            _assert_all_finite_element_wise(X, xp=np, allow_nan=True, **common_kwargs)

        # If the sum is not nan, then there are no missing values
        if not np.isnan(overall_sum):
            return None

        missing_values_in_feature_mask = _any_isnan_axis0(X)
        return missing_values_in_feature_mask

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a survival tree from the training set (X, y).

        If ``splitter='best'``, `X` is allowed to contain missing
        values. In addition to evaluating each potential threshold on
        the non-missing data, the splitter will evaluate the split
        with all the missing values going to the left node or the
        right node. See :ref:`custom_missing_value_support` for details.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self
        """
        self._fit(X, y, sample_weight, check_input)
        return self

    def _fit(self, X, y, sample_weight=None, check_input=True, missing_values_in_feature_mask=None):
        random_state = check_random_state(self.random_state)

        if check_input:
            X = self._validate_data(X, dtype=DTYPE, ensure_min_samples=2, accept_sparse="csc", force_all_finite=False)
            event, time = check_array_survival(X, y)
            time = time.astype(np.float64)
            self.unique_times_, self.is_event_time_ = get_unique_times(time, event)
            if issparse(X):
                X.sort_indices()

            y_numeric = np.empty((X.shape[0], 2), dtype=np.float64)
            y_numeric[:, 0] = time
            y_numeric[:, 1] = event.astype(np.float64)
        else:
            y_numeric, self.unique_times_, self.is_event_time_ = y

        n_samples, self.n_features_in_ = X.shape
        params = self._check_params(n_samples)

        if self.low_memory:
            self.n_outputs_ = 1
            # one "class" only, for the sum over the CHF
            self.n_classes_ = np.ones(self.n_outputs_, dtype=np.intp)
        else:
            self.n_outputs_ = self.unique_times_.shape[0]
            # one "class" for CHF, one for survival function
            self.n_classes_ = np.ones(self.n_outputs_, dtype=np.intp) * 2

        self.custom_.fit(X, y_numeric, sample_weight)

        return self

    def _check_max_features(self):
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))

        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, (Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        if not 0 < max_features <= self.n_features_in_:
            raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

    def _check_low_memory(self, function):
        """Check if `function` is supported in low memory mode and throw if it is not."""
        if self.low_memory:
            raise NotImplementedError(
                f"{function} is not implemented in low memory mode."
                + " run fit with low_memory=False to disable low memory mode."
            )

    def _validate_X_predict(self, X, check_input, accept_sparse="csr"):
        """Validate X whenever one tries to predict"""
        if check_input:
            if self._support_missing_values(X):
                force_all_finite = "allow-nan"
            else:
                force_all_finite = True
            X = self._validate_data(
                X,
                dtype=DTYPE,
                accept_sparse=accept_sparse,
                reset=False,
                force_all_finite=force_all_finite,
            )
        else:
            # The number of features is checked regardless of `check_input`
            self._check_n_features(X, reset=False)

        return X

    def predict(self, X, check_input=True):
        """Predict risk score.

        The risk score is the total number of events, which can
        be estimated by the sum of the estimated cumulative
        hazard function :math:`\\hat{H}_h` in terminal node :math:`h`.

        .. math::

            \\sum_{j=1}^{n(h)} \\hat{H}_h(T_{j} \\mid x) ,

        where :math:`n(h)` denotes the number of distinct event times
        of samples belonging to the same terminal node as :math:`x`.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            Data matrix.
            If ``splitter='best'``, `X` is allowed to contain missing
            values and decisions are made as described in
            :ref:`custom_missing_value_support`.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        risk_scores : ndarray, shape = (n_samples,)
            Predicted risk scores.
        """

        if self.low_memory:
            check_is_fitted(self, "custom_")
            X = self._validate_X_predict(X, check_input, accept_sparse="csr")
            pred = self.custom_.predict(X)
            return pred[..., 0]

        chf = self.predict_cumulative_hazard_function(X, check_input, return_array=True)
        return chf[:, self.is_event_time_].sum(1)

    def predict_cumulative_hazard_function(self, X, check_input=True, return_array=False):
        """Predict cumulative hazard function.

        The cumulative hazard function (CHF) for an individual
        with feature vector :math:`x` is computed from
        all samples of the training data that are in the
        same terminal node as :math:`x`.
        It is estimated by the Nelson–Aalen estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            Data matrix.
            If ``splitter='best'``, `X` is allowed to contain missing
            values and decisions are made as described in
            :ref:`custom_missing_value_support`.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        return_array : boolean, default: False
            If set, return an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of
            :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        cum_hazard : ndarray
            If `return_array` is set, an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of length `n_samples`
            of :class:`sksurv.functions.StepFunction` instances will be returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_whas500
        >>> from sksurv.tree import SurvivalCustom

        Load and prepare the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = SurvivalCustom().fit(X, y)

        Estimate the cumulative hazard function for the first 5 samples.

        >>> chf_funcs = estimator.predict_cumulative_hazard_function(X.iloc[:5])

        Plot the estimated cumulative hazard functions.

        >>> for fn in chf_funcs:
        ...    plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        self._check_low_memory("predict_cumulative_hazard_function")
        check_is_fitted(self, "custom_")
        X = self._validate_X_predict(X, check_input, accept_sparse="csr")

        pred = self.custom_.predict(X)
        arr = pred[..., 0]
        if return_array:
            return arr
        return _array_to_step_function(self.unique_times_, arr)

    def predict_survival_function(self, X, check_input=True, return_array=False):
        """Predict survival function.

        The survival function for an individual
        with feature vector :math:`x` is computed from
        all samples of the training data that are in the
        same terminal node as :math:`x`.
        It is estimated by the Kaplan-Meier estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            Data matrix.
            If ``splitter='best'``, `X` is allowed to contain missing
            values and decisions are made as described in
            :ref:`custom_missing_value_support`.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        return_array : boolean, default: False
            If set, return an array with the probability
            of survival for each `self.unique_times_`,
            otherwise an array of :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        survival : ndarray
            If `return_array` is set, an array with the probability of
            survival for each `self.unique_times_`, otherwise an array of
            length `n_samples` of :class:`sksurv.functions.StepFunction`
            instances will be returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_whas500
        >>> from sksurv.tree import SurvivalCustom

        Load and prepare the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = SurvivalCustom().fit(X, y)

        Estimate the survival function for the first 5 samples.

        >>> surv_funcs = estimator.predict_survival_function(X.iloc[:5])

        Plot the estimated survival functions.

        >>> for fn in surv_funcs:
        ...    plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        self._check_low_memory("predict_survival_function")
        check_is_fitted(self, "custom_")
        X = self._validate_X_predict(X, check_input, accept_sparse="csr")

        pred = self.custom_.predict(X)
        arr = pred[..., 1]
        if return_array:
            return arr
        return _array_to_step_function(self.unique_times_, arr)

