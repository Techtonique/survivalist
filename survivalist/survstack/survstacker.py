import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from .transformer import SurvivalStacker
from ..util import check_array_survival
from ..base import SurvivalAnalysisMixin
from ..linear_model.coxph import BreslowEstimator
from ..ensemble.survival_loss import (
    LOSS_FUNCTIONS,
    CoxPH,
)
from ..functions import StepFunction


class SurvStacker(SurvivalAnalysisMixin):
    """
    A class to create a Survival Stacker for any classifier.
    """

    def __init__(
        self, clf=RandomForestClassifier(), loss="squared", random_state=42, **kwargs
    ):
        """
        Parameters
        ----------
        clf : classifier, default: RandomForestClassifier()
            The classifier to be used for stacking.

        loss : {'coxph', 'squared', 'ipcwls'}, optional, default: 'squared'
            Loss function to be optimized.

        random_state : int seed, RandomState instance, or None, default: 42
            The seed of the pseudo random number generator.

        kwargs : additional parameters to be passed to CalibratedClassifierCV
        """
        self.random_state = random_state
        self.clf = clf
        try:
            self.clf.set_params(random_state=self.random_state)
        except Exception:
            pass
        self.clf = CalibratedClassifierCV(clf, cv=3, **kwargs)
        self.ss = SurvivalStacker()
        self._baseline_model = None
        self.loss = loss
        self._loss = LOSS_FUNCTIONS[self.loss]()
        
        if self.loss not in ["coxph", "squared", "ipcwls"]:
            raise ValueError(
                f"Invalid loss value: {self.loss}. Choose from 'coxph', 'squared', or 'ipcwls'."
            )
        
        self.times_ = None
        self.unique_times_ = None

    def _get_baseline_model(self):
        """Get the baseline model for the survival stacker."""
        return self._baseline_model

    def _set_baseline_model(self, X, event, time):
        if isinstance(self._loss, CoxPH):
            risk_scores = self.predict(X)
            self._baseline_model = BreslowEstimator().fit(risk_scores, event, time)
        else:
            self._baseline_model = None

    def fit(self, X, y, **kwargs):
        """
        Fit the Survival Stacker to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        y : array-like, shape (n_samples,)
            The target values (survival times).

        kwargs : additional parameters to be passed to the fitting function

        Returns
        -------
        self : object
            Returns self.
        """
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        # Get survival stacker predictions
        X_oo, y_oo = self.ss.fit_transform(X, y)
        self.times_ = self.ss.times
        self.unique_times_ = np.sort(np.unique(self.ss.times))

        # Fit classifier
        self.clf.fit(X_oo, y_oo, **kwargs)
        
        # Set baseline model
        event, time = check_array_survival(X, y)
        self._set_baseline_model(X, event, time)
        
        return self

    def _predict_survival_function(self, X):
        """
        Predict survival function.
        """
        X_risk, _ = self.ss.transform(X)
        oo_test_estimates = self.clf.predict_proba(X_risk)[:, 1]
        return self.ss.predict_survival_function(oo_test_estimates)

    def predict(self, X, threshold=0.5):
        """
        Predict survival times using a threshold.
        """
        surv = self._predict_survival_function(X)
        
        crossings = surv <= threshold
        cross_indices = np.argmax(crossings, axis=1)
        valid_crossings = crossings[np.arange(len(crossings)), cross_indices]
        
        predicted_times = np.where(
            valid_crossings,
            self.unique_times_[cross_indices],
            self.unique_times_[-1],
        )
        return predicted_times

    def predict_cumulative_hazard_function(self, X, return_array=False):
        """
        Predict cumulative hazard function.
        """
        return self._predict_cumulative_hazard_function(
            self._get_baseline_model(), self.predict(X), return_array
        )

    def predict_survival_function(self, X, return_array=False):
        """
        Predict survival function.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        return_array : bool, default=False
            Whether to return the survival function as an array.
            
        Returns
        -------
        array-like or list of StepFunction
            Predicted survival function for each sample.
        """
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        surv = self._predict_survival_function(X)

        if return_array:
            return surv
        
        funcs = []
        surv = np.asarray(surv)
        if surv.ndim == 1:
            surv = surv.reshape(1, -1)

        for i in range(surv.shape[0]):
            if len(self.unique_times_) != len(surv[i]):
                x_old = np.linspace(0, 1, len(surv[i]))
                x_new = np.linspace(0, 1, len(self.unique_times_))
                surv_interp = np.interp(x_new, x_old, surv[i])
            else:
                surv_interp = surv[i]
            func = StepFunction(x=self.unique_times_, y=surv_interp)
            funcs.append(func)
        
        return np.array(funcs)