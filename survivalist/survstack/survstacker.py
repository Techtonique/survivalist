import numpy as np 

from ..base import SurvivalAnalysisMixin
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from .transformer import SurvivalStacker
from ..util import check_array_survival
from ..base import SurvivalAnalysisMixin
from ..linear_model.coxph import BreslowEstimator
from ..util import check_array_survival
from ..ensemble.survival_loss import (
    LOSS_FUNCTIONS,
    CensoredSquaredLoss,
    CoxPH,
    IPCWLeastSquaresError,
)
from ..functions import StepFunction


class SurvStacker(SurvivalAnalysisMixin):
    """
    A class to create a Survival Stacker for any classifier.
    """
    def __init__(
        self,
        clf=LogisticRegression(),
        loss="squared",
        random_state=42,
        **kwargs      
    ):
        """
        Parameters
        ----------
        clf : classifier, default: LogisticRegression()
            The classifier to be used for stacking.
        
        loss : {'coxph', 'squared', 'ipcwls'}, optional, default: 'coxph'
        loss function to be optimized. 'coxph' refers to partial likelihood loss
        of Cox's proportional hazards model. The loss 'squared' minimizes a
        squared regression loss that ignores predictions beyond the time of censoring,
        and 'ipcwls' refers to inverse-probability of censoring weighted least squares error.
        
        random_state : int seed, RandomState instance, or None, default: None
            The seed of the pseudo random number generator to use when
            shuffling the data.

        kwargs : additional parameters to be passed to CalibratedClassifierCV    
        """        
        self.random_state = random_state
        self.clf = clf
        try: 
            self.clf.set_params(random_state=self.random_state)
        except Exception as e:
            pass 
        self.clf = CalibratedClassifierCV(clf, 
                                          **kwargs)        
        self.ss = SurvivalStacker()
        self.times_ = None
        self.unique_times_ = None
        self._baseline_model = None
        self.loss = loss 
        self._loss = LOSS_FUNCTIONS[self.loss]()
    
    def _get_baseline_model(self):
        """
        Get the baseline model for the survival stacker.

        Returns
        -------
        self.ss : SurvivalStacker
            The fitted survival stacker.
        """
        return self._baseline_model

    def _set_baseline_model(self, X, event, time):
        if isinstance(self._loss, CoxPH):
            risk_scores = self.predict(X)
            self._baseline_model = BreslowEstimator().fit(
                risk_scores, event, time
            )
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
        
        kwargs : additional parameters to be passed to the fitting function of the classifier (e.g., `sample_weight`)

        Returns
        -------
        self : object
            Returns self.
        """        
        # Convert X to numpy array if needed
        if hasattr(X, 'to_numpy'):
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
        
    def _predict_survival_function_temp(self, X):
        """
        Predict the survival function for the given input samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        array-like, shape (n_samples, n_timepoints)
            The predicted survival function for each sample at each timepoint.
        """
        X_risk, _ = self.ss.transform(X)
        oo_test_estimates = self.clf.predict_proba(X_risk)[:, 1]
        return self.ss.predict_survival_function(oo_test_estimates)

    def predict(self, X, threshold=0.5):
        surv = self._predict_survival_function_temp(X)  # shape: (n_samples, n_times)
        crossings = surv <= threshold  # Boolean array of threshold crossings
        
        # For each sample, get the index of the first crossing (or -1 if none)
        cross_indices = np.argmax(crossings, axis=1)
        
        # Handle cases where survival never crosses the threshold:
        # argmax returns 0 if no True found, so we need to check if the crossing is valid
        valid_crossings = crossings[np.arange(len(crossings)), cross_indices]
        
        # Map to actual times
        predicted_times = np.where(
            valid_crossings,
            self.unique_times_[cross_indices],
            self.unique_times_[-1]  # use max time if no crossing
        )
        
        return predicted_times
    

    def predict_cumulative_hazard_function(self, X, return_array=False):
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
        # Convert X to numpy array if needed
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
            
        # Get predictions using temporary method
        surv = self._predict_survival_function_temp(X)
        
        if return_array:
            return surv
            
        # Convert to StepFunction instances
        funcs = []
        for i in range(surv.shape[0]):
            func = StepFunction(x=self.unique_times_, y=surv[i])
            funcs.append(func)
        return np.array(funcs)