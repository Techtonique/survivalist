import numpy as np 

from collections import namedtuple
from ..base import SurvivalAnalysisMixin
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from .transformer import SurvivalStacker
from ..util import check_array_survival
from ..utils import simulate_replications
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
        clf=RandomForestClassifier(),
        loss="squared",
        type_sim="none",
        replications=250,
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

        type_sim (str): Method for simulation:
                      - 'none': No simulation. 
                      - 'kde': Kernel Density Estimation.
                      - 'bootstrap': Bootstrap resampling.                      
                      - 'normal': Parametric distribution fitting.
                      - 'ecdf': Empirical CDF-based sampling.
                      - 'permutation': Permutation resampling.
                      - 'smooth_bootstrap': Smoothed bootstrap with added noise.
        
        replications : int, default: 250
            The number of replications for the simulation.
        
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
        self.clf = CalibratedClassifierCV(clf, **kwargs)        
        self.ss = SurvivalStacker()        
        self._baseline_model = None
        self.loss = loss 
        self._loss = LOSS_FUNCTIONS[self.loss]()
        self.type_sim = type_sim
        if self.type_sim not in ["none", "kde", "bootstrap", "normal", "ecdf", "permutation", "smooth_bootstrap"]:
            raise ValueError(
                f"Invalid type_sim value: {self.type_sim}. "
                "Choose from 'none', 'kde', 'bootstrap', 'normal', 'ecdf', 'permutation', or 'smooth_bootstrap'."
            )
        if self.loss not in ["coxph", "squared", "ipcwls"]:
            raise ValueError(
                f"Invalid loss value: {self.loss}. "
                "Choose from 'coxph', 'squared', or 'ipcwls'."
            )
        self.replications = replications 
        self.times_ = None
        self.unique_times_ = None
        self.calibrated_residuals_ = None 

    
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
        
        #print("X shape:", X.shape)
        
        # Get survival stacker predictions
        X_oo, y_oo = self.ss.fit_transform(X, y)
        self.times_ = self.ss.times
        self.unique_times_ = np.sort(np.unique(self.ss.times))
        #print("X_oo shape:", X_oo.shape)
        #print("y_oo shape:", y_oo.shape)
        #print("self.times_ shape:", self.times_.shape)
        #print("self.unique_times_ shape:", self.unique_times_.shape)

        if self.type_sim != "none":
            half_n = X_oo.shape[0] // 2
            X_train_oo, X_calib_oo, y_train_oo, y_calib_oo = train_test_split(
                X_oo, y_oo, test_size=half_n, random_state=self.random_state, 
                stratify = y_oo)        
            # Fit classifier
            self.clf.fit(X_train_oo, y_train_oo, **kwargs)
            #print("y_train_oo:", y_train_oo)
            # Calibrate classifier
            calib_probs = self.clf.predict_proba(X_calib_oo)[:,1]
            #print("calib_probs shape:", calib_probs.shape)
            encoder = OneHotEncoder(sparse_output=False)
            #print("calib response shape:", y_calib_oo.shape)
            test_probs = encoder.fit_transform(y_calib_oo.reshape(-1, 1))
            #print("test_probs shape:", test_probs.shape)
            self.calibrated_residuals_ = test_probs[:,1] - calib_probs
            self.clf.fit(X_calib_oo, y_calib_oo, **kwargs)
            # Set baseline model
            event, time = check_array_survival(X, y)
            self._set_baseline_model(X, event, time)
            return self
        
        # Set baseline model
        self.clf.fit(X_oo, y_oo, **kwargs)
        event, time = check_array_survival(X, y)
        self._set_baseline_model(X, event, time)
        return self        
        

    def _predict_survival_function_temp(self, X):
        """
        Predict survival function with normalized probabilities.
        """
        #print("X shape:", X.shape)
        X_risk, _ = self.ss.transform(X)
        #print("X_risk shape:", X_risk.shape)
        oo_test_estimates = self.clf.predict_proba(X_risk)[:, 1]
        #print("oo_test_estimates shape:", oo_test_estimates.shape)

        if self.type_sim == "none":
            # If no simulation, return the test estimates
            return self.ss.predict_survival_function(oo_test_estimates)
        
        # Apply the calibrated residuals  
        #print("Calibrated residuals shape:", self.calibrated_residuals_.shape)
        #print("oo_test_estimates shape:", oo_test_estimates.shape)
        # Simulate the residuals      
        simulations_oo_test_estimates = simulate_replications(
            data=self.calibrated_residuals_, 
            num_replications=self.replications, 
            n_obs=oo_test_estimates.shape[0],
            method=self.type_sim
        )
        #print("simulations_oo_test_estimates shape:", simulations_oo_test_estimates.shape)
        # Add the calibrated residuals to the test estimates
        oo_test_estimates = np.tile(oo_test_estimates, (self.replications, 1)).T + simulations_oo_test_estimates
        # clip values to be between 0 and 1
        oo_test_estimates = np.clip(oo_test_estimates, 0, 1)
        
        # Normalize probabilities to sum to 1 for each replication
        for i in range(self.replications):
            probs = oo_test_estimates[:, i]
            probs = np.maximum(probs, 0)  # Ensure probabilities are positive
            probs = probs / np.sum(probs)  # Normalize to sum to 1
            oo_test_estimates[:, i] = probs
        
        return [self.ss.predict_survival_function(oo_test_estimates[:, i]) 
                for i in range(self.replications)]


    def predict(self, X, threshold=0.5):

        surv = self._predict_survival_function_temp(X)  

        if self.type_sim == "none":
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
        
        crossings = [s <= threshold for s in surv]
        cross_indices = [np.argmax(cross) for cross in crossings]
        # Handle cases where survival never crosses the threshold:
        # argmax returns 0 if no True found, so we need to check if the crossing is valid
        valid_crossings = [cross[ci] for cross, ci in zip(crossings, cross_indices)]
        # Map to actual times
        predicted_times = [
            np.where(
                valid,
                self.unique_times_[ci],
                self.unique_times_[-1]  # use max time if no crossing
            ) for valid, ci in zip(valid_crossings, cross_indices)
        ]
        return np.array(predicted_times)
    

    def predict_cumulative_hazard_function(self, X, return_array=False):
        """
        Predict cumulative hazard function.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        return_array : bool, default=False
            Whether to return the cumulative hazard function as an array.
        Returns
        -------
        array-like or list of StepFunction
            Predicted cumulative hazard function for each sample.
        """
        if self.type_sims == "none":
            # Convert X to numpy array if needed
            return self._predict_cumulative_hazard_function(
                self._get_baseline_model(), self.predict(X), return_array)
        return [self._predict_cumulative_hazard_function(
            self._get_baseline_model(), s, return_array
        ) for s in self._predict_survival_function_temp(X)]


    def predict_survival_function(self, X, return_array=False, level=95):
        """
        Predict survival function.
        """
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
            
        surv = self._predict_survival_function_temp(X)

        #print("surv: ", surv)

        if self.type_sim == "none":
            if return_array:
                return surv
                
            funcs = []
            surv = np.asarray(surv)
            if surv.ndim == 1:
                surv = surv.reshape(1, -1)
            
            for i in range(surv.shape[0]):
                if len(self.unique_times_) != len(surv[i]):
                    # Créer une grille temporelle appropriée
                    x_old = np.linspace(0, 1, len(surv[i]))
                    x_new = np.linspace(0, 1, len(self.unique_times_))
                    surv_interp = np.interp(x_new, x_old, surv[i])
                else:
                    surv_interp = surv[i]
                func = StepFunction(x=self.unique_times_, y=surv_interp)
                funcs.append(func)
            return np.array(funcs)
        else:
            SurvivalCurves = namedtuple('SurvivalCurves', 
                                        ['mean', 'lower', 'upper'])
            # `surv` contains `replications` number of survival functions, 
            # for each sample in the test set
            n_obs = X.shape[0]        
            results = {}
            for j in range(n_obs): 
                key = "obs" + str(j)
                results[key] = []
                for i in range(self.replications):
                    results[key].append(surv[i][j])
            # Calculate mean, lower and upper bounds
            mean_surv = []
            lower_surv = []
            upper_surv = []
            for key in results.keys():
                mean_surv.append(np.mean(results[key], axis=0))
                lower_surv.append(np.percentile(results[key], (100 - level) / 2, axis=0))
                upper_surv.append(np.percentile(results[key], 100 - (100 - level) / 2, axis=0))
            mean_surv = np.array(mean_surv)
            lower_surv = np.array(lower_surv)
            upper_surv = np.array(upper_surv)
            if return_array:
                return mean_surv, lower_surv, upper_surv
            # Create StepFunction objects for mean, lower and upper bounds
            mean_funcs = []
            lower_funcs = []
            upper_funcs = []
            for i in range(mean_surv.shape[0]):
                mean_func = StepFunction(x=self.unique_times_, y=mean_surv[i])
                lower_func = StepFunction(x=self.unique_times_, y=lower_surv[i])
                upper_func = StepFunction(x=self.unique_times_, y=upper_surv[i])
                mean_funcs.append(mean_func)
                lower_funcs.append(lower_func)
                upper_funcs.append(upper_func)
            return SurvivalCurves(mean=mean_funcs, lower=lower_funcs, upper=upper_funcs)

