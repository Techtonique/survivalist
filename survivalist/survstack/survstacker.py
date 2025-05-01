from sklearn.linear_model import LogisticRegression
from .transformer import SurvivalStacker

class SurvStacker:
    """
    A class to create a Survival Stacker for any classifier.
    """
    def __init__(
        self,
        clf=LogisticRegression(),
        random_state=None,      
    ):
        """
        Parameters
        ----------
        clf : classifier, default: LogisticRegression()
            The classifier to be used for stacking.
        
        random_state : int seed, RandomState instance, or None, default: None
            The seed of the pseudo random number generator to use when
            shuffling the data.
        """
        self.clf = clf
        self.random_state = random_state
        self.ss = SurvivalStacker()

    def fit(self, X, y):
        """
        Fit the Survival Stacker to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        
        y : array-like, shape (n_samples,)
            The target values (survival times).

        Returns
        -------
        self : object
            Returns self.
        """
        X_oo, y_oo = self.ss.fit_transform(X, y)
        self.clf.fit(X_oo, y_oo)
        return self
    
    def predict_cumulative_hazard(self, X):
        """
        Predict the cumulative hazard for the given input samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        array-like, shape (n_samples, n_timepoints)
            The predicted cumulative hazard for each sample at each timepoint.
        """
        X_risk, _ = self.ss.transform(X)
        oo_test_estimates = self.clf.predict_proba(X_risk)[:, 1]
        return self.ss.cumulative_hazard_function(oo_test_estimates)
    
    def predict_survival_function(self, X):
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