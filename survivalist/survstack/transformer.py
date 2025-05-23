import numpy as np

from . import functional as ssf


class SurvivalStacker:
    """Casts a survival analysis problem as a classification problem as
    proposed in Craig E., et al. 2021 (arXiv:2107.13480)
    """

    def __init__(self, times: np.ndarray | None = None) -> None:
        """Generate a SurvivalStacker instance

        :param times: array of time points on which to create risk sets
        """
        if times is not None:
            self.times = times
        else:
            self.times = np.empty(0)

    def fit(self, X: np.ndarray, y: np.ndarray, time_step: float | None = None):
        """Generate the risk time points

        :param X: survival input samples
        :param y: structured array with two fields. The binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        :param time_step: a base multiple on which to bin times. If none, the
            times for all observed events are used.
        :return: self
        """
        event_field, time_field = y.dtype.names
        event_times = np.unique(y[time_field][y[event_field]])
        if time_step is None:
            self.times = event_times
        else:
            self.times = ssf.digitize_times(event_times, time_step)
        return self

    def transform(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Convert the input survival dataset to a stacked survival dataset

        :param X: survival input samples
        :param y: structured array with two fields. The binary event indicator
            as first field, and time of event or time of censoring as
            second field. If None, the returned dataset is constructed for
            evaluation.
        :return: a tuple containing the predictor matrix and response vector
        """
        if y is None:
            X_stacked = ssf.stack_eval(X, self.times)
            y_stacked = None
        else:
            X_stacked, y_stacked = ssf.stack_timepoints(X, y, self.times)
        return X_stacked, y_stacked

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, time_step: float | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Fit to data, then transform it.

        :param X: survival input samples
        :param y: structured array with two fields. The binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        :param time_step: a base multiple on which to bin times. If none, the
            times for all observed events are used.
        :return: a tuple containing the predictor matrix and response vector
        """
        self.fit(X, y, time_step)
        return self.transform(X, y)

    def cumulative_hazard_function(self, X: np.ndarray) -> np.ndarray:
        """Calculate the cumulative hazard function from the stacked survival
        estimates.

        :param X: estimates as returned from a model trained on
        an evaluation set
        :return: a cumulative risk matrix for the fitted time-points
        """
        return ssf.cumulative_hazard_function(X, self.times)

    def predict_survival_function(self, X: np.ndarray) -> np.ndarray:
        """Calculate the survival function from the stacked survival estimates.

        :param X: estimates as returned from a model trained on
        an evaluation set
        :return: the survival function for the fitted time-points
        """
        return ssf.survival_function(X, self.times)

    def risk_score(self, estimates: np.ndarray) -> np.ndarray:
        """Calculate risk score from stacked survival estimates.

        :param estimates: estimates as returned from a model trained on
        an evaluation set
        :return: the risk score
        """
        return ssf.risk_score(estimates, self.times)
