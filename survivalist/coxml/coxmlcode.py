import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.pipeline import Pipeline
import warnings


class CoxMLWrapper(BaseEstimator, RegressorMixin):
    """
    Universal Cox Proportional Hazards wrapper for any scikit-learn regressor.

    Implements Cox (1972) partial likelihood with automatic gradient selection:
    - Analytical gradients for linear models (Ridge, Lasso, ElasticNet, etc.)
    - Finite difference gradients for non-linear models
    - Functional gradient descent for tree-based models without extractable parameters

    Parameters
    ----------
    base_model : sklearn estimator
        Any scikit-learn regressor (can be a Pipeline).
    max_iter : int, default=100
        Maximum optimization iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    verbose : bool, default=False
        Print optimization progress.
    fd_epsilon : float, default=1e-5
        Step size for finite differences.
    learning_rate : float, default=0.01
        Learning rate for non-parametric optimization.
    random_state : int, optional
        Random seed for reproducibility.
    use_analytic_grad : bool, default=True
        Use analytical gradients for linear models if available.

    Attributes
    ----------
    loss_history_ : list
        Cox partial likelihood loss during optimization.
    baseline_hazard_ : tuple
        (unique_times, cumulative_baseline_hazard) from Breslow estimator.
    """

    def __init__(self, base_model, max_iter=100, tol=1e-4, verbose=False,
                 fd_epsilon=1e-5, learning_rate=0.01, random_state=None,
                 use_analytic_grad=True):
        self.base_model = base_model
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.fd_epsilon = fd_epsilon
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.use_analytic_grad = use_analytic_grad
        self.loss_history_ = []
        self.baseline_hazard_ = None

    def _final_estimator(self):
        """Extract final estimator from pipeline or return base model."""
        if isinstance(self.base_model, Pipeline):
            return self.base_model.steps[-1][1]
        return self.base_model

    def _has_extractable_params(self):
        """Check if model has extractable parameters (coef_/intercept_)."""
        est = self._final_estimator()
        return (hasattr(est, 'coef_') and hasattr(est, 'intercept_') and
                not hasattr(est, 'coefs_'))  # Exclude neural networks

    def _get_params_vector(self):
        """Extract and flatten model parameters."""
        est = self._final_estimator()
        params = []

        if hasattr(est, 'coef_'):
            params.append(est.coef_.ravel())
        if hasattr(est, 'intercept_'):
            params.append(np.atleast_1d(est.intercept_).ravel())
        if hasattr(est, 'coefs_'):  # Neural networks
            for w in est.coefs_:
                params.append(w.ravel())
        if hasattr(est, 'intercepts_'):
            for b in est.intercepts_:
                params.append(b.ravel())

        if not params:
            raise ValueError("Model has no extractable parameters")

        return np.concatenate(params)

    def _set_params_vector(self, params_flat):
        """Set model parameters from flattened vector."""
        est = self._final_estimator()
        idx = 0

        if hasattr(est, 'coef_'):
            n = est.coef_.size
            est.coef_ = params_flat[idx:idx + n].reshape(est.coef_.shape)
            idx += n

        if hasattr(est, 'intercept_'):
            if np.isscalar(est.intercept_):
                est.intercept_ = float(params_flat[idx])
                idx += 1
            else:
                n = est.intercept_.size
                est.intercept_ = params_flat[idx:idx + n].reshape(est.intercept_.shape)
                idx += n

        if hasattr(est, 'coefs_'):
            for i, w in enumerate(est.coefs_):
                n = w.size
                est.coefs_[i] = params_flat[idx:idx + n].reshape(w.shape)
                idx += n

        if hasattr(est, 'intercepts_'):
            for i, b in enumerate(est.intercepts_):
                n = b.size
                est.intercepts_[i] = params_flat[idx:idx + n].reshape(b.shape)
                idx += n

    def _cox_loss(self, risk_scores, times, events):
        """
        Compute negative Cox partial log-likelihood.

        Returns
        -------
        loss : float
            Negative log partial likelihood (to minimize).
        """
        # Sort by descending time
        order = np.argsort(-times)
        risk_scores = risk_scores[order]
        times = times[order]
        events = events[order]

        unique_times = np.unique(times[events == 1])
        if len(unique_times) == 0:
            return 0.0

        loglik = 0.0
        for t in unique_times:
            at_risk = times >= t
            failed = (times == t) & (events == 1)
            n_failed = np.sum(failed)

            if n_failed > 0:
                loglik += np.sum(risk_scores[failed])
                loglik -= n_failed * logsumexp(risk_scores[at_risk])

        return -loglik  # Negative for minimization

    def _analytic_gradient(self, params, X, times, events):
        """
        Analytical gradient for linear models: grad = X^T @ grad_risk.
        This computes the gradient of the NEGATIVE log-likelihood (for minimization).

        Returns
        -------
        grad : ndarray
            Gradient w.r.t. [coef_, intercept_].
        """
        # Set parameters and predict
        self._set_params_vector(params)
        risk = self.base_model.predict(X)

        # Sort by descending time
        order = np.argsort(-times)
        X_sorted = X[order]
        risk_sorted = risk[order]
        times_sorted = times[order]
        events_sorted = events[order]

        unique_times = np.unique(times_sorted[events_sorted == 1])
        if len(unique_times) == 0:
            return np.zeros_like(params)

        # Compute gradient w.r.t. risk scores
        # For NEGATIVE log-likelihood, gradient signs are flipped
        grad_risk = np.zeros_like(risk_sorted)
        exp_risk = np.exp(risk_sorted)

        for t in unique_times:
            at_risk = times_sorted >= t
            failed = (times_sorted == t) & (events_sorted == 1)
            n_failed = np.sum(failed)

            if n_failed > 0:
                sum_exp_at_risk = np.sum(exp_risk[at_risk]) + 1e-15

                # Gradient of NEGATIVE log-likelihood:
                # -d(-logL)/d(risk_i) = -(delta_i - n_failed * exp(risk_i) / sum_exp)
                grad_risk[failed] -= 1  # Negative because we minimize -logL
                grad_risk[at_risk] += n_failed * exp_risk[at_risk] / sum_exp_at_risk

        # Map gradient back to original order
        grad_risk_original = np.empty_like(grad_risk)
        grad_risk_original[order] = grad_risk

        # Chain rule: grad_params = X^T @ grad_risk
        grad_coef = X.T @ grad_risk_original
        grad_intercept = np.sum(grad_risk_original)

        return np.concatenate([grad_coef.ravel(), [grad_intercept]])

    def _finite_diff_gradient(self, params, X, times, events):
        """Finite difference gradient (fallback for non-linear models)."""
        grad = np.zeros_like(params)
        f0 = self._objective(params, X, times, events, record=False)

        for i in range(len(params)):
            p = params.copy()
            p[i] += self.fd_epsilon
            grad[i] = (self._objective(p, X, times, events, record=False) - f0) / self.fd_epsilon

        return grad

    def _objective(self, params, X, times, events, record=True):
        """Objective function for optimization."""
        self._set_params_vector(params)
        risk = self.base_model.predict(X)
        loss = self._cox_loss(risk, times, events)
        if record:
            self.loss_history_.append(loss)
        return loss

    def fit(self, X, times, events):
        """
        Fit the Cox model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        times : array-like, shape (n_samples,)
            Observed times (either event or censoring).
        events : array-like, shape (n_samples,)
            Event indicators (1=event, 0=censored).

        Returns
        -------
        self : CoxMLWrapper
            Fitted model.
        """
        # Validate inputs
        X, times = check_X_y(X, times, dtype=np.float64)
        events = check_array(events, ensure_2d=False, dtype=np.float64)

        self.n_features_in_ = X.shape[1]

        # Set random state
        if self.random_state is not None and hasattr(self.base_model, 'random_state'):
            self.base_model.random_state = self.random_state

        # Phase 1: Warm start with actual survival times (helps initialization)
        if self.verbose:
            print("Phase 1: Warm start with survival times...")
        self.base_model.fit(X, times)

        # Phase 2: Cox optimization
        try:
            params0 = self._get_params_vector()

            if self.verbose:
                print(f"Phase 2: Optimizing {len(params0)} parameters...")

            # Choose gradient method
            use_analytic = (self._has_extractable_params() and self.use_analytic_grad)

            if use_analytic:
                if self.verbose:
                    print("Using analytical gradients (fast)...")
                jac = lambda p: self._analytic_gradient(p, X, times, events)
            else:
                if self.verbose:
                    print("Using finite difference gradients...")
                jac = lambda p: self._finite_diff_gradient(p, X, times, events)

            res = minimize(
                fun=lambda p: self._objective(p, X, times, events, record=True),
                x0=params0,
                method='L-BFGS-B',
                jac=jac,
                options={'maxiter': self.max_iter, 'ftol': self.tol}
            )

            self._set_params_vector(res.x)

            if self.verbose:
                print(f"Optimization {'converged' if res.success else 'stopped'}")
                print(f"Final loss: {res.fun:.6f} (iterations: {res.nit})")

        except (ValueError, AttributeError) as e:
            if self.verbose:
                print(f"Parameter extraction failed: {e}")
                print("Falling back to non-parametric optimization...")
            self._fit_nonparametric(X, times, events)

        # Phase 3: Estimate baseline hazard
        self._estimate_baseline_hazard(X, times, events)

        return self

    def _fit_nonparametric(self, X, times, events):
        """Functional gradient descent for models without extractable parameters."""
        for it in range(self.max_iter):
            risk = self.base_model.predict(X)

            # Compute pseudo-residuals
            order = np.argsort(-times)
            risk_s = risk[order]
            times_s = times[order]
            events_s = events[order]
            exp_risk = np.exp(risk_s)
            residuals = np.zeros_like(risk_s)

            for t in np.unique(times_s[events_s == 1]):
                at_risk = times_s >= t
                failed = (times_s == t) & (events_s == 1)
                n_failed = failed.sum()

                if n_failed > 0:
                    residuals[failed] += 1
                    residuals[at_risk] -= n_failed * exp_risk[at_risk] / (exp_risk[at_risk].sum() + 1e-15)

            # Map back to original order
            res_unsorted = np.empty_like(residuals)
            res_unsorted[order] = residuals

            # Update with gradient step
            y_pseudo = risk + self.learning_rate * res_unsorted
            self.base_model.fit(X, y_pseudo)

            # Check convergence
            loss = self._cox_loss(self.base_model.predict(X), times, events)
            self.loss_history_.append(loss)

            if self.verbose and (it + 1) % 10 == 0:
                print(f"  Iteration {it+1}/{self.max_iter}, Loss: {loss:.6f}")

            if len(self.loss_history_) > 5:
                recent = self.loss_history_[-5:]
                if max(recent) - min(recent) < self.tol * abs(np.mean(recent)):
                    if self.verbose:
                        print(f"  Converged at iteration {it+1}")
                    break

    def _estimate_baseline_hazard(self, X, times, events):
        """Estimate cumulative baseline hazard using Breslow estimator."""
        risk = self.predict(X)
        order = np.argsort(times)
        times_s = times[order]
        events_s = events[order]
        risk_s = risk[order]

        unique_times = np.unique(times_s[events_s == 1])
        if len(unique_times) == 0:
            self.baseline_hazard_ = (np.array([0.0]), np.array([0.0]))
            return

        cum_hazard = np.zeros(len(unique_times))
        for i, t in enumerate(unique_times):
            at_risk = times_s >= t
            n_events = np.sum((times_s == t) & (events_s == 1))
            cum_hazard[i] = n_events / (np.sum(np.exp(risk_s[at_risk])) + 1e-15)

        self.baseline_hazard_ = (unique_times, np.cumsum(cum_hazard))

    def predict(self, X):
        """
        Predict log hazard ratios (risk scores).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        risk_scores : ndarray, shape (n_samples,)
            Predicted log hazard ratios.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        return self.base_model.predict(X)

    def predict_survival_function(self, X, times):
        """
        Predict survival probabilities S(t|x) = exp(-exp(risk) * H0(t)).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        times : array-like, shape (n_times,)
            Time points for prediction.

        Returns
        -------
        survival_probs : ndarray, shape (n_samples, n_times)
            Survival probabilities at each time point.
        """
        check_is_fitted(self)

        if self.baseline_hazard_ is None:
            raise ValueError("Model not fitted")

        risk = self.predict(X)
        t0, H0 = self.baseline_hazard_
        times = np.asarray(times)

        # Interpolate baseline hazard
        Ht = np.interp(times, t0, H0, left=0, right=H0[-1])

        # Compute survival function
        return np.exp(-np.outer(np.exp(risk), Ht))

    def concordance_index(self, X, times, events):
        """
        Compute Harrell's C-index.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        times : array-like
            Observed times.
        events : array-like
            Event indicators.

        Returns
        -------
        c_index : float
            Concordance index (0.5 = random, 1.0 = perfect).
        """
        risk = self.predict(X)

        concordant = 0
        discordant = 0
        tied_risk = 0
        comparable = 0

        event_indices = np.where(events == 1)[0]
        for i in event_indices:
            longer_survival = times > times[i]
            comparable += np.sum(longer_survival)

            if np.any(longer_survival):
                risk_diff = risk[i] - risk[longer_survival]
                concordant += np.sum(risk_diff > 0)
                discordant += np.sum(risk_diff < 0)
                tied_risk += np.sum(risk_diff == 0)

        if comparable == 0:
            return 0.5

        return (concordant + 0.5 * tied_risk) / comparable

    def score(self, X, times, events):
        """Score method for sklearn compatibility (returns C-index)."""
        return self.concordance_index(X, times, events)
