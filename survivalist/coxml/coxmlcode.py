"""
Comprehensive Survival Analysis Comparison Study
=================================================

Compares CoxMLWrapper against lifelines, scikit-survival, and tests with
all applicable sklearn estimators using defensive programming.

Dependencies:
    pip install numpy pandas scikit-learn lifelines scikit-survival matplotlib seaborn
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from scipy.special import logsumexp
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor,
    Lars,
    LassoLars,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    SGDRegressor,
    TheilSenRegressor,
)
import traceback
from typing import Dict, Any, Optional, Tuple
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# Import all potential sklearn regressors

# Import survival analysis libraries
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index as lifelines_cindex

    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("Warning: lifelines not available")

try:
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    print("Warning: scikit-survival not available")

# Assume CoxMLWrapper is available (from previous artifact)


class CoxMLWrapper(BaseEstimator):
    """
    Universal Cox Proportional Hazards wrapper for ANY sklearn regressor.

    Implements Cox (1972) partial likelihood with:
      h(t|X) = h0(t) * exp(f_theta(X))

    where f_theta(X) is provided by an arbitrary sklearn model.

    This implementation matches the CoxMLWrapper described in the
    Survivalist Premium notebook (NOT lifelines).
    """

    def __init__(
        self, base_model, max_iter=100, tol=1e-4, verbose=True, fd_epsilon=1e-5, learning_rate=0.01, random_state=None
    ):
        self.base_model = base_model
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.fd_epsilon = fd_epsilon
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.loss_history_ = []
        self.baseline_hazard_ = None

    # ------------------------------------------------------------------
    # Cox partial likelihood
    # ------------------------------------------------------------------
    def _cox_loss(self, risk_scores, times, events):
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

            loglik += np.sum(risk_scores[failed])
            loglik -= n_failed * logsumexp(risk_scores[at_risk])

        return -loglik

    # ------------------------------------------------------------------
    # Parameter vector handling (linear / NN models)
    # ------------------------------------------------------------------
    def _final_estimator(self):
        return self.base_model[-1] if isinstance(self.base_model, Pipeline) else self.base_model

    def _get_params_vector(self):
        est = self._final_estimator()
        params = []

        if hasattr(est, "coef_"):
            params.append(est.coef_.ravel())
        if hasattr(est, "intercept_"):
            params.append(np.atleast_1d(est.intercept_).ravel())
        if hasattr(est, "coefs_"):
            for w in est.coefs_:
                params.append(w.ravel())
        if hasattr(est, "intercepts_"):
            for b in est.intercepts_:
                params.append(b.ravel())

        if len(params) == 0:
            raise ValueError("Model has no extractable parameters")

        return np.concatenate(params)

    def _set_params_vector(self, params_flat):
        est = self._final_estimator()
        idx = 0

        if hasattr(est, "coef_"):
            n = est.coef_.size
            est.coef_ = params_flat[idx: idx + n].reshape(est.coef_.shape)
            idx += n

        if hasattr(est, "intercept_"):
            if np.isscalar(est.intercept_):
                est.intercept_ = float(params_flat[idx])
                idx += 1
            else:
                n = est.intercept_.size
                est.intercept_ = params_flat[idx: idx +
                                             n].reshape(est.intercept_.shape)
                idx += n

        if hasattr(est, "coefs_"):
            for i, w in enumerate(est.coefs_):
                n = w.size
                est.coefs_[i] = params_flat[idx: idx + n].reshape(w.shape)
                idx += n

        if hasattr(est, "intercepts_"):
            for i, b in enumerate(est.intercepts_):
                n = b.size
                est.intercepts_[i] = params_flat[idx: idx + n].reshape(b.shape)
                idx += n

    # ------------------------------------------------------------------
    # Optimization (FIXED: gradient computation efficiency)
    # ------------------------------------------------------------------
    def _objective(self, params, X, times, events, record=True):
        """Compute loss, optionally recording to history."""
        self._set_params_vector(params)
        risk = self.base_model.predict(X)
        loss = self._cox_loss(risk, times, events)
        if record:
            self.loss_history_.append(loss)
        return loss

    def _finite_diff_grad(self, params, X, times, events):
        """Compute gradient via finite differences without polluting loss history."""
        grad = np.zeros_like(params)
        # Don't record baseline evaluation
        f0 = self._objective(params, X, times, events, record=False)

        for i in range(len(params)):
            p = params.copy()
            p[i] += self.fd_epsilon
            # Don't record perturbation evaluations
            grad[i] = (self._objective(p, X, times, events,
                       record=False) - f0) / self.fd_epsilon

        return grad

    # ------------------------------------------------------------------
    # Fit with input validation
    # ------------------------------------------------------------------
    def fit(self, X, times, events):
        # Input validation
        X = np.asarray(X, float)
        times = np.asarray(times, float)
        events = np.asarray(events, float)

        if len(X) != len(times) or len(X) != len(events):
            raise ValueError(
                f"Mismatched lengths: X={len(X)}, times={len(times)}, events={len(events)}")

        if len(X) == 0:
            raise ValueError("Empty dataset provided")

        if np.any(times < 0):
            raise ValueError("Negative times detected")

        if not np.all(np.isin(events, [0, 1])):
            raise ValueError("Events must be 0 or 1")

        # Set random state if provided
        if self.random_state is not None and hasattr(self.base_model, "random_state"):
            self.base_model.random_state = self.random_state

        if self.verbose:
            print("Initializing base model...")
        self.base_model.fit(X, times)

        try:
            params0 = self._get_params_vector()

            if self.verbose:
                print(f"Optimizing {len(params0)} parameters...")

            res = minimize(
                fun=lambda p: self._objective(
                    p, X, times, events, record=True),
                x0=params0,
                method="L-BFGS-B",
                jac=lambda p: self._finite_diff_grad(p, X, times, events),
                options={"maxiter": self.max_iter, "ftol": self.tol},
            )

            self._set_params_vector(res.x)

            if self.verbose:
                print(
                    f"Optimization complete. Final loss: {self.loss_history_[-1]:.4f}")

        except ValueError:
            if self.verbose:
                print("Using non-parametric Cox boosting fallback...")
            self._fit_nonparametric(X, times, events)

        self._estimate_baseline_hazard(X, times, events)
        return self

    # ------------------------------------------------------------------
    # Tree-based fallback (functional gradient descent)
    # ------------------------------------------------------------------
    def _fit_nonparametric(self, X, times, events):
        # Check for warm start capability
        warm_rf = False
        base_n = 0
        if isinstance(self.base_model, RandomForestRegressor):
            if hasattr(self.base_model, "warm_start") and hasattr(self.base_model, "n_estimators"):
                warm_rf = self.base_model.warm_start
                base_n = self.base_model.n_estimators

        for it in range(self.max_iter):
            risk = self.base_model.predict(X)
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

                residuals[failed] -= 1
                residuals[at_risk] += n_failed * \
                    exp_risk[at_risk] / exp_risk[at_risk].sum()

            res_unsorted = np.empty_like(residuals)
            res_unsorted[order] = residuals

            # Use configurable learning rate
            y_pseudo = risk + self.learning_rate * res_unsorted

            if warm_rf:
                self.base_model.n_estimators = base_n + it + 1

            self.base_model.fit(X, y_pseudo)

            loss = self._cox_loss(self.base_model.predict(X), times, events)
            self.loss_history_.append(loss)

            if self.verbose and it % 20 == 0:
                print(f"Iter {it:3d}, Loss: {loss:.4f}")

            # Improved convergence check with window
            if len(self.loss_history_) > 5:
                recent_losses = self.loss_history_[-5:]
                if max(recent_losses) - min(recent_losses) < self.tol * abs(np.mean(recent_losses)):
                    if self.verbose:
                        print(f"Converged at iteration {it}")
                    break

    # ------------------------------------------------------------------
    # Baseline hazard + prediction
    # ------------------------------------------------------------------
    def _estimate_baseline_hazard(self, X, times, events):
        risk = self.predict(X)
        order = np.argsort(times)

        times_s = times[order]
        events_s = events[order]
        risk_s = risk[order]

        uniq = np.unique(times_s[events_s == 1])
        cum_h = np.zeros(len(uniq))

        for i, t in enumerate(uniq):
            at_risk = times_s >= t
            d = np.sum((times_s == t) & (events_s == 1))
            cum_h[i] = d / (np.sum(np.exp(risk_s[at_risk])) + 1e-8)

        self.baseline_hazard_ = (uniq, np.cumsum(cum_h))

    def predict(self, X):
        return self.base_model.predict(X)

    def predict_survival_function(self, X, times):
        if self.baseline_hazard_ is None:
            raise ValueError("Model not fitted")

        risk = self.predict(X)
        t0, H0 = self.baseline_hazard_

        # Warn about extrapolation
        max_time = t0[-1]
        if np.any(times > max_time):
            import warnings

            warnings.warn(
                f"Predicting beyond observed times (max={max_time:.2f}). " "Extrapolation assumes constant hazard.",
                UserWarning,
            )

        Ht = np.interp(times, t0, H0, left=0, right=H0[-1])

        return np.exp(-np.outer(np.exp(risk), Ht))

    def concordance_index(self, X, times, events):
        """Compute concordance index (C-index) with improved efficiency."""
        risk = self.predict(X)

        # Vectorized implementation for better performance
        n = len(times)
        concordant = 0
        discordant = 0
        tied_risk = 0
        comparable = 0

        # Only compare pairs where first patient has event
        event_indices = np.where(events == 1)[0]

        for i in event_indices:
            # Find all patients with longer survival
            longer_survival = times > times[i]
            comparable += np.sum(longer_survival)

            # Compare risks
            risk_diff = risk[i] - risk[longer_survival]
            concordant += np.sum(risk_diff > 0)
            discordant += np.sum(risk_diff < 0)
            tied_risk += np.sum(risk_diff == 0)

        if comparable == 0:
            return 0.5

        return (concordant + 0.5 * tied_risk) / comparable

    def score(self, X, times, events):
        return self.concordance_index(X, times, events)

    # ============================================================================
    # Data Generation
    # ============================================================================

    def generate_synthetic_portfolio(
        self, n_samples: int = 5000, random_state: int = 42
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Generate synthetic insurance portfolio with known risk structure.

        Returns:
            X: Feature DataFrame
            times: Survival times
            events: Event indicators (1=death, 0=censored)
        """
        np.random.seed(random_state)

        # Generate covariates
        age = np.clip(np.random.normal(50, 12, n_samples), 25, 80)
        sex = np.random.binomial(1, 0.48, n_samples)
        bmi = np.clip(np.random.normal(27, 5, n_samples), 18, 50)
        smoker = np.random.binomial(1, 0.25, n_samples)
        exercise = np.random.binomial(1, 0.35, n_samples)
        disease = np.random.binomial(1, 0.25, n_samples)

        # Disease severity (1-5) for those with disease
        severity = np.where(
            disease == 1, np.random.randint(1, 6, n_samples), 0)

        # Compute risk factors
        rf_age = 1.04 ** (age - 50)
        rf_sex = 1.5**sex
        rf_smoker = 2.5**smoker
        rf_bmi = 1.03 ** np.maximum(bmi - 25, 0)
        rf_exercise = 0.7**exercise
        rf_disease = 2.0**disease

        # Interaction: smoking × high BMI
        rf_interaction = 1 + 0.5 * smoker * np.maximum((bmi - 30) / 10, 0)

        # Combined hazard rate (Weibull scale parameter)
        lambda_i = 30 / (rf_age * rf_sex * rf_smoker * rf_bmi *
                         rf_exercise * rf_disease * rf_interaction)

        # Generate survival times (Weibull with shape=1.8)
        shape = 1.8
        times = lambda_i * \
            (-np.log(np.random.uniform(0, 1, n_samples))) ** (1 / shape)
        times = np.clip(times, 0.1, 50)

        # Generate informative censoring
        censor_prob = np.where(disease == 1, 0.75 +
                               0.05 * (severity / 5), 0.45 - 0.1 * exercise)
        events = np.random.binomial(1, censor_prob, n_samples)

        # Create feature DataFrame
        X = pd.DataFrame(
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "smoker": smoker,
                "exercise": exercise,
                "disease": disease,
                "severity": severity,
                "age_disease": age * disease,
                "smoker_bmi": smoker * bmi,
            }
        )

        return X, times, events

    # ============================================================================
    # Model Definitions
    # ============================================================================

    def get_sklearn_regressors(self) -> Dict[str, Any]:
        """
        Return dictionary of sklearn regressors to test with CoxMLWrapper.
        Uses defensive defaults to avoid training failures.
        """
        return {
            # Linear models
            "Ridge": Ridge(alpha=1.0, random_state=42),
            "Lasso": Lasso(alpha=0.1, max_iter=2000, random_state=42),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42),
            "BayesianRidge": BayesianRidge(),
            "Lars": Lars(n_nonzero_coefs=5),
            "HuberRegressor": HuberRegressor(max_iter=200),
            # Tree-based models
            "DecisionTree": DecisionTreeRegressor(
                max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42
            ),
            "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_split=20, random_state=42),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=50, max_depth=8, min_samples_split=20, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_iter=50, max_depth=5, learning_rate=0.1, random_state=42
            ),
            "AdaBoost": AdaBoostRegressor(n_estimators=50, learning_rate=0.5, random_state=42),
            # Ensemble
            "Bagging": BaggingRegressor(n_estimators=30, random_state=42),
            # Neighbors
            "KNeighbors": KNeighborsRegressor(n_neighbors=10, weights="distance"),
            # Note: SVR, GaussianProcess, KernelRidge excluded due to computational cost
        }

    def get_baseline_models(self):
        """Return baseline survival models from lifelines and scikit-survival."""
        models = {}

        if LIFELINES_AVAILABLE:
            models["Lifelines_CoxPH"] = CoxPHFitter(penalizer=0.1)

        if SKSURV_AVAILABLE:
            models["SkSurv_CoxPH"] = CoxPHSurvivalAnalysis(alpha=0.1)
            models["SkSurv_RSF"] = RandomSurvivalForest(
                n_estimators=100, min_samples_split=10, min_samples_leaf=5, max_depth=8, random_state=42
            )
            models["SkSurv_GBSA"] = GradientBoostingSurvivalAnalysis(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )

        return models

    # ============================================================================
    # Training and Evaluation
    # ============================================================================

    def train_coxmlwrapper(
        self,
        name: str,
        base_model: Any,
        X_train: np.ndarray,
        times_train: np.ndarray,
        events_train: np.ndarray,
        timeout: int = 300,
    ):
        """
        Train CoxMLWrapper with defensive programming and timeout.

        Returns:
            Trained model or None if failed
        """
        try:
            print(f"  Training CoxMLWrapper({name})...", end=" ")
            start = time.time()

            model = CoxMLWrapper(
                base_model=base_model, max_iter=50, tol=1e-3, verbose=False, learning_rate=0.01, random_state=42
            )

            model.fit(X_train, times_train, events_train)

            elapsed = time.time() - start
            print(f"✓ ({elapsed:.1f}s)")
            return model

        except Exception as e:
            print(f"✗ Failed: {str(e)[:50]}")
            return None

    def train_baseline_model(
        self, name: str, model: Any, X_train: pd.DataFrame, times_train: np.ndarray, events_train: np.ndarray
    ):
        """Train lifelines or scikit-survival model."""
        try:
            print(f"  Training {name}...", end=" ")
            start = time.time()

            if "Lifelines" in name:
                df = X_train.copy()
                df["time"] = times_train
                df["event"] = events_train
                model.fit(df, duration_col="time", event_col="event")
            else:  # scikit-survival
                # Convert to structured array
                y = np.array(
                    [(bool(e), t) for e, t in zip(events_train, times_train)], dtype=[("event", bool), ("time", float)]
                )
                model.fit(X_train.values, y)

            elapsed = time.time() - start
            print(f"✓ ({elapsed:.1f}s)")
            return model

        except Exception as e:
            print(f"✗ Failed: {str(e)[:50]}")
            return None

    def evaluate_model(
        self, model: Any, model_name: str, X_test: np.ndarray, times_test: np.ndarray, events_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model and return metrics.

        Returns:
            Dictionary with c_index, time_taken, and success flag
        """
        try:
            start = time.time()

            if "Lifelines" in model_name:
                # Lifelines CoxPH
                df_test = pd.DataFrame(
                    X_test, columns=[f"x{i}" for i in range(X_test.shape[1])])
                risk_scores = -model.predict_partial_hazard(df_test).values
                c_index = lifelines_cindex(
                    times_test, risk_scores, events_test)

            elif "SkSurv" in model_name:
                # scikit-survival models
                risk_scores = model.predict(X_test)
                c_index = concordance_index_censored(
                    events_test.astype(bool), times_test, risk_scores)[0]

            else:
                # CoxMLWrapper
                risk_scores = model.predict(X_test)
                c_index = model.concordance_index(
                    X_test, times_test, events_test)

            elapsed = time.time() - start

            return {"c_index": c_index, "eval_time": elapsed, "success": True}

        except Exception as e:
            print(f"    Evaluation failed: {str(e)[:50]}")
            return {"c_index": np.nan, "eval_time": np.nan, "success": False}

    # ============================================================================
    # Main Comparison Study
    # ============================================================================

    def run_comparison_study(self, n_samples: int = 250, test_size: float = 0.25, random_state: int = 42):
        """
        Run comprehensive comparison of all models.
        """
        print("=" * 80)
        print("COMPREHENSIVE SURVIVAL ANALYSIS MODEL COMPARISON")
        print("=" * 80)

        # Generate data
        print("\n1. Generating synthetic portfolio...")
        X, times, events = self.generate_synthetic_portfolio(
            n_samples, random_state)
        print(
            f"   Generated {n_samples} samples with {events.sum()} events ({100*events.mean():.1f}%)")

        # Split data
        X_train, X_test, times_train, times_test, events_train, events_test = train_test_split(
            X, times, events, test_size=test_size, random_state=random_state, stratify=events
        )

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        # Results storage
        results = []

        # ========================================================================
        # Part 1: Baseline Models (lifelines, scikit-survival)
        # ========================================================================
        print("\n2. Training baseline survival models...")
        baseline_models = self.get_baseline_models()

        for name, model in baseline_models.items():
            trained_model = self.train_baseline_model(
                name, model, X_train, times_train, events_train)

            if trained_model is not None:
                metrics = self.evaluate_model(
                    trained_model, name, X_test_scaled, times_test, events_test)
                results.append(
                    {
                        "Model": name,
                        "Category": "Baseline",
                        "Base_Learner": name.split("_")[1] if "_" in name else name,
                        "C_Index": metrics["c_index"],
                        "Train_Success": True,
                        "Eval_Success": metrics["success"],
                    }
                )

        # ========================================================================
        # Part 2: CoxMLWrapper with all sklearn estimators
        # ========================================================================
        print("\n3. Training CoxMLWrapper with sklearn estimators...")
        sklearn_regressors = self.get_sklearn_regressors()

        for name, base_model in sklearn_regressors.items():
            trained_model = self.train_coxmlwrapper(
                name, base_model, X_train_scaled, times_train, events_train)

            if trained_model is not None:
                metrics = self.evaluate_model(
                    trained_model, f"CoxML_{name}", X_test_scaled, times_test, events_test)
                results.append(
                    {
                        "Model": f"CoxML_{name}",
                        "Category": "CoxMLWrapper",
                        "Base_Learner": name,
                        "C_Index": metrics["c_index"],
                        "Train_Success": True,
                        "Eval_Success": metrics["success"],
                    }
                )
            else:
                results.append(
                    {
                        "Model": f"CoxML_{name}",
                        "Category": "CoxMLWrapper",
                        "Base_Learner": name,
                        "C_Index": np.nan,
                        "Train_Success": False,
                        "Eval_Success": False,
                    }
                )

        # ========================================================================
        # Part 3: Summarize Results
        # ========================================================================
        print("\n4. Results Summary")
        print("=" * 80)

        df_results = pd.DataFrame(results)

        # Overall statistics
        print(f"\nTotal models attempted: {len(df_results)}")
        print(f"Successfully trained: {df_results['Train_Success'].sum()}")
        print(f"Successfully evaluated: {df_results['Eval_Success'].sum()}")

        # Top performers
        print("\n" + "=" * 80)
        print("TOP 10 MODELS BY C-INDEX")
        print("=" * 80)

        top_models = df_results.sort_values(
            "C_Index", ascending=False).head(10)
        print(top_models[["Model", "Category", "C_Index"]
                         ].to_string(index=False))

        # Category comparison
        print("\n" + "=" * 80)
        print("PERFORMANCE BY CATEGORY")
        print("=" * 80)

        category_stats = (
            df_results[df_results["Eval_Success"]]
            .groupby("Category")
            .agg({"C_Index": ["mean", "std", "min", "max", "count"]})
            .round(4)
        )
        print(category_stats)

        # Best in each category
        print("\n" + "=" * 80)
        print("BEST MODEL IN EACH CATEGORY")
        print("=" * 80)

        for category in df_results["Category"].unique():
            cat_data = df_results[df_results["Category"] == category]
            best = cat_data.loc[cat_data["C_Index"].idxmax()]
            print(f"\n{category}:")
            print(f"  Model: {best['Model']}")
            print(f"  C-Index: {best['C_Index']:.4f}")

        # Failed models
        failed = df_results[~df_results["Train_Success"]]
        if len(failed) > 0:
            print("\n" + "=" * 80)
            print(f"FAILED MODELS ({len(failed)} total)")
            print("=" * 80)
            print(failed[["Model", "Base_Learner"]].to_string(index=False))

        return df_results
