"""Tests for SHAP-based model explainability."""

import pytest

try:
    import shap  # noqa: F401

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@pytest.mark.skipif(not HAS_SHAP, reason="shap not installed")
class TestSHAPExplainability:
    def test_explain_prediction_returns_contributions(self):
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        from aria.engine.analysis.explainability import explain_prediction

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        y = X[:, 0] * 5 + X[:, 3] * 3 + rng.standard_normal(100)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X_scaled, y)

        names = [f"feat_{i}" for i in range(10)]
        sample = X[0]

        contributions = explain_prediction(model, scaler, names, sample, top_n=5)

        assert len(contributions) == 5
        assert all("feature" in c and "contribution" in c for c in contributions)
        top_features = [c["feature"] for c in contributions]
        assert "feat_0" in top_features

    def test_explain_prediction_has_direction_and_raw_value(self):
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        from aria.engine.analysis.explainability import explain_prediction

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = X[:, 0] * 10 + rng.standard_normal(100)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X_scaled, y)

        names = [f"feat_{i}" for i in range(5)]
        contributions = explain_prediction(model, scaler, names, X[0], top_n=3)

        for c in contributions:
            assert c["direction"] in ("positive", "negative")
            assert "raw_value" in c
            assert isinstance(c["raw_value"], float)

    def test_explain_prediction_sorted_by_absolute_contribution(self):
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        from aria.engine.analysis.explainability import explain_prediction

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        y = X[:, 0] * 5 + X[:, 3] * 3 + rng.standard_normal(100)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X_scaled, y)

        names = [f"feat_{i}" for i in range(10)]
        contributions = explain_prediction(model, scaler, names, X[0], top_n=10)

        values = [c["contribution"] for c in contributions]
        assert values == sorted(values, reverse=True)

    def test_build_attribution_report(self):
        from aria.engine.analysis.explainability import build_attribution_report

        contributions = [
            {"feature": "weather_temp_f", "contribution": 35.2, "direction": "positive"},
            {"feature": "people_home_count", "contribution": 22.1, "direction": "positive"},
            {"feature": "is_weekend", "contribution": -12.7, "direction": "negative"},
        ]
        report = build_attribution_report(
            metric="power_watts",
            predicted=450.0,
            actual=520.0,
            contributions=contributions,
        )
        assert report["metric"] == "power_watts"
        assert report["delta"] == 70.0
        assert report["predicted"] == 450.0
        assert report["actual"] == 520.0
        assert len(report["top_drivers"]) == 3

    def test_build_attribution_report_negative_delta(self):
        from aria.engine.analysis.explainability import build_attribution_report

        report = build_attribution_report(
            metric="lights_on",
            predicted=10.0,
            actual=7.0,
            contributions=[],
        )
        assert report["delta"] == -3.0
        assert report["top_drivers"] == []


class TestExplainabilityImport:
    def test_has_shap_flag_exists(self):
        from aria.engine.analysis.explainability import HAS_SHAP

        assert isinstance(HAS_SHAP, bool)

    def test_build_attribution_report_works_without_shap(self):
        """build_attribution_report has no shap dependency."""
        from aria.engine.analysis.explainability import build_attribution_report

        report = build_attribution_report(metric="test", predicted=1.0, actual=2.0, contributions=[])
        assert report["metric"] == "test"
