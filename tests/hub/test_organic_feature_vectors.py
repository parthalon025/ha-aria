"""Unit tests for organic discovery feature vector builder.

Tests the build_feature_matrix function that produces numeric matrices
suitable for HDBSCAN clustering from HA entity data.
"""

import numpy as np

from aria.modules.organic_discovery.feature_vectors import build_feature_matrix

# ============================================================================
# Helpers
# ============================================================================


def make_entity(  # noqa: PLR0913
    entity_id: str,
    state: str = "on",
    device_id: str = "",
    area_id: str = "",
    device_class: str = "",
    unit_of_measurement: str = "",
    attributes: dict | None = None,
) -> dict:
    """Build an entity dict matching the discovery cache format."""
    domain = entity_id.split(".")[0]
    return {
        "entity_id": entity_id,
        "state": state,
        "domain": domain,
        "device_id": device_id,
        "area_id": area_id,
        "device_class": device_class or None,
        "unit_of_measurement": unit_of_measurement or None,
        "attributes": attributes or {},
        "friendly_name": entity_id.replace(".", " ").title(),
    }


def make_device(
    device_id: str,
    area_id: str = "",
    manufacturer: str = "",
) -> dict:
    return {
        "device_id": device_id,
        "name": f"Device {device_id}",
        "area_id": area_id or None,
        "manufacturer": manufacturer or None,
        "model": None,
    }


# ============================================================================
# Tests
# ============================================================================


class TestBuildFeatureMatrix:
    """Tests for build_feature_matrix."""

    def test_correct_shape(self):
        """Matrix has n_entities rows and n_features columns."""
        entities = [
            make_entity("light.living_room", device_id="dev1"),
            make_entity("sensor.temperature", device_id="dev2", device_class="temperature", unit_of_measurement="°C"),
            make_entity("binary_sensor.motion", device_id="dev1", device_class="motion"),
        ]
        devices = {
            "dev1": make_device("dev1", area_id="living_room", manufacturer="Philips"),
            "dev2": make_device("dev2", area_id="kitchen", manufacturer="Aqara"),
        }
        activity_rates = {
            "light.living_room": 12.5,
            "sensor.temperature": 48.0,
            "binary_sensor.motion": 30.0,
        }

        matrix, entity_ids, feature_names = build_feature_matrix(
            entities,
            devices,
            {},
            activity_rates,
        )

        assert isinstance(matrix, np.ndarray)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] == len(feature_names)
        assert len(entity_ids) == 3

    def test_entity_ids_match_input_order(self):
        """Entity IDs in output match input order exactly."""
        entities = [
            make_entity("light.a"),
            make_entity("switch.b"),
            make_entity("sensor.c"),
        ]

        _, entity_ids, _ = build_feature_matrix(entities, {}, {}, {})

        assert entity_ids == ["light.a", "switch.b", "sensor.c"]

    def test_includes_domain_features(self):
        """Domain one-hot features are present and correct."""
        entities = [
            make_entity("light.lamp"),
            make_entity("sensor.temp"),
        ]

        matrix, entity_ids, feature_names = build_feature_matrix(
            entities,
            {},
            {},
            {},
        )

        # Find domain feature columns
        domain_features = [f for f in feature_names if f.startswith("domain_")]
        assert len(domain_features) >= 2
        assert "domain_light" in feature_names
        assert "domain_sensor" in feature_names

        # light entity should have domain_light=1
        light_idx = entity_ids.index("light.lamp")
        light_col = feature_names.index("domain_light")
        assert matrix[light_idx, light_col] == 1.0

        # sensor entity should have domain_sensor=1, domain_light=0
        sensor_idx = entity_ids.index("sensor.temp")
        sensor_col = feature_names.index("domain_sensor")
        assert matrix[sensor_idx, sensor_col] == 1.0
        assert matrix[sensor_idx, light_col] == 0.0

    def test_activity_rates_included(self):
        """Activity rates are correctly placed in the matrix."""
        entities = [
            make_entity("light.a"),
            make_entity("sensor.b"),
        ]
        activity_rates = {
            "light.a": 25.0,
            "sensor.b": 100.0,
        }

        matrix, entity_ids, feature_names = build_feature_matrix(
            entities,
            {},
            {},
            activity_rates,
        )

        assert "avg_daily_changes" in feature_names
        rate_col = feature_names.index("avg_daily_changes")
        assert matrix[entity_ids.index("light.a"), rate_col] == 25.0
        assert matrix[entity_ids.index("sensor.b"), rate_col] == 100.0

    def test_missing_activity_rate_defaults_to_zero(self):
        """Entity not in activity_rates gets 0."""
        entities = [make_entity("light.a")]

        matrix, _, feature_names = build_feature_matrix(entities, {}, {}, {})

        rate_col = feature_names.index("avg_daily_changes")
        assert matrix[0, rate_col] == 0.0

    def test_handles_missing_device(self):
        """Entity without device_id or with unknown device doesn't crash.

        Area and manufacturer should be 0 across all one-hot columns.
        """
        entities = [
            make_entity("sensor.orphan"),  # no device_id
            make_entity("light.ghost", device_id="nonexistent"),
        ]
        devices = {}  # empty device registry

        matrix, entity_ids, feature_names = build_feature_matrix(
            entities,
            devices,
            {},
            {},
        )

        # Should still produce valid matrix
        assert matrix.shape[0] == 2
        assert not np.isnan(matrix).any()

    def test_handles_empty_input(self):
        """Empty entity list returns empty matrix."""
        matrix, entity_ids, feature_names = build_feature_matrix([], {}, {}, {})

        assert matrix.shape[0] == 0
        assert len(entity_ids) == 0
        assert len(feature_names) > 0  # features are still defined

    def test_area_resolved_via_device(self):
        """Area one-hot is set by resolving entity → device → area."""
        entities = [
            make_entity("light.lamp", device_id="dev1"),
        ]
        devices = {
            "dev1": make_device("dev1", area_id="bedroom"),
        }

        matrix, _, feature_names = build_feature_matrix(
            entities,
            devices,
            {},
            {},
        )

        [f for f in feature_names if f.startswith("area_")]
        assert "area_bedroom" in feature_names
        col = feature_names.index("area_bedroom")
        assert matrix[0, col] == 1.0

    def test_area_direct_on_entity_takes_priority(self):
        """Entity with direct area_id uses it over device area."""
        entities = [
            make_entity("light.lamp", device_id="dev1", area_id="kitchen"),
        ]
        devices = {
            "dev1": make_device("dev1", area_id="bedroom"),
        }

        matrix, _, feature_names = build_feature_matrix(
            entities,
            devices,
            {},
            {},
        )

        assert "area_kitchen" in feature_names
        col = feature_names.index("area_kitchen")
        assert matrix[0, col] == 1.0

    def test_manufacturer_from_device(self):
        """Manufacturer one-hot is resolved via device registry."""
        entities = [
            make_entity("light.a", device_id="dev1"),
            make_entity("light.b", device_id="dev2"),
        ]
        devices = {
            "dev1": make_device("dev1", manufacturer="Philips"),
            "dev2": make_device("dev2", manufacturer="IKEA"),
        }

        matrix, entity_ids, feature_names = build_feature_matrix(
            entities,
            devices,
            {},
            {},
        )

        assert "manufacturer_Philips" in feature_names
        assert "manufacturer_IKEA" in feature_names

        a_idx = entity_ids.index("light.a")
        philips_col = feature_names.index("manufacturer_Philips")
        assert matrix[a_idx, philips_col] == 1.0

    def test_state_cardinality(self):
        """State cardinality is set based on entity characteristics."""
        entities = [
            make_entity("binary_sensor.door", state="on", device_class="door"),
            make_entity("sensor.power", state="150.5", unit_of_measurement="W"),
            make_entity("light.lamp", state="on"),
        ]

        matrix, entity_ids, feature_names = build_feature_matrix(
            entities,
            {},
            {},
            {},
        )

        assert "state_cardinality" in feature_names
        card_col = feature_names.index("state_cardinality")

        # binary_sensor → 2
        assert matrix[entity_ids.index("binary_sensor.door"), card_col] == 2.0
        # sensor with unit → numeric → 100
        assert matrix[entity_ids.index("sensor.power"), card_col] == 100.0
        # other → 5
        assert matrix[entity_ids.index("light.lamp"), card_col] == 5.0

    def test_available_flag(self):
        """Available flag is 0 for unavailable entities, 1 otherwise."""
        entities = [
            make_entity("light.ok", state="on"),
            make_entity("sensor.dead", state="unavailable"),
        ]

        matrix, entity_ids, feature_names = build_feature_matrix(
            entities,
            {},
            {},
            {},
        )

        assert "available" in feature_names
        avail_col = feature_names.index("available")
        assert matrix[entity_ids.index("light.ok"), avail_col] == 1.0
        assert matrix[entity_ids.index("sensor.dead"), avail_col] == 0.0

    def test_capability_flags(self):
        """Capability flags are set from entity attributes."""
        entities = [
            make_entity(
                "light.rgb",
                attributes={
                    "brightness": 200,
                    "color_temp": 350,
                    "rgb_color": [255, 0, 0],
                },
            ),
            make_entity(
                "climate.hvac",
                attributes={
                    "hvac_modes": ["heat", "cool"],
                    "temperature": 21,
                },
            ),
            make_entity("sensor.basic"),
        ]

        matrix, entity_ids, feature_names = build_feature_matrix(
            entities,
            {},
            {},
            {},
        )

        # Light capabilities
        rgb_idx = entity_ids.index("light.rgb")
        assert matrix[rgb_idx, feature_names.index("has_brightness")] == 1.0
        assert matrix[rgb_idx, feature_names.index("has_color_temp")] == 1.0
        assert matrix[rgb_idx, feature_names.index("has_rgb")] == 1.0

        # Climate capabilities
        hvac_idx = entity_ids.index("climate.hvac")
        assert matrix[hvac_idx, feature_names.index("has_hvac")] == 1.0
        assert matrix[hvac_idx, feature_names.index("has_temperature_target")] == 1.0

        # Basic sensor — no capabilities
        basic_idx = entity_ids.index("sensor.basic")
        assert matrix[basic_idx, feature_names.index("has_brightness")] == 0.0
        assert matrix[basic_idx, feature_names.index("has_hvac")] == 0.0

    def test_device_class_one_hot(self):
        """Device class one-hot features are present."""
        entities = [
            make_entity("sensor.temp", device_class="temperature"),
            make_entity("binary_sensor.motion", device_class="motion"),
        ]

        matrix, entity_ids, feature_names = build_feature_matrix(
            entities,
            {},
            {},
            {},
        )

        assert "device_class_temperature" in feature_names
        assert "device_class_motion" in feature_names

        temp_idx = entity_ids.index("sensor.temp")
        temp_col = feature_names.index("device_class_temperature")
        assert matrix[temp_idx, temp_col] == 1.0

    def test_unit_of_measurement_one_hot(self):
        """Unit of measurement one-hot features are present."""
        entities = [
            make_entity("sensor.power", unit_of_measurement="W"),
            make_entity("sensor.temp", unit_of_measurement="°C"),
        ]

        matrix, entity_ids, feature_names = build_feature_matrix(
            entities,
            {},
            {},
            {},
        )

        assert "unit_W" in feature_names
        assert "unit_°C" in feature_names

        power_idx = entity_ids.index("sensor.power")
        w_col = feature_names.index("unit_W")
        assert matrix[power_idx, w_col] == 1.0

    def test_no_nan_values(self):
        """Matrix should never contain NaN."""
        entities = [
            make_entity("light.a", device_id="dev1"),
            make_entity("sensor.b", state="unavailable"),
            make_entity("binary_sensor.c", device_class="motion"),
        ]
        devices = {"dev1": make_device("dev1", area_id="room", manufacturer="Acme")}

        matrix, _, _ = build_feature_matrix(entities, devices, {}, {})

        assert not np.isnan(matrix).any()

    def test_all_values_finite(self):
        """All matrix values must be finite numbers."""
        entities = [
            make_entity("light.a"),
            make_entity("sensor.b", unit_of_measurement="%"),
        ]
        activity_rates = {"light.a": 999.9}

        matrix, _, _ = build_feature_matrix(entities, {}, {}, activity_rates)

        assert np.isfinite(matrix).all()
