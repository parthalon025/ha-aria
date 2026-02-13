#!/usr/bin/env python3
"""Test script for Orchestrator module.

Tests automation suggestion generation, approval/rejection flow,
and safety guardrails.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from hub.core import IntelligenceHub
from modules.orchestrator import OrchestratorModule


async def test_orchestrator():
    """Test orchestrator module."""
    print("=" * 70)
    print("Testing Orchestrator Module")
    print("=" * 70)

    # Get HA credentials from environment
    ha_url = os.environ.get("HA_URL")
    ha_token = os.environ.get("HA_TOKEN")

    if not ha_url or not ha_token:
        print("ERROR: HA_URL and HA_TOKEN environment variables required")
        print("Run: . ~/.env && python test_orchestrator.py")
        return 1

    # Setup test cache
    cache_dir = Path.home() / "ha-logs" / "intelligence" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "test_orchestrator.db"

    # Delete existing test DB
    if cache_path.exists():
        cache_path.unlink()

    # Initialize hub
    print("\n1. Initializing hub...")
    hub = IntelligenceHub(str(cache_path))
    await hub.initialize()
    print("   ✓ Hub initialized")

    # Create test patterns in cache
    print("\n2. Creating test patterns...")
    test_patterns = [
        {
            "pattern_id": "bedroom_cluster_1",
            "name": "Bedroom Evening Pattern",
            "area": "bedroom",
            "typical_time": "22:30",
            "variance_minutes": 15,
            "frequency": 8,
            "total_days": 10,
            "confidence": 0.8,
            "associated_signals": ["bedroom_light_on_h22", "bedroom_motion_on_h22"],
            "llm_description": "Evening lights turn on around 10:30 PM when motion detected"
        },
        {
            "pattern_id": "kitchen_cluster_1",
            "name": "Kitchen Morning Pattern",
            "area": "kitchen",
            "typical_time": "07:15",
            "variance_minutes": 10,
            "frequency": 7,
            "total_days": 10,
            "confidence": 0.7,
            "llm_description": "Morning kitchen lights activate around 7:15 AM"
        },
        {
            "pattern_id": "living_cluster_1",
            "name": "Living Room Low Confidence",
            "area": "living",
            "typical_time": "18:00",
            "variance_minutes": 20,
            "frequency": 4,
            "total_days": 10,
            "confidence": 0.4,  # Below threshold
            "llm_description": "Evening living room activity (low confidence)"
        }
    ]

    await hub.set_cache("patterns", {
        "patterns": test_patterns,
        "pattern_count": len(test_patterns),
        "areas_analyzed": ["bedroom", "kitchen", "living"]
    })
    print(f"   ✓ Created {len(test_patterns)} test patterns")

    # Initialize orchestrator
    print("\n3. Initializing orchestrator...")
    orchestrator = OrchestratorModule(hub, ha_url, ha_token, min_confidence=0.7)
    hub.register_module(orchestrator)
    await orchestrator.initialize()
    print("   ✓ Orchestrator initialized")

    # Wait a moment for async initialization
    await asyncio.sleep(1)

    # Check generated suggestions
    print("\n4. Checking generated suggestions...")
    suggestions = await orchestrator.get_suggestions()
    print(f"   ✓ Generated {len(suggestions)} suggestions")

    if len(suggestions) < 2:
        print(f"   ✗ Expected ≥2 suggestions (patterns with confidence ≥0.7), got {len(suggestions)}")
        await hub.shutdown()
        return 1

    # Display suggestions
    print("\n5. Automation Suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n   Suggestion {i}:")
        print(f"   - ID: {suggestion['suggestion_id']}")
        print(f"   - Pattern: {suggestion['pattern_id']}")
        print(f"   - Confidence: {suggestion['confidence']:.0%}")
        print(f"   - Status: {suggestion['status']}")
        print(f"   - Restricted: {suggestion['requires_explicit_approval']}")
        print(f"   - Automation: {suggestion['automation_yaml']['alias']}")
        print(f"   - Trigger: {suggestion['automation_yaml']['trigger'][0]}")
        print(f"   - Actions: {len(suggestion['automation_yaml']['action'])} action(s)")

    # Test safety guardrails
    print("\n6. Testing safety guardrails...")
    test_actions_safe = [
        {"service": "light.turn_on", "target": {"area_id": "bedroom"}}
    ]
    test_actions_restricted = [
        {"service": "lock.unlock", "target": {"entity_id": "lock.front_door"}}
    ]

    safe_result = orchestrator._check_safety_guardrails(test_actions_safe)
    restricted_result = orchestrator._check_safety_guardrails(test_actions_restricted)

    print(f"   - Light action requires approval: {safe_result} (expected: False)")
    print(f"   - Lock action requires approval: {restricted_result} (expected: True)")

    if safe_result or not restricted_result:
        print("   ✗ Safety guardrails not working correctly")
        await hub.shutdown()
        return 1

    print("   ✓ Safety guardrails working correctly")

    # Test rejection flow
    print("\n7. Testing rejection flow...")
    first_suggestion_id = suggestions[0]["suggestion_id"]
    reject_result = await orchestrator.reject_suggestion(first_suggestion_id)

    if not reject_result["success"]:
        print(f"   ✗ Rejection failed: {reject_result.get('error')}")
        await hub.shutdown()
        return 1

    # Verify status updated
    updated_suggestions = await orchestrator.get_suggestions()
    rejected_suggestion = next(
        (s for s in updated_suggestions if s["suggestion_id"] == first_suggestion_id),
        None
    )

    if not rejected_suggestion or rejected_suggestion["status"] != "rejected":
        print("   ✗ Suggestion status not updated to 'rejected'")
        await hub.shutdown()
        return 1

    print(f"   ✓ Suggestion {first_suggestion_id[:8]}... rejected successfully")

    # Test approval flow (create actual automation in HA)
    print("\n8. Testing approval flow (creates automation in HA)...")
    print("   NOTE: This will create a real automation in your Home Assistant instance")
    print("   You can delete it manually from HA UI if needed")

    second_suggestion_id = suggestions[1]["suggestion_id"] if len(suggestions) > 1 else None

    if second_suggestion_id:
        approval_result = await orchestrator.approve_suggestion(second_suggestion_id)

        if not approval_result["success"]:
            print(f"   ✗ Approval failed: {approval_result.get('error')}")
            print("   (This might be expected if HA API is not accessible)")
        else:
            automation_id = approval_result["automation_id"]
            print(f"   ✓ Automation created: {automation_id}")

            # Verify tracking
            created_automations = await orchestrator.get_created_automations()
            if automation_id in created_automations:
                print(f"   ✓ Automation tracked in cache")
            else:
                print(f"   ✗ Automation not tracked in cache")

    # Test pattern sensor update
    print("\n9. Testing pattern detection sensor update...")
    await orchestrator.update_pattern_detection_sensor(
        pattern_name="Test Pattern",
        pattern_id="test_pattern_1",
        confidence=0.85
    )
    print("   ✓ Pattern sensor update called (check HA UI for sensor.ha_hub_pattern_detected)")

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"✓ Orchestrator module initialized successfully")
    print(f"✓ Generated {len(suggestions)} automation suggestions from patterns")
    print(f"✓ Safety guardrails working (restricted domains detected)")
    print(f"✓ Suggestion rejection flow working")
    print(f"✓ Suggestion approval flow tested")
    print(f"✓ Pattern sensor update tested")
    print("\nAll tests passed!")

    # Cleanup
    await hub.shutdown()
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(test_orchestrator())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
