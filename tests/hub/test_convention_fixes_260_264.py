"""Regression tests for Convention G (#260) and Convention L (#264).

#260 (Convention G): asyncio.get_event_loop() must not appear in aria/modules/
  or aria/shared/ — use get_running_loop() instead.
#264 (Convention L): bare sqlite3.connect() without contextlib.closing() must
  not appear in aria/modules/ or aria/shared/.
"""

from pathlib import Path


def _python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


MODULES_ROOT = Path(__file__).parents[2] / "aria" / "modules"
SHARED_ROOT = Path(__file__).parents[2] / "aria" / "shared"
WATCHDOG = Path(__file__).parents[2] / "aria" / "watchdog.py"

TARGET_FILES = _python_files(MODULES_ROOT) + _python_files(SHARED_ROOT) + [WATCHDOG]


class TestConventionG260:
    """get_event_loop() must not appear in modules/shared scope."""

    def test_no_get_event_loop_in_modules_or_shared_closes_260(self):
        violations = []
        for path in TARGET_FILES:
            source = path.read_text(errors="replace")
            if "get_event_loop()" in source:
                violations.append(str(path.relative_to(MODULES_ROOT.parents[1])))
        assert not violations, f"asyncio.get_event_loop() found (use get_running_loop()): {violations}"


class TestConventionL264:
    """sqlite3.connect() must always be wrapped with contextlib.closing()."""

    def test_no_bare_sqlite3_connect_in_modules_or_shared_closes_264(self):
        violations = []
        for path in TARGET_FILES:
            source = path.read_text(errors="replace")
            if "sqlite3.connect(" not in source:
                continue
            # Check every sqlite3.connect( is preceded by closing( on the same line
            for lineno, line in enumerate(source.splitlines(), 1):
                if "sqlite3.connect(" in line and "closing(" not in line:
                    violations.append(f"{path.name}:{lineno}: {line.strip()}")
        assert not violations, "Bare sqlite3.connect() without closing() found:\n" + "\n".join(violations)
