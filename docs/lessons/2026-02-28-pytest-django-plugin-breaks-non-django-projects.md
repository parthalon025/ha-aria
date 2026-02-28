# Lesson: Installing pytest-django Breaks Non-Django Tests in the Same Environment

**Date:** 2026-02-28
**System:** community (pytest-dev/pytest-django)
**Tier:** lesson
**Category:** testing
**Keywords:** pytest, pytest-django, plugin, non-django, DJANGO_SETTINGS_MODULE, monorepo, conftest, plugin isolation
**Source:** https://github.com/pytest-dev/pytest-django/issues/1003

---

## Observation (What Happened)
A monorepo had both Django and non-Django test directories. After installing `pytest-django`, running `pytest` from a non-Django directory failed with: "pytest-django could not find a Django project (no manage.py file could be found)". The non-Django tests had no Django dependency but pytest-django's plugin auto-activated and blocked the test run.

## Analysis (Root Cause — 5 Whys)
**Why #1:** pytest auto-discovers all installed plugins including `pytest-django`, which activates on every pytest invocation regardless of the project type.
**Why #2:** pytest-django's plugin `conftest_options` hook runs at session start and checks for Django configuration, raising an error when `DJANGO_SETTINGS_MODULE` is not set.
**Why #3:** The plugin does not check whether the current working directory is a Django project — it activates globally.
**Why #4:** A single shared virtual environment with both Django and non-Django projects causes cross-contamination because pytest plugins are process-global.
**Why #5:** There is no per-directory plugin enable/disable mechanism in pytest by default.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add `DJANGO_SETTINGS_MODULE` to environment only for Django test runs — or set it in a Django-specific `conftest.py` using `os.environ.setdefault()` | proposed | community | issue #1003 |
| 2 | Use separate virtual environments for Django and non-Django projects in a monorepo — shared venvs with opinionated plugins cause cross-contamination | proposed | community | issue #1003 |
| 3 | Add `django_settings_module = "myapp.settings"` to the `[pytest]`/`[tool.pytest.ini_options]` section of the Django project's `pyproject.toml` so pytest-django only activates in the Django project's test context | proposed | community | issue #1003 |
| 4 | Use `pytest -p no:django` in non-Django test invocations to explicitly disable the plugin per run | proposed | community | issue #1003 |

## Key Takeaway
pytest-django auto-activates globally for all pytest runs in environments where it is installed — it will break non-Django tests unless `DJANGO_SETTINGS_MODULE` is configured or the plugin is explicitly disabled with `-p no:django`; use separate venvs or per-project `pytest.ini` settings to scope plugin activation.
