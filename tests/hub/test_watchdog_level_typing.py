"""Failing test for #326: WatchdogResult.level must be Literal["OK","WARNING","CRITICAL"].

A plain str annotation allows typos that silently miss COOLDOWN_SECONDS lookups.
"""

import typing

from aria.watchdog import WatchdogResult


def test_watchdog_result_level_is_literal_closes_326():
    """WatchdogResult.level must be Literal["OK","WARNING","CRITICAL"] not plain str."""
    hints = typing.get_type_hints(WatchdogResult)
    level_type = hints.get("level")
    # get_args on a Literal returns the allowed values; plain str has no args
    args = typing.get_args(level_type)
    assert args == ("OK", "WARNING", "CRITICAL"), (
        f"WatchdogResult.level should be Literal['OK','WARNING','CRITICAL'] but type is {level_type} with args {args}"
    )
