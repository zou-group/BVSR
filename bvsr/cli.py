from __future__ import annotations

import argparse
import copy


def _shows_default(action: argparse.Action) -> bool:
    d = action.default
    return bool(action.option_strings) and d is not None and d is not False and d != "" \
        and d is not argparse.SUPPRESS and action.help is not argparse.SUPPRESS


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str:
        text = action.help or ""
        if _shows_default(action) and "%(default)" not in text:
            text += (" " if text else "") + "(default: %(default)s)"
        return text

    def _format_action(self, action: argparse.Action) -> str:
        if not action.help and _shows_default(action):
            action = copy.copy(action)
            action.help = "(default: %(default)s)"
        return super()._format_action(action)


__all__ = ["HelpFormatter"]
