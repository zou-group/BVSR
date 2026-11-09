"""SimpleTES evaluator of the sl_lrbsz task: the BVSR benchmark evaluator, via ``bvsr.stes.bridge``."""
from bvsr.stes.bridge import BRIDGE_TIMEOUT, evaluate_with

TIMEOUT_SECONDS = BRIDGE_TIMEOUT["sl_lrbsz"]


def evaluate(program_path, use_test_data=False):
    return evaluate_with("sl_lrbsz", program_path, use_test_data)
