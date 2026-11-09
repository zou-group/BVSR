"""SimpleTES evaluator of the svg task: the BVSR benchmark evaluator, via ``bvsr.stes.bridge``."""
from bvsr.stes.bridge import BRIDGE_TIMEOUT, evaluate_with

TIMEOUT_SECONDS = BRIDGE_TIMEOUT["svg"]


def evaluate(program_path, use_test_data=False):
    return evaluate_with("svg", program_path, use_test_data)
