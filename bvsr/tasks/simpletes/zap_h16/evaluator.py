"""SimpleTES evaluator of the zap_h16 task: the BVSR benchmark evaluator, via ``bvsr.stes.bridge``."""
from bvsr.stes.bridge import BRIDGE_TIMEOUT, evaluate_with

TIMEOUT_SECONDS = BRIDGE_TIMEOUT["zap_h16"]


def evaluate(program_path, use_test_data=False):
    return evaluate_with("zap_h16", program_path, use_test_data)
