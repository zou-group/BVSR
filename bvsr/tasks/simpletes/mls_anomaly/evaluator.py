"""SimpleTES evaluator of the mls_anomaly task: the BVSR benchmark evaluator, via ``bvsr.stes.bridge``."""
from bvsr.stes.bridge import BRIDGE_TIMEOUT, evaluate_with

TIMEOUT_SECONDS = BRIDGE_TIMEOUT["mls_anomaly"]


def evaluate(program_path, use_test_data=False):
    return evaluate_with("mls_anomaly", program_path, use_test_data)
