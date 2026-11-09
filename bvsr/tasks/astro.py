from .base import Task, read_upstream


def _read(mission: str, fname: str) -> str:
    return read_upstream("SIMPLETES_ASTRO_HOME", "datasets", "astrodynamics", mission, fname)


class _AstroTask(Task):
    mission = ""
    init_score = 0.0
    marker = "load_problem_for_candidate"

    transplant = ("Implement a REAL transplant of THAT ONE mechanism into the trajectory search — "
                  "not a generic method wearing its name. ONE mechanism only; do NOT add other "
                  "tricks.")
    code_format = "<the COMPLETE program file, header block kept verbatim>"

    def interface(self):
        return ("OFFICIAL TASK INSTRUCTION (verbatim):\n"
                + _read(self.mission, f"{self.mission}.txt")
                + "\n\nSCORING: combined_score = 2*target_dv/(target_dv + total_dv), in [0,1) — "
                "higher = cheaper trajectory; an invalid/failing program scores 0.0. "
                f"The official init program measures {self.init_score:.4f}.")

    def baseline_pool(self):
        return ("EXISTING BASELINE (the official init program — keep its pre-defined bindings "
                "block verbatim; rewrite the algorithm below it):\n```python\n"
                + _read(self.mission, "init_program.py") + "\n```")

    def baselines(self):
        return {"init_program": self.init_score}

    def guidance(self):
        return ("\n3'. Transplant THAT ONE mechanism into the SEARCH/OPTIMIZATION strategy of the "
                "trajectory design (how launch windows, flyby sequences, times-of-flight and DSMs "
                "are explored and refined) — not a rename of the baseline.\n"
                "- The output must be the COMPLETE program file: keep the header block (sys.path "
                "setup, problem = load_problem_for_candidate(__file__), tools = Tools()) EXACTLY "
                "as in the baseline; only the code between the EVOLVE-BLOCK markers may change.\n"
                "- Imports limited to numpy/scipy/itertools/math/random. All physics must go "
                "through the provided `problem` and `tools` objects — no other I/O, no network.\n"
                "- Keep total runtime well under the task timeout; seed any randomness.")


class AstroCassiniTask(_AstroTask):
    name = "astro_cassini"
    title = "Cassini"
    mission = "cassini"
    init_score = 0.27855


class AstroVoyager2Task(_AstroTask):
    name = "astro_voyager2"
    title = "Voyager-2"
    mission = "voyager_2"
    init_score = 0.66103
