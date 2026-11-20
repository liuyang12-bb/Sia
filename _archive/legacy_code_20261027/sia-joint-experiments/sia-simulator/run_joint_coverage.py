"""Persist the expanded integration-test result and console trace."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
import unittest
from pathlib import Path


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", choices=["coverage", "all"], default="coverage")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite = (
        unittest.defaultTestLoader.discover("tests")
        if args.suite == "all"
        else unittest.defaultTestLoader.loadTestsFromName("tests.test_joint_integration_coverage")
    )
    case_ids = [test.id() for test in flatten(suite)]
    stream = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - start
    log = stream.getvalue()
    log_name = "all_tests.log" if args.suite == "all" else "coverage_test.log"
    summary_name = "all_tests_summary.json" if args.suite == "all" else "coverage_summary.json"
    (output / log_name).write_text(log, encoding="utf-8")
    summary = {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "elapsed_seconds": elapsed,
        "cases": case_ids,
        "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
        "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors],
        "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
        "scope": (
            "all joint prototype and integration tests"
            if args.suite == "all"
            else "actual Cluster/JobInfo/JointSiaPolicy/JointScheduler/Job paths"
        ),
    }
    (output / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
