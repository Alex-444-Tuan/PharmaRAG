"""The latency report says what machine it ran on (the CI gate compares against a recording)."""

import os
import sys

from pharmarag.eval.latency import build_stats, cpu_model, render_markdown, runner_info
from tests.unit.eval.test_latency import grid

CPUINFO = """processor\t: 0
vendor_id\t: AuthenticAMD
model name\t: AMD EPYC 7763 64-Core Processor
cpu MHz\t\t: 2445.426

processor\t: 1
model name\t: AMD EPYC 7763 64-Core Processor
"""


def test_cpu_model_is_read_from_proc_cpuinfo(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text(CPUINFO)
    assert cpu_model(f) == "AMD EPYC 7763 64-Core Processor"


def test_cpu_model_accepts_the_arm_linux_spelling(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text("processor\t: 0\nModel\t\t: Raspberry Pi 5 Model B\n")
    assert cpu_model(f) == "Raspberry Pi 5 Model B"


def test_cpu_model_falls_back_when_there_is_no_cpuinfo(tmp_path):
    name = cpu_model(tmp_path / "missing")
    assert name and isinstance(name, str)  # sysctl on macOS, platform.processor() elsewhere
    if sys.platform != "darwin":
        assert name != ""


def test_runner_info_has_cpu_model_and_count(tmp_path):
    f = tmp_path / "cpuinfo"
    f.write_text(CPUINFO)
    info = runner_info(f)
    assert info["cpu_model"] == "AMD EPYC 7763 64-Core Processor"
    assert info["cpu_count"] == os.cpu_count() and info["cpu_count"] >= 1
    assert info["platform"] and info["machine"]


def report(**extra):
    return {
        "split": "replay_fake",
        "timestamp": "t",
        "corpus_version": "c",
        "model_id": "m",
        "runs": 3,
        "config": {"verifier": "fake", "questions": 8, "note": "n"},
        "stats": build_stats(grid(questions=6, reps=3), n_boot=50, seed=0),
        **extra,
    }


def test_the_markdown_summary_names_the_runner():
    md = render_markdown(
        report(
            runner={
                "cpu_model": "AMD EPYC 7763 64-Core Processor",
                "cpu_count": 4,
                "machine": "x86_64",
                "platform": "Linux-6.8",
            }
        )
    )
    assert "- runner: AMD EPYC 7763 64-Core Processor, 4 logical CPUs (Linux-6.8)" in md


def test_a_report_without_a_runner_still_renders():
    """The committed reports in reports/ predate the field."""
    assert "runner:" not in render_markdown(report())
