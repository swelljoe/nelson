import json

from nelson.db import Database
from nelson.remediate import build_remediation_prompt, extract_git_diff


def test_extract_git_diff_accepts_plain_or_fenced_output():
    patch = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-bad
+good
"""
    assert extract_git_diff(patch) == patch
    assert extract_git_diff(f"Here is the fix:\n```diff\n{patch}```") == patch
    assert extract_git_diff("No patch available") is None


def test_remediation_prompt_contains_only_reported_finding_fields():
    finding = {
        "file": "src/Foo.java",
        "line_start": 42,
        "confidence": "high",
        "cwe": "CWE-22",
        "description": "Untrusted path escapes the destination.",
    }
    prompt = build_remediation_prompt(finding)
    assert "src/Foo.java:42" in prompt
    assert "Untrusted path escapes" in prompt
    assert "CWE-22" in prompt


def test_remediation_run_persists_independent_stage_results(tmp_path):
    db = Database(tmp_path / "test.db")
    case_id = db.upsert_case(
        {
            "source": "manual",
            "ext_id": "CVE-test",
            "status": "vetted",
        }
    )
    competitor_id = db.upsert_competitor(
        {
            "name": "raw-api-loop/test",
            "model": "test",
            "runtime": "raw-api-loop",
        }
    )
    run_id = db.create_run(case_id, competitor_id, "src/Foo.java")
    db.complete_run(run_id)
    finding_id = db.add_run_finding(
        run_id,
        file="src/Foo.java",
        line_start=42,
        description="path traversal",
    )
    remediation_id = db.create_remediation_run(
        finding_id,
        competitor_id,
        config={"thinking": True, "max_output_tokens": 16384},
    )
    db.start_remediation_run(remediation_id)
    db.complete_remediation_run(
        remediation_id,
        tokens_in=100,
        tokens_out=200,
        cost_usd=0.0,
        wall_clock_s=3.5,
        transcript_path="run.jsonl",
        raw_output="diff --git ...",
        patch_text="diff --git ...",
        patch_applied=True,
        build_passed=True,
        witnesses_passed=False,
        controls_passed=True,
        verified=False,
        error_msg=None,
    )

    row = db.get_remediation_run(remediation_id)
    assert row["status"] == "complete"
    assert json.loads(row["config"])["max_output_tokens"] == 16384
    assert row["patch_applied"] == 1
    assert row["build_passed"] == 1
    assert row["witnesses_passed"] == 0
    assert row["controls_passed"] == 1
    assert row["verified"] == 0
    context = db.get_run_finding_context(finding_id)
    assert context["case_ext_id"] == "CVE-test"
    assert context["detection_competitor_name"] == "raw-api-loop/test"
