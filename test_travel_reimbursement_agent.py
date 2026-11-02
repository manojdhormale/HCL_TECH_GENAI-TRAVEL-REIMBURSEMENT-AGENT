"""Standalone validation tests for Manoj_dhormale.ipynb.

Run with:
    python -m unittest test_travel_reimbursement_agent.py -v

The notebook is executed in a temporary working directory with OpenRouter
environment variables removed, so this test suite does not make live API calls.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import nbformat
from nbclient import NotebookClient


WORKSPACE = Path(__file__).resolve().parent
NOTEBOOK_PATH = WORKSPACE / "Manoj_dhormale.ipynb"
REQUIRED_RESULT_KEYS = {
    "claim_id",
    "decision",
    "approved_amount",
    "deducted_amount",
    "missing_docs",
    "policy_refs",
    "confidence",
    "explanation",
    "tools_used",
}
EXPECTED_RESULTS = {
    "CLM-001": ("APPROVE", 1110.0, 0.0),
    "CLM-002": ("REJECT", 0.0, 380.0),
    "CLM-003": ("PARTIAL_APPROVE", 840.0, 100.0),
    "CLM-004": ("MANUAL_REVIEW", 0.0, 0.0),
    "CLM-005": ("MANUAL_REVIEW", 0.0, 0.0),
}


class TravelReimbursementNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not NOTEBOOK_PATH.is_file():
            raise AssertionError(f"Notebook not found: {NOTEBOOK_PATH}")

        cls.notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
        cls.notebook = nbformat.validator.normalize(
            nbformat.read(NOTEBOOK_PATH, as_version=4)
        )[1]

        cls.original_environment = {
            name: os.environ.pop(name, None)
            for name in (
                "OPENROUTER_API_KEY",
                "OPENROUTER_MODEL",
                "OPENROUTER_BASE_URL",
            )
        }
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                executed = NotebookClient(
                    cls.notebook,
                    timeout=240,
                    kernel_name="python3",
                    allow_errors=False,
                ).execute(cwd=temporary_directory)
                cls.executed_notebook = executed
        finally:
            for name, value in cls.original_environment.items():
                if value is not None:
                    os.environ[name] = value

        cls.code_cells = [
            cell for cell in cls.executed_notebook.cells if cell.cell_type == "code"
        ]
        cls.stream_outputs = [
            output.text
            for cell in cls.code_cells
            for output in cell.outputs
            if output.output_type == "stream"
        ]
        cls.final_results = cls._read_final_results()

    @classmethod
    def _read_final_results(cls):
        for output in reversed(cls.stream_outputs):
            if '"claim_id": "CLM-001"' not in output:
                continue
            try:
                value = json.loads(output)
            except json.JSONDecodeError:
                continue
            if isinstance(value, list) and len(value) == 5:
                return value
        raise AssertionError("The notebook did not print the final five-claim JSON array.")

    def test_notebook_executes_without_cell_errors(self):
        self.assertTrue(self.code_cells)
        self.assertTrue(all(cell.execution_count is not None for cell in self.code_cells))
        errors = [
            output
            for cell in self.code_cells
            for output in cell.outputs
            if output.output_type == "error"
        ]
        self.assertEqual(errors, [])

    def test_sample_decisions_and_amounts(self):
        actual = {
            result["claim_id"]: (
                result["decision"],
                result["approved_amount"],
                result["deducted_amount"],
            )
            for result in self.final_results
        }
        self.assertEqual(actual, EXPECTED_RESULTS)

    def test_final_results_have_exact_schema(self):
        self.assertEqual(len(self.final_results), 5)
        for result in self.final_results:
            self.assertEqual(set(result), REQUIRED_RESULT_KEYS)
            self.assertIn(
                result["decision"],
                {"APPROVE", "PARTIAL_APPROVE", "REJECT", "MANUAL_REVIEW"},
            )
            self.assertGreaterEqual(result["confidence"], 0)
            self.assertLessEqual(result["confidence"], 1)

    def test_business_rule_tests_pass(self):
        self.assertTrue(
            any(
                "Focused policy, boundary, schema, duplicate, and safe-dispatch tests passed."
                in output
                for output in self.stream_outputs
            )
        )

    def test_dashboard_metrics_are_present(self):
        dashboard_output = next(
            (
                output
                for output in self.stream_outputs
                if "'total_claims': 5" in output
            ),
            "",
        )
        self.assertIn("'total_claims': 5", dashboard_output)
        self.assertIn("'total_claimed': '5650.00'", dashboard_output)
        self.assertIn("'finalized_approved': '1950.00'", dashboard_output)
        self.assertIn("'finalized_deducted': '480.00'", dashboard_output)
        self.assertIn("'pending_manual_review': 2", dashboard_output)

    def test_live_tool_call_evidence_is_saved_without_a_secret(self):
        self.assertNotIn("sk-or-v1-", self.notebook_text)
        self.assertTrue(
            '"source": "llm_tool_call"' in self.notebook_text
            or '\\"source\\": \\"llm_tool_call\\"' in self.notebook_text
        )
        self.assertIn("nvidia/nemotron-3.5-lightning:free", self.notebook_text)

    def test_design_notes_are_not_present(self):
        self.assertNotIn("## Design Notes & Reasoning", self.notebook_text)
        self.assertIn("## Submission checklist", self.notebook_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
