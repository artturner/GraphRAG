"""Tests for CLI scripts (ingest, eval, query)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Ingest script tests
# ---------------------------------------------------------------------------


class TestIngestScript:
    """Tests for scripts/ingest.py."""

    def test_import(self):
        from scripts.ingest import main, create_parser
        assert callable(main)
        assert callable(create_parser)

    def test_parser_defaults(self):
        from scripts.ingest import create_parser

        parser = create_parser()
        args = parser.parse_args([])

        assert args.corpus is None
        assert args.config == "configs/default.yaml"
        assert args.chunk_size == 500
        assert args.chunk_overlap == 0
        assert args.chunker == "fixed"
        assert args.min_size == 200
        assert args.max_size == 1000
        assert args.no_progress is False
        assert args.verbose is False
        assert args.index is False

    def test_parser_custom_args(self):
        from scripts.ingest import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--corpus", "./mydata",
            "--chunk-size", "300",
            "--chunk-overlap", "50",
            "--chunker", "sentence",
            "--min-size", "100",
            "--max-size", "800",
            "--no-progress",
            "--verbose",
            "--index",
        ])

        assert args.corpus == "./mydata"
        assert args.chunk_size == 300
        assert args.chunk_overlap == 50
        assert args.chunker == "sentence"
        assert args.min_size == 100
        assert args.max_size == 800
        assert args.no_progress is True
        assert args.verbose is True
        assert args.index is True

    def test_missing_corpus_returns_1(self):
        from scripts.ingest import main

        rc = main(["--corpus", "/nonexistent/path/surely"])
        assert rc == 1

    def test_progress_bar(self):
        from scripts.ingest import _progress_bar

        bar = _progress_bar(5, 10, width=20)
        assert "5/10" in bar
        assert "#" in bar
        assert "-" in bar

    def test_progress_bar_complete(self):
        from scripts.ingest import _progress_bar

        bar = _progress_bar(10, 10, width=10)
        assert "10/10" in bar
        assert "-" not in bar.split("]")[0]  # all filled

    def test_progress_bar_zero_total(self):
        from scripts.ingest import _progress_bar

        bar = _progress_bar(0, 0, width=10)
        assert "?" in bar

    def test_print_progress(self, capsys):
        from scripts.ingest import print_progress
        from src.ingestion import IngestProgress

        progress = IngestProgress(
            documents_processed=3,
            chunks_created=15,
            current_file="document.txt",
        )
        print_progress(progress, total_docs=10)
        captured = capsys.readouterr()
        assert "3/10" in captured.out or "3" in captured.out
        assert "15" in captured.out

    def test_print_summary(self, capsys):
        from scripts.ingest import print_summary
        from src.ingestion import IngestProgress

        result = IngestProgress(
            documents_processed=5,
            chunks_created=42,
            errors=[],
        )
        print_summary(result, elapsed=2.5)
        captured = capsys.readouterr()
        assert "5" in captured.out
        assert "42" in captured.out
        assert "2.5" in captured.out

    def test_print_summary_with_errors(self, capsys):
        from scripts.ingest import print_summary
        from src.ingestion import IngestProgress

        result = IngestProgress(
            documents_processed=3,
            chunks_created=20,
            errors=["file1 failed", "file2 failed"],
        )
        print_summary(result, elapsed=1.0, verbose=True)
        captured = capsys.readouterr()
        assert "2" in captured.out  # error count
        assert "file1 failed" in captured.out

    def test_print_summary_with_indexed(self, capsys):
        from scripts.ingest import print_summary
        from src.ingestion import IngestProgress

        result = IngestProgress(documents_processed=5, chunks_created=42)
        print_summary(result, elapsed=1.0, indexed=42)
        captured = capsys.readouterr()
        assert "indexed" in captured.out.lower()


# ---------------------------------------------------------------------------
# Eval script tests
# ---------------------------------------------------------------------------


class TestEvalScript:
    """Tests for scripts/eval.py."""

    def test_import(self):
        from scripts.eval import main, create_parser
        assert callable(main)
        assert callable(create_parser)

    def test_parser_defaults(self):
        from scripts.eval import create_parser

        parser = create_parser()
        args = parser.parse_args([])

        assert args.suite is None
        assert args.dataset is None
        assert args.output is None
        assert args.format == "all"
        assert args.baseline is None
        assert args.verbose is False

    def test_parser_custom_args(self):
        from scripts.eval import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "--suite", "sample_qna",
            "--output", "reports/",
            "--format", "json",
            "--baseline", "reports/baseline.json",
            "--verbose",
        ])

        assert args.suite == "sample_qna"
        assert args.output == "reports/"
        assert args.format == "json"
        assert args.baseline == "reports/baseline.json"
        assert args.verbose is True

    def test_resolve_dataset_builtin(self):
        from scripts.eval import resolve_dataset_path

        class NS:
            dataset = None
            suite = "sample_qna"

        path = resolve_dataset_path(NS())
        assert path.name == "sample_qna.yaml"
        assert path.exists()

    def test_resolve_dataset_custom(self, tmp_path):
        from scripts.eval import resolve_dataset_path

        p = tmp_path / "custom.yaml"
        p.write_text(yaml.dump({"name": "x", "questions": [{"question": "Q?"}]}))

        class NS:
            dataset = str(p)
            suite = None

        path = resolve_dataset_path(NS())
        assert path == p

    def test_resolve_dataset_unknown_suite(self):
        from scripts.eval import resolve_dataset_path

        class NS:
            dataset = None
            suite = "nonexistent_suite"

        with pytest.raises(SystemExit):
            resolve_dataset_path(NS())

    def test_resolve_dataset_missing_file(self):
        from scripts.eval import resolve_dataset_path

        class NS:
            dataset = "/nonexistent/path.yaml"
            suite = None

        with pytest.raises(SystemExit):
            resolve_dataset_path(NS())

    def test_load_baseline(self, tmp_path):
        from scripts.eval import load_baseline

        data = {
            "metrics": {
                "avg_groundedness": 0.85,
                "avg_relevance": 0.90,
                "refusal_accuracy": 1.0,
            },
        }
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        baseline = load_baseline(str(path))
        assert baseline is not None
        assert baseline["avg_groundedness"] == 0.85

    def test_load_baseline_not_found(self):
        from scripts.eval import load_baseline

        assert load_baseline("/nonexistent/baseline.json") is None

    def test_load_baseline_invalid_json(self, tmp_path):
        from scripts.eval import load_baseline

        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        assert load_baseline(str(path)) is None

    def test_delta_positive(self):
        from scripts.eval import _delta

        result = _delta(0.90, 0.85)
        assert "+0.0500" in result

    def test_delta_negative(self):
        from scripts.eval import _delta

        result = _delta(0.80, 0.85)
        assert "-0.0500" in result

    def test_delta_equal(self):
        from scripts.eval import _delta

        result = _delta(0.85, 0.85)
        assert "=" in result

    def test_print_summary(self, capsys):
        from scripts.eval import print_summary
        from src.eval import EvalReport, QuestionResult

        report = EvalReport(suite_name="test")
        report.results.append(
            QuestionResult(
                question="Q?",
                answer="A.",
                groundedness=0.9,
                relevance=0.8,
                refusal_correct=True,
                latency_ms=100.0,
            ),
        )
        report.compute_metrics()

        print_summary(report)
        captured = capsys.readouterr()
        assert "test" in captured.out
        assert "0.9000" in captured.out

    def test_print_summary_with_baseline(self, capsys):
        from scripts.eval import print_summary
        from src.eval import EvalReport, QuestionResult

        report = EvalReport(suite_name="test")
        report.results.append(
            QuestionResult(
                question="Q?",
                answer="A.",
                groundedness=0.9,
                relevance=0.8,
                refusal_correct=True,
                latency_ms=100.0,
            ),
        )
        report.compute_metrics()

        baseline = {
            "avg_groundedness": 0.85,
            "avg_relevance": 0.75,
            "refusal_accuracy": 1.0,
            "avg_latency_ms": 80.0,
        }
        print_summary(report, baseline=baseline)
        captured = capsys.readouterr()
        assert "baseline" in captured.out
        assert "+" in captured.out  # positive deltas

    def test_save_report_json(self, tmp_path):
        from scripts.eval import save_report
        from src.eval import EvalReport

        report = EvalReport(suite_name="save_test")
        report.compute_metrics()

        save_report(report, str(tmp_path), "json")
        assert (tmp_path / "save_test.json").exists()
        assert not (tmp_path / "save_test.html").exists()

    def test_save_report_html(self, tmp_path):
        from scripts.eval import save_report
        from src.eval import EvalReport

        report = EvalReport(suite_name="save_test")
        report.compute_metrics()

        save_report(report, str(tmp_path), "html")
        assert (tmp_path / "save_test.html").exists()
        assert not (tmp_path / "save_test.json").exists()

    def test_save_report_text(self, tmp_path):
        from scripts.eval import save_report
        from src.eval import EvalReport, QuestionResult

        report = EvalReport(suite_name="save_test")
        report.results.append(
            QuestionResult(question="Q?", answer="A."),
        )
        report.compute_metrics()

        save_report(report, str(tmp_path), "text")
        txt_path = tmp_path / "save_test.txt"
        assert txt_path.exists()
        content = txt_path.read_text(encoding="utf-8")
        assert "save_test" in content
        assert "Q?" in content

    def test_save_report_all(self, tmp_path):
        from scripts.eval import save_report
        from src.eval import EvalReport

        report = EvalReport(suite_name="all_test")
        report.compute_metrics()

        save_report(report, str(tmp_path), "all")
        assert (tmp_path / "all_test.json").exists()
        assert (tmp_path / "all_test.html").exists()
        assert (tmp_path / "all_test.txt").exists()


# ---------------------------------------------------------------------------
# Query script tests
# ---------------------------------------------------------------------------


class TestQueryScript:
    """Tests for scripts/query.py."""

    def test_import(self):
        from scripts.query import main, create_parser
        assert callable(main)
        assert callable(create_parser)

    def test_parser_defaults(self):
        from scripts.query import create_parser

        parser = create_parser()
        args = parser.parse_args([])

        assert args.question is None
        assert args.interactive is False
        assert args.debug is False
        assert args.format == "text"
        assert args.config == "configs/default.yaml"
        assert args.verbose is False

    def test_parser_question(self):
        from scripts.query import create_parser

        parser = create_parser()
        args = parser.parse_args(["--question", "What is federalism?"])

        assert args.question == "What is federalism?"

    def test_parser_short_flags(self):
        from scripts.query import create_parser

        parser = create_parser()
        args = parser.parse_args(["-q", "Q?", "-i"])

        assert args.question == "Q?"
        assert args.interactive is True

    def test_parser_debug_and_format(self):
        from scripts.query import create_parser

        parser = create_parser()
        args = parser.parse_args(["-q", "Q?", "--debug", "--format", "json"])

        assert args.debug is True
        assert args.format == "json"

    def test_no_question_no_interactive_returns_1(self):
        from scripts.query import main

        rc = main([])
        assert rc == 1

    def test_run_question_text(self, capsys):
        from scripts.query import run_question

        graph = MagicMock()
        graph.invoke.return_value = {
            "question": "Q?",
            "answer": "A.",
            "confidence": 0.9,
            "citations": [],
            "refusal_reason": None,
        }

        state = run_question("Q?", graph, debug=False, fmt="text")
        captured = capsys.readouterr()
        assert "A." in captured.out
        assert "0.90" in captured.out
        assert state["answer"] == "A."

    def test_run_question_json(self, capsys):
        from scripts.query import run_question

        graph = MagicMock()
        graph.invoke.return_value = {
            "question": "Q?",
            "answer": "A.",
            "confidence": 0.9,
            "citations": [],
            "refusal_reason": None,
        }

        run_question("Q?", graph, debug=False, fmt="json")
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["question"] == "Q?"
        assert data["answer"] == "A."
        assert data["confidence"] == 0.9

    def test_run_question_refusal(self, capsys):
        from scripts.query import run_question

        graph = MagicMock()
        graph.invoke.return_value = {
            "question": "Q?",
            "answer": None,
            "confidence": 0.0,
            "citations": [],
            "refusal_reason": "Not in corpus",
        }

        run_question("Q?", graph, debug=False, fmt="text")
        captured = capsys.readouterr()
        assert "REFUSED" in captured.out
        assert "Not in corpus" in captured.out

    def test_run_question_error(self, capsys):
        from scripts.query import run_question

        graph = MagicMock()
        graph.invoke.side_effect = RuntimeError("boom")

        state = run_question("Q?", graph, debug=False, fmt="text")
        captured = capsys.readouterr()
        assert "boom" in captured.err
        assert state.get("error") == "boom"

    def test_run_question_debug(self, capsys):
        from scripts.query import run_question

        graph = MagicMock()
        graph.invoke.return_value = {
            "question": "Q?",
            "answer": "A.",
            "query_type": "factual",
            "confidence": 0.9,
            "is_grounded": True,
            "retry_count": 0,
            "action": "accept",
            "citations": [],
            "refusal_reason": None,
        }

        run_question("Q?", graph, debug=True, fmt="text")
        captured = capsys.readouterr()
        assert "debug" in captured.out.lower()
        assert "query_type: factual" in captured.out
        assert "is_grounded: True" in captured.out

    def test_run_question_with_citations(self, capsys):
        from scripts.query import run_question
        from src.types import Citation

        cit = Citation(source="doc.txt", chunk_id="c-1", text="text", score=0.9)
        graph = MagicMock()
        graph.invoke.return_value = {
            "question": "Q?",
            "answer": "A.",
            "confidence": 0.9,
            "citations": [cit],
            "refusal_reason": None,
        }

        run_question("Q?", graph, debug=False, fmt="text")
        captured = capsys.readouterr()
        assert "doc.txt" in captured.out
        assert "Citations" in captured.out

    def test_print_debug(self, capsys):
        from scripts.query import print_debug

        state = {
            "query_type": "factual",
            "confidence": 0.85,
            "is_grounded": True,
            "retry_count": 1,
            "action": "accept",
            "chunks": [1, 2, 3, 4, 5],  # long list
        }
        print_debug(state)
        captured = capsys.readouterr()
        assert "query_type: factual" in captured.out
        assert "5 items" in captured.out

    def test_interactive_loop_exit(self, monkeypatch, capsys):
        from scripts.query import interactive_loop

        inputs = iter(["exit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        graph = MagicMock()
        interactive_loop(graph, debug=False, fmt="text")

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out
        graph.invoke.assert_not_called()

    def test_interactive_loop_question_then_quit(self, monkeypatch, capsys):
        from scripts.query import interactive_loop

        inputs = iter(["What is X?", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        graph = MagicMock()
        graph.invoke.return_value = {
            "question": "What is X?",
            "answer": "X is Y.",
            "confidence": 0.8,
            "citations": [],
            "refusal_reason": None,
        }

        interactive_loop(graph, debug=False, fmt="text")

        captured = capsys.readouterr()
        assert "X is Y." in captured.out
        graph.invoke.assert_called_once()

    def test_interactive_loop_empty_input_skipped(self, monkeypatch, capsys):
        from scripts.query import interactive_loop

        inputs = iter(["", "  ", "exit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        graph = MagicMock()
        interactive_loop(graph, debug=False, fmt="text")
        graph.invoke.assert_not_called()

    def test_interactive_loop_eof(self, monkeypatch, capsys):
        from scripts.query import interactive_loop

        monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(EOFError))

        graph = MagicMock()
        interactive_loop(graph, debug=False, fmt="text")

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out


# ---------------------------------------------------------------------------
# Integration-style: scripts can be parsed without external services
# ---------------------------------------------------------------------------


class TestScriptEntryPoints:
    """Verify that all three scripts have a working main() with --help."""

    def test_ingest_help(self):
        from scripts.ingest import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_eval_help(self):
        from scripts.eval import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_query_help(self):
        from scripts.query import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Verify key functions are importable."""

    def test_ingest_exports(self):
        from scripts.ingest import main, create_parser, print_progress, print_summary
        assert all(callable(f) for f in [main, create_parser, print_progress, print_summary])

    def test_eval_exports(self):
        from scripts.eval import (
            main, create_parser, resolve_dataset_path,
            load_baseline, print_summary, save_report,
        )
        assert all(callable(f) for f in [
            main, create_parser, resolve_dataset_path,
            load_baseline, print_summary, save_report,
        ])

    def test_query_exports(self):
        from scripts.query import (
            main, create_parser, run_question,
            interactive_loop, print_debug,
        )
        assert all(callable(f) for f in [
            main, create_parser, run_question,
            interactive_loop, print_debug,
        ])
