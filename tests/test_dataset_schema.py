import ast
from contextlib import nullcontext
import gzip
import hashlib
import json
import os
import threading
import time
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import plotly.express as px
import re


def dataset_functions():
	module = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
	function_names = {"load_dataset", "clean_text", "metadata_value", "article_length_series", "article_document", "split_category_cell", "normalize_funnel_stage", "explode_category_values", "report_context", "topic_frequency_context", "csv_source_fingerprints", "new_editorial_metrics_bucket", "build_editorial_metrics_cache", "write_editorial_metrics_cache", "load_or_build_editorial_metrics_cache", "save_editorial_metrics_cache", "editorial_metrics_by_site", "editorial_metrics_context", "is_feature_search_query", "prompt_month_window", "filter_frame_for_prompt_month_window", "build_editorial_metric_charts", "build_editorial_category_chart", "build_dashboard_topic_chart", "build_dashboard_charts", "render_dashboard", "render_analytics", "format_elapsed_duration", "run_operation_with_progress", "ingest_dataset", "get_indexed_collection"}
	constant_names = {"REQUIRED_COLUMNS", "CONTENT_SCHEMA_VERSION", "COLLECTION_NAME", "EMBED_BATCH_SIZE", "CHAT_RESULT_COUNT", "EDITORIAL_METRICS_SCHEMA_VERSION", "EDITORIAL_METRIC_FIELDS", "EDITORIAL_MULTI_VALUE_FIELDS"}
	nodes = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in function_names
		or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constant_names for target in node.targets)]
	namespace = {
		"pd": pd, "px": px, "re": re, "time": time, "hashlib": hashlib, "json": json,
		"gzip": gzip, "os": os, "tempfile": tempfile, "datetime": datetime, "timezone": timezone,
		"ThreadPoolExecutor": ThreadPoolExecutor, "Path": Path,
			"FutureTimeoutError": FutureTimeoutError,
			"EDITORIAL_METRICS_CACHE_PATH": Path("chroma_db/editorial_metrics.json.gz"),
	}
	exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
	return namespace


class DatasetSchemaTests(unittest.TestCase):
	def setUp(self):
		self.functions = dataset_functions()
		self.row = {column: f"Value for {column}" for column in self.functions["REQUIRED_COLUMNS"]}
		self.row.update({"Date published": pd.Timestamp("2026-09-10"), "URL": "https://example.com/article", "Title": "Example article", "Word count": "1,250", "Funnel stage": "MOFU", "Sito": "Example", "_source_file": "Example.csv"})
		self.frame = pd.DataFrame([self.row])

	def test_load_new_schema_and_missing_columns(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "Example.csv"
			self.frame.drop(columns=["Sito", "_source_file"]).to_csv(path, index=False)
			frame, errors = self.functions["load_dataset"]([path])
			self.assertEqual(errors, [])
			self.assertEqual(frame.iloc[0]["Sito"], "Example")
			self.assertIsInstance(frame.iloc[0]["Date published"], pd.Timestamp)
			self.frame.drop(columns=["Solution"]).to_csv(path, index=False)
			frame, errors = self.functions["load_dataset"]([path])
			self.assertTrue(frame.empty)
			self.assertIn("Solution", errors[0])
			self.assertIn("Word count", frame.columns)

	def test_word_counts(self):
		counts = self.functions["article_length_series"](pd.DataFrame({"Word count": ["1,250", "800 words", None, "unknown"]}))
		self.assertEqual(counts.iloc[:2].tolist(), [1250, 800])
		self.assertTrue(counts.iloc[2:].isna().all())

	def test_operation_progress_updates_while_work_is_running(self):
		release_operation = threading.Event()
		statuses = []
		def operation():
			release_operation.wait(timeout=2)
			return "complete"
		def update_status(message):
			statuses.append(message)
			if "running" in message:
				release_operation.set()
		started = time.monotonic()
		result = self.functions["run_operation_with_progress"](
			operation, "Generating embeddings · batch 1/2", update_status, started, poll_interval=0.01,
		)
		self.assertEqual(result, "complete")
		self.assertTrue(any("running" in message and "total" in message for message in statuses))
		self.assertIn("complete in", statuses[-1])

	def test_documents_and_report_use_all_context(self):
		document = self.functions["article_document"](self.row)
		context = self.functions["report_context"](self.frame)
		for column in self.functions["REQUIRED_COLUMNS"]:
			self.assertIn(f"{column}:", document)
			self.assertIn(f"{column}:", context)
		self.assertIn("Average word count: 1250.0", context)
		self.assertNotIn("nan", self.functions["article_document"]({"Title": "Example", "Solution": float("nan")}))

	def test_topic_frequency_context_counts_all_rows_by_website(self):
		frame = pd.DataFrame({
			"Sito": ["Alpha", "Alpha", "Alpha", "Beta", "Beta"],
			"Topics": ["AI", "AI", "SEO", "AI", ""],
		})
		context = self.functions["topic_frequency_context"](frame)
		self.assertIn("Website: Alpha; articles with a populated Topics value: 3", context)
		self.assertIn("- AI: 2", context)
		self.assertIn("Website: Beta; articles with a populated Topics value: 1", context)

	def test_editorial_metrics_use_all_selected_csv_rows(self):
		frame = pd.DataFrame({
			"Sito": ["Alpha", "Alpha", "Alpha", "Beta"],
			"Date published": pd.to_datetime(["2026-01-05", "2026-01-20", "2026-02-03", "2026-01-11"]),
			"Word count": [1000, 3000, 3000, 5000],
			"Topics": ["AI; SEO", "AI", "SEO", "Hosting"],
		})
		prompt = "How does publishing frequency differ across competitors over time, and what changes in editorial priorities and average article length are visible?"
		context = self.functions["editorial_metrics_context"](frame, prompt)
		self.assertIn("Monthly article counts · Alpha: 2026-01=2, 2026-02=1", context)
		self.assertIn("Monthly article counts · Beta: 2026-01=1, 2026-02=0", context)
		self.assertIn("Average article length · Alpha: 2333.3 words", context)
		self.assertIn("Top editorial topics · Alpha · 2026: AI (2), SEO (2)", context)
		self.assertEqual(self.functions["editorial_metrics_context"](frame, "Which features were introduced?"), "")
		self.assertTrue(self.functions["is_feature_search_query"]("Which hosting features were introduced?"))
		cache = self.functions["build_editorial_metrics_cache"](frame, {})
		figures, summary = self.functions["build_editorial_metric_charts"](
			cache, ["Alpha"], "All data", pd.Timestamp("2026-02-28"),
		)
		self.assertEqual(summary["average_word_count"], 7000 / 3)
		self.assertEqual(list(figures["Monthly Publishing Frequency"].data[0].y), [2, 1])
		self.assertEqual(list(figures["Average Article Length by Month"].data[0].y), [2000, 3000])

	def test_prompt_date_window_filters_older_articles(self):
		frame = pd.DataFrame({
			"Date published": pd.to_datetime(["2026-06-30", "2026-07-01", "2026-08-15", "2026-10-01"]),
			"Title": ["June", "July", "August", "October"],
		})
		self.assertEqual(self.functions["prompt_month_window"]("features introduced in the last three months"), 3)
		self.assertEqual(self.functions["prompt_month_window"]("feature introdotte negli ultimi tre mesi"), 3)
		filtered = self.functions["filter_frame_for_prompt_month_window"](
			frame, "features introduced in the last three months", pd.Timestamp("2026-09-30"),
		)
		self.assertEqual(filtered["Title"].tolist(), ["July", "August"])

	def test_metrics_sidecar_is_cached_and_invalidated_by_csv_fingerprint(self):
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			source = root / "Example.csv"
			source.write_text("version one", encoding="utf-8")
			cache_path = root / "chroma_db" / "editorial_metrics.json.gz"
			first = self.functions["load_or_build_editorial_metrics_cache"](self.frame, [source], cache_path)
			second = self.functions["load_or_build_editorial_metrics_cache"](self.frame, [source], cache_path)
			self.assertEqual(first["generated_at"], second["generated_at"])
			self.assertTrue(cache_path.read_bytes().startswith(b"\x1f\x8b"))
			source.write_text("version two", encoding="utf-8")
			updated_frame = pd.concat([self.frame, self.frame], ignore_index=True)
			updated = self.functions["load_or_build_editorial_metrics_cache"](updated_frame, [source], cache_path)
			self.assertNotEqual(first["source_fingerprints"], updated["source_fingerprints"])
			self.assertEqual(updated["row_count"], 2)
			summary = self.functions["editorial_metrics_by_site"](
				updated, ["Example"], "All data", pd.Timestamp("2026-09-30"),
			)
			self.assertEqual(summary["Example"]["article_count"], 2)

	def test_render_analytics_uses_cached_metrics_and_selected_competitors(self):
		frame = pd.DataFrame({
			"Sito": ["Alpha", "Alpha", "Beta"],
			"Date published": pd.to_datetime(["2026-01-05", "2026-02-03", "2026-01-11"]),
			"Word count": [1000, 3000, 5000],
			"Topics": ["AI", "SEO", "Hosting"],
		})
		cache = self.functions["build_editorial_metrics_cache"](frame, {})
		figures = []
		metrics = []
		self.functions["st"] = SimpleNamespace(
			subheader=lambda *args, **kwargs: None,
			info=lambda *args, **kwargs: None,
			selectbox=lambda label, choices, **kwargs: choices[0],
			container=lambda **kwargs: nullcontext(),
			metric=lambda label, value, **kwargs: metrics.append((label, value)),
			caption=lambda *args, **kwargs: None,
			plotly_chart=lambda figure, **kwargs: figures.append(figure),
		)
		self.functions["render_analytics"](frame, cache, "All data", pd.Timestamp("2026-02-28"))
		self.assertEqual(metrics[0], ("Articles in scope", "3"))
		self.assertEqual(metrics[1], ("Competitors in scope", 2))
		self.assertEqual(len(figures), 3)
		self.assertEqual(figures[0].layout.title.text, "Monthly Publishing Frequency")
		self.assertEqual(figures[1].layout.title.text, "Average Article Length by Month")

	def test_semicolon_separated_categories_are_counted_individually(self):
		frame = pd.DataFrame({
			"Sito": ["Alpha", "Alpha", "Alpha"],
			"Topics": ["AI; SEO", "AI", "SEO; SEO"],
		})
		figure = self.functions["build_dashboard_topic_chart"](frame, "Topics")
		counts = dict(zip(figure.data[0].y, figure.data[0].x))
		self.assertEqual(counts, {"AI": 2, "SEO": 2})

	def test_dashboard_dimensions(self):
		figures = []
		dimensions = ["Topics", "Post type", "Vertical", "Buyer persona", "Strategic initiative", "Features Mentioned", "Target audience"]
		for dimension in dimensions:
			with self.subTest(dimension=dimension):
				figures.clear()
				self.functions["st"] = SimpleNamespace(
					columns=lambda count: [SimpleNamespace(metric=lambda *args: None) for index in range(count)],
					selectbox=lambda label, choices, **kwargs: dimension if label == "Analyze by" else choices[0],
					plotly_chart=lambda figure, **kwargs: figures.append(figure), divider=lambda: None,
					info=lambda text: None, subheader=lambda text: None, metric=lambda *args: None,
				)
				self.functions["render_dashboard"](self.frame)
				self.assertEqual(len(figures), 4)
				self.assertEqual(figures[2].layout.yaxis.title.text, "Word count")

	def test_funnel_normalizes_aliases_and_excludes_other_values(self):
		figures = []
		self.functions["st"] = SimpleNamespace(
			columns=lambda count: [SimpleNamespace(metric=lambda *args: None) for index in range(count)],
			selectbox=lambda label, choices, **kwargs: choices[0],
			plotly_chart=lambda figure, **kwargs: figures.append(figure), divider=lambda: None, info=lambda text: None,
			subheader=lambda text: None, metric=lambda *args: None,
		)
		stages = ["TOFU", " top of funnel ", "MOFU", "Middle  of\nFunnel", "BOFU", "Bottom of funnel", "Other", "", None]
		frame = pd.DataFrame([self.row | {"Funnel stage": stage} for stage in stages])
		self.functions["render_dashboard"](frame)
		funnel = figures[0]
		self.assertEqual([trace.name for trace in funnel.data], ["TOFU", "MOFU", "BOFU"])
		for trace in funnel.data:
			self.assertAlmostEqual(trace.y[0], 100 / 3)
			self.assertEqual(trace.customdata[0][0], 2)

	def test_funnel_aliases_and_combined_stages_normalize_to_canonical_values(self):
		normalize = self.functions["normalize_funnel_stage"]
		self.assertEqual(normalize("Top of Funnel (TOFU)"), ["TOFU"])
		self.assertEqual(normalize("Awareness / Consideration"), ["TOFU", "MOFU"])
		self.assertEqual(normalize("Consideration / Decision"), ["MOFU", "BOFU"])
		self.assertEqual(normalize("Awareness; Consideration; Decision"), ["TOFU", "MOFU", "BOFU"])
		self.assertEqual(normalize("Other"), [])

	def test_ingestion_retains_context_metadata_and_version(self):
		collection = Mock()
		collection.count.return_value = 1
		client = Mock()
		client.create_collection.return_value = collection
		self.functions["get_chroma_client"] = lambda: client
		self.functions["embed_texts"] = Mock(return_value=[[1.0, 0.0]])
		self.assertEqual(self.functions["ingest_dataset"](self.frame, object(), lambda value: None), 1)
		arguments = collection.add.call_args.kwargs
		self.assertEqual(arguments["ids"], ["Example.csv::0"])
		self.assertIn("Main Pain Point: Value for Main Pain Point", arguments["documents"][0])
		self.assertEqual(arguments["metadatas"][0]["Buyer persona"], "Value for Buyer persona")
		self.assertEqual(client.create_collection.call_args.kwargs["metadata"]["content_schema_version"], 2)
		self.assertEqual(self.functions["CHAT_RESULT_COUNT"], 12)

	def test_old_index_requires_reingestion(self):
		collection = Mock()
		collection.count.return_value = 1
		client = Mock()
		client.get_collection.return_value = collection
		self.functions["get_chroma_client"] = lambda: client
		for metadata in (None, {"hnsw:space": "cosine"}, {"content_schema_version": 1}):
			collection.metadata = metadata
			self.assertIsNone(self.functions["get_indexed_collection"]())
		collection.metadata = {"content_schema_version": 2}
		self.assertIs(self.functions["get_indexed_collection"](), collection)


if __name__ == "__main__":
	unittest.main()