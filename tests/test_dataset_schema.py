import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import plotly.express as px


def dataset_functions():
	module = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
	function_names = {"load_dataset", "clean_text", "metadata_value", "article_length_series", "article_document", "report_context", "render_dashboard", "ingest_dataset", "get_indexed_collection"}
	constant_names = {"REQUIRED_COLUMNS", "CONTENT_SCHEMA_VERSION", "COLLECTION_NAME", "EMBED_BATCH_SIZE", "CHAT_RESULT_COUNT"}
	nodes = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in function_names
		or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constant_names for target in node.targets)]
	namespace = {"pd": pd, "px": px}
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

	def test_documents_and_report_use_all_context(self):
		document = self.functions["article_document"](self.row)
		context = self.functions["report_context"](self.frame)
		for column in self.functions["REQUIRED_COLUMNS"]:
			self.assertIn(f"{column}:", document)
			self.assertIn(f"{column}:", context)
		self.assertIn("Average word count: 1250.0", context)
		self.assertNotIn("nan", self.functions["article_document"]({"Title": "Example", "Solution": float("nan")}))

	def test_dashboard_dimensions(self):
		figures = []
		dimensions = ["Topics", "Post type", "Vertical", "Buyer persona", "Strategic initiative", "Features Mentioned", "Target audience"]
		for dimension in dimensions:
			with self.subTest(dimension=dimension):
				figures.clear()
				self.functions["st"] = SimpleNamespace(
					columns=lambda count: [SimpleNamespace(metric=lambda *args: None) for index in range(count)],
					selectbox=lambda label, choices, **kwargs: dimension if label == "Analyze by" else choices[0],
					plotly_chart=lambda figure, **kwargs: figures.append(figure), divider=lambda: None, info=lambda text: None,
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
		)
		stages = ["TOFU", " top of funnel ", "MOFU", "Middle  of\nFunnel", "BOFU", "Bottom of funnel", "Other", "", None]
		frame = pd.DataFrame([self.row | {"Funnel stage": stage} for stage in stages])
		self.functions["render_dashboard"](frame)
		funnel = figures[0]
		self.assertEqual([trace.name for trace in funnel.data], ["TOFU", "MOFU", "BOFU"])
		for trace in funnel.data:
			self.assertAlmostEqual(trace.y[0], 100 / 3)
			self.assertEqual(trace.customdata[0][0], 2)

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
		self.assertEqual(self.functions["CHAT_RESULT_COUNT"], 8)

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