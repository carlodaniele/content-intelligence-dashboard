import ast
import shutil
import subprocess
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import plotly.io as pio
from google.genai import types as google_genai_types
from streamlit.testing.v1 import AppTest


class SavedReportsTests(unittest.TestCase):
	def setUp(self):
		self.module = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
		names = {"export_scope_details", "render_export_controls", "create_report_item", "report_export", "render_report", "render_report_actions", "queue_saved_report_operation", "queue_saved_chat_operation", "queue_saved_item_operation", "conversation_export", "clean_text", "render_conversation_actions", "remove_chat_exchange", "render_custom_report", "custom_report_content", "remove_custom_report_question", "split_category_cell", "normalize_funnel_stage", "explode_category_values", "new_editorial_metrics_bucket", "build_editorial_metrics_cache", "editorial_metrics_by_site", "editorial_metrics_context", "is_csv_metrics_only_question", "is_feature_search_query", "prompt_month_window", "filter_frame_for_prompt_month_window", "build_dashboard_topic_chart", "build_dashboard_charts", "render_report_visuals", "article_length_series", "is_current_relevant_result", "select_article_context_fields", "article_context", "generate_complete_response", "response_reached_output_limit", "generate_chat_answer", "queue_chat_rerun", "render_chat"}
		self.nodes = [node for node in self.module.body if isinstance(node, ast.FunctionDef) and node.name in names]
		self.script = """import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.io as pio
from google.genai import types
from types import SimpleNamespace
from datetime import datetime, timezone
from uuid import uuid4
from html import escape
GEMINI_API_KEY = 'mock'
CHAT_MODEL = 'mock'
MAX_GENERATION_OUTPUT_TOKENS = 8192
MAX_GENERATION_CONTINUATIONS = 2
EDITORIAL_METRICS_SCHEMA_VERSION = 1
EDITORIAL_METRIC_FIELDS = ('Topics','Vertical','Funnel stage','Buyer persona','Sales pitch','Strategic initiative','Features Mentioned','Main Pain Point','Solution','Target audience','Post type')
EDITORIAL_MULTI_VALUE_FIELDS = {'Topics','Vertical','Funnel stage','Buyer persona','Strategic initiative','Features Mentioned','Target audience'}
analysis_time_range = 'Last year'
period_end_date = pd.NaT
frame = pd.DataFrame({'_source_file': ['Kinsta.csv']})
editorial_metrics_cache = {}
st.session_state.setdefault('saved_reports_ready', True)
st.session_state.setdefault('saved_reports', [])
st.session_state.setdefault('saved_chats', [{'id': 'existing-chat'}])
def report_context(frame): return 'Mock context'
def generate_complete_response(client, prompt): return ('## Report body', False)
def clean_text(value): return '' if value is None else str(value)
def render_sources(sources, key_prefix='sources'): pass
def export_scope_details(item): return 'Scope details'
def article_document(row): return row['Document']
def topic_frequency_context(frame): return 'Complete dataset counts'
def render_suggested_questions(): pass
def render_sources(sources, key_prefix='sources'): pass
def render_conversation_actions(item): pass
def get_genai_client(key):
    return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: SimpleNamespace(text='## Report body')))
"""
		self.script += ast.unparse(ast.Module(body=self.nodes, type_ignores=[]))

	def test_report_save_and_identity(self):
		app = AppTest.from_string(self.script + "\nrender_report(frame)\n").run()
		next(button for button in app.button if button.label == "Generate Content Strategy Report").click().run()
		self.assertFalse(app.exception)
		report = app.session_state["strategy_report"]
		self.assertEqual(report["scope"], {"files": ["Kinsta.csv"], "time_range": "Last year", "articles": 1})
		app.run()
		self.assertEqual(app.session_state["strategy_report"]["id"], report["id"])
		app.button(key=f"report_save_{report['id']}").click().run()
		self.assertEqual(app.session_state["_saved_report_operation"]["item"], report)
		self.assertEqual(app.session_state["saved_chats"], [{"id": "existing-chat"}])
		app.session_state["_saved_report_operation"] = None
		app.session_state["saved_reports"] = [report]
		app.run()
		self.assertTrue(app.button(key=f"report_save_{report['id']}").disabled)
		next(button for button in app.button if button.label == "Generate Content Strategy Report").click().run()
		self.assertFalse(app.exception)
		self.assertNotEqual(app.session_state["strategy_report"]["id"], report["id"])
		self.assertEqual(app.session_state["saved_reports"], [report])

	def test_legacy_report_and_exports(self):
		app = AppTest.from_string(self.script + "\nst.session_state.setdefault('strategy_report', 'Legacy report')\nrender_report(frame)\n").run()
		self.assertFalse(app.exception)
		self.assertEqual(app.session_state["strategy_report"]["content"], "Legacy report")
		namespace = {}
		exec("from html import escape\nimport pandas as pd\n", namespace)
		exec(compile(ast.Module(body=self.nodes, type_ignores=[]), "app.py", "exec"), namespace)
		item = {"id": "report", "title": "Report <script>", "content": "## Analysis\n<script>alert(1)</script>", "created_at": "2026-10-09", "scope": {"files": ["Kinsta.csv"], "time_range": "Last year", "articles": 10}}
		for file_format in ("Markdown", "Text", "HTML"):
			with self.subTest(file_format=file_format):
				exported = namespace["report_export"](item, file_format)
				self.assertIn("Kinsta.csv", exported)
				self.assertIn("Articles analyzed: 10", exported)
				self.assertIn("Last year", exported)
				self.assertIn("Analysis", exported)
		html = namespace["report_export"](item, "HTML")
		self.assertNotIn("<script>", html)
		self.assertIn("&lt;script&gt;", html)
		conversation = {"question": "Question", "answer": "Answer", "sources": []}
		self.assertIn("## Question", namespace["conversation_export"](conversation, "Markdown"))

	def test_custom_report_composes_answers_and_exports_charts(self):
		script = self.script.replace(
			"frame = pd.DataFrame({'_source_file': ['Kinsta.csv']})",
			"frame = pd.DataFrame({'_source_file': ['Kinsta.csv'], 'Sito': ['Kinsta'], 'Topics': ['AI'], 'Funnel stage': ['TOFU'], 'Word count': ['900'], 'Date published': [pd.Timestamp('2026-01-10')]})",
		)
		script += """
st.session_state.custom_report_answers = [{'id':'question-1','question':'How much AI content?','answer':'Kinsta has 1 AI topic article.','sources':[],'include':True}]
render_custom_report(frame)
"""
		app = AppTest.from_string(script).run()
		self.assertFalse(app.exception)
		next(button for button in app.button if button.label == "Generate custom report").click().run()
		self.assertFalse(app.exception)
		report = app.session_state["custom_strategy_report"]
		self.assertIn("Kinsta has 1 AI topic article.", report["content"])
		self.assertEqual(len(report["charts"]), 4)
		namespace = {"pio": pio, "escape": __import__("html").escape, "export_scope_details": lambda item: "Scope details"}
		exec(compile(ast.Module(body=self.nodes, type_ignores=[]), "app.py", "exec"), namespace)
		html = namespace["report_export"](report, "HTML")
		self.assertIn("cdn.plot.ly", html)
		self.assertIn("Funnel Stage Distribution by Website", report["content"])

	def test_custom_report_question_form_uses_report_answer_mode(self):
		script = self.script.replace(
			"frame = pd.DataFrame({'_source_file': ['Kinsta.csv']})",
			"frame = pd.DataFrame({'_source_file': ['Kinsta.csv'], 'Sito': ['Kinsta'], 'Topics': ['AI'], 'Funnel stage': ['TOFU'], 'Word count': ['900'], 'Date published': [pd.Timestamp('2026-01-10')]})",
		)
		script += """
calls = []
get_indexed_collection = lambda: object()
generate_chat_answer = lambda question, collection, **kwargs: (f"report_mode={kwargs['report_mode']}; articles={len(kwargs['frame'])}", [])
render_custom_report(frame)
"""
		app = AppTest.from_string(script).run()
		app.text_area(key="custom_report_question_input").set_value("Count AI topics by website")
		next(button for button in app.button if button.label == "Generate answer").click().run()
		self.assertFalse(app.exception)
		self.assertTrue(
			app.session_state["custom_report_answers"],
			f"warnings={[item.value for item in app.warning]}; errors={[item.value for item in app.error]}",
		)
		self.assertEqual(
			app.session_state["custom_report_answers"][0]["question"], "Count AI topics by website",
			f"warnings={[item.value for item in app.warning]}; errors={[item.value for item in app.error]}",
		)
		self.assertEqual(app.session_state["custom_report_answers"][0]["answer"], "report_mode=True; articles=1")

	def test_chat_answers_csv_metrics_when_search_index_is_missing(self):
		script = self.script.replace(
			"frame = pd.DataFrame({'_source_file': ['Kinsta.csv']})",
			"frame = pd.DataFrame({'_source_file': ['Alpha.csv','Alpha.csv','Beta.csv'], 'Sito':['Alpha','Alpha','Beta'], 'Date published':pd.to_datetime(['2026-01-05','2026-01-20','2026-02-03']), 'Word count':[1000,3000,5000], 'Topics':['AI; SEO','AI','Hosting']})\ndataset = frame\nselected_files = [__import__('pathlib').Path('Alpha.csv'), __import__('pathlib').Path('Beta.csv')]\nanalysis_time_range = 'All data'",
		)
		script += """
st.session_state.metric_call = None
editorial_metrics_cache = build_editorial_metrics_cache(dataset, {})
get_indexed_collection = lambda: None
def generate_chat_answer(question, collection, **kwargs):
    st.session_state.metric_call = {'question': question, 'collection_is_none': collection is None, 'context': kwargs['metrics_context']}
    return 'Computed metrics [Dataset metrics].', []
render_chat()
"""
		app = AppTest.from_string(script).run()
		prompt = "How does publishing frequency differ across competitors over time, and what changes in editorial priorities are visible?"
		app.chat_input[0].set_value(prompt).run()
		self.assertFalse(app.exception)
		self.assertIsNotNone(app.session_state["metric_call"])
		self.assertTrue(app.session_state["metric_call"]["collection_is_none"])
		self.assertIn("Monthly article counts · Beta: 2026-01=0, 2026-02=1", app.session_state["metric_call"]["context"])
		self.assertIn("Top editorial topics · Alpha · 2026: AI (2), SEO (1)", app.session_state["metric_call"]["context"])
		self.assertFalse(any("search index" in item.value.lower() for item in app.warning))

	def test_generation_continues_on_max_tokens_and_flags_exhausted_retries(self):
		names = {"generate_complete_response", "response_reached_output_limit"}
		nodes = [node for node in self.module.body if isinstance(node, ast.FunctionDef) and node.name in names]
		namespace = {
			"types": google_genai_types, "CHAT_MODEL": "mock",
			"MAX_GENERATION_OUTPUT_TOKENS": 8192, "MAX_GENERATION_CONTINUATIONS": 2,
		}
		exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
		responses = [
			SimpleNamespace(text="Partial first section.", candidates=[SimpleNamespace(finish_reason=google_genai_types.FinishReason.MAX_TOKENS)]),
			SimpleNamespace(text="Completed second section.", candidates=[SimpleNamespace(finish_reason=google_genai_types.FinishReason.STOP)]),
		]
		calls = []
		client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: calls.append(kwargs) or responses.pop(0)))
		answer, incomplete = namespace["generate_complete_response"](client, "Original question and evidence")
		self.assertEqual(answer, "Partial first section.\nCompleted second section.")
		self.assertFalse(incomplete)
		self.assertEqual(len(calls), 2)
		self.assertIn("Do not repeat earlier text", calls[1]["contents"])
		self.assertEqual(calls[0]["config"].max_output_tokens, 8192)

		responses = [
			SimpleNamespace(text=f"Part {index}.", candidates=[SimpleNamespace(finish_reason=google_genai_types.FinishReason.MAX_TOKENS)])
			for index in range(3)
		]
		client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: responses.pop(0)))
		answer, incomplete = namespace["generate_complete_response"](client, "Long question")
		self.assertTrue(incomplete)
		self.assertEqual(answer, "Part 0.\nPart 1.\nPart 2.")

	def test_chat_retrieval_rejects_irrelevant_stale_and_uncited_answers(self):
		import pandas as pd
		import re

		def ask(distance, indexed_document, generated_answer, prompt="What does the data say?", report_mode=False, current_document="Current imported CSV article about product features.", reference_end_date=None):
			frame = pd.DataFrame({
				"_source_file": ["Kinsta.csv"], "Sito": ["Kinsta"], "Document": [current_document],
				"Date published": [pd.Timestamp("2026-01-05")], "Word count": [900],
				"Title": ["Imported article"], "Topics": ["AI; SEO"], "Summary": ["Concise grounded summary."],
				"Sales pitch": ["Unused pitch"], "Main Pain Point": ["Unused pain"], "Solution": ["Unused solution"],
			})
			generation_calls = []
			collection = SimpleNamespace(
				count=lambda: 1,
				query=lambda **kwargs: {
					"ids": [["Kinsta.csv::0"]], "documents": [[indexed_document]],
					"metadatas": [[{"Sito": "Kinsta", "Title": "Imported article", "URL": "https://example.com"}]],
					"distances": [[distance]],
				},
			)
			client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: (
				generation_calls.append(kwargs) or SimpleNamespace(text=generated_answer)
			)))
			namespace = {
				"pd": pd, "re": re, "types": google_genai_types, "dataset": frame,
				"GEMINI_API_KEY": "mock", "CHAT_MODEL": "mock", "CHAT_RESULT_COUNT": 12,
				"MAX_GENERATION_OUTPUT_TOKENS": 8192, "MAX_GENERATION_CONTINUATIONS": 2,
				"MAX_RETRIEVAL_COSINE_DISTANCE": 0.65, "EMBEDDING_MODEL": "mock",
				"datetime": datetime, "timezone": timezone,
				"EDITORIAL_METRICS_SCHEMA_VERSION": 3,
				"EDITORIAL_METRIC_FIELDS": ("Topics", "Vertical", "Funnel stage", "Buyer persona", "Strategic initiative", "Features Mentioned", "Target audience", "Post type"),
				"EDITORIAL_MULTI_VALUE_FIELDS": {"Topics", "Vertical", "Funnel stage", "Buyer persona", "Strategic initiative", "Features Mentioned", "Target audience"},
				"clean_text": lambda value: str(value or ""),
				"article_document": lambda row: row["Document"],
				"topic_frequency_context": lambda source_frame: "Complete dataset counts",
				"embed_texts": lambda *args: [[0.0]], "get_genai_client": lambda key: client,
			}
			exec(compile(ast.Module(body=self.nodes, type_ignores=[]), "app.py", "exec"), namespace)
			result = namespace["generate_chat_answer"](
				prompt, collection, frame=frame, report_mode=report_mode, reference_end_date=reference_end_date,
			)
			return result, generation_calls

		with self.subTest("irrelevant hit"):
			result, calls = ask(0.91, "Current imported CSV article about product features.", "Unsupported answer [Source 1]")
			self.assertIsNone(result)
			self.assertEqual(calls, [])
		with self.subTest("stale indexed text"):
			result, calls = ask(0.2, "Outdated article text.", "Unsupported answer [Source 1]")
			self.assertIsNone(result)
			self.assertEqual(calls, [])
		with self.subTest("missing citation"):
			result, calls = ask(0.2, "Current imported CSV article about product features.", "Unsupported answer")
			self.assertEqual(result, ("The selected data does not provide enough verified evidence to answer this question.", []))
			self.assertEqual(len(calls), 1)
		with self.subTest("valid citation"):
			result, calls = ask(
				0.2, "Current imported CSV article about product features.",
				"The source discusses reliability [Source 1].", prompt="Why does the imported article discuss reliability?",
			)
			self.assertEqual(result[0], "The source discusses reliability [Source 1].")
			self.assertEqual(len(result[1]), 1)
			self.assertEqual(len(calls), 1)
			contents = calls[0]["contents"]
			self.assertIn("respond in english or italian", contents.lower())
			self.assertIn("Summary: Concise grounded summary.", contents)
			self.assertNotIn("Sales pitch: Unused pitch", contents)
			self.assertNotIn("Main Pain Point: Unused pain", contents)
			self.assertNotIn("Solution: Unused solution", contents)
		with self.subTest("feature search uses complete indexed document"):
			feature_document = (
				"Title: New Cloudways hosting feature\n"
				"Features Mentioned: Cloudways Velocity\n"
				"Summary: The feature enables isolated staging environments and one-click deployment controls."
			)
			result, calls = ask(
				0.2, feature_document, "Cloudways Velocity adds isolated staging and one-click deployments [Source 1].",
				prompt="Which hosting features did Cloudways introduce in the last three months? Describe them in detail.",
				current_document=feature_document, reference_end_date=pd.Timestamp("2026-03-31"),
			)
			self.assertIn("Cloudways Velocity adds", result[0])
			self.assertEqual(len(result[1]), 1)
			self.assertIn("Summary: The feature enables isolated staging environments", calls[0]["contents"])
			self.assertNotIn("Top Features Mentioned values", calls[0]["contents"])
		with self.subTest("custom report language"):
			result, calls = ask(
				0.2, "Current imported CSV article about product features.",
				"I temi più frequenti sono AI e SEO [Dataset metrics] [Dataset counts].",
				prompt="Qual è la frequenza dei temi?", report_mode=True,
			)
			self.assertEqual(result[0], "I temi più frequenti sono AI e SEO [Dataset metrics] [Dataset counts].")
			self.assertIn("answer in english or italian", calls[0]["contents"].lower())
			self.assertIn("Complete dataset counts", calls[0]["contents"])

	def test_chat_controls_still_save_and_export(self):
		script = self.script + """
st.session_state.setdefault('saved_chats_ready', True)
item = {'id':'new-chat','question':'Question','answer':'Answer','sources':[]}
render_conversation_actions(item)
"""
		app = AppTest.from_string(script).run()
		self.assertFalse(app.exception)
		app.button(key="chat_save_new-chat").click().run()
		self.assertEqual(app.session_state["_saved_chat_operation"]["item"]["question"], "Question")
		self.assertNotIn("_saved_report_operation", app.session_state)
		app.selectbox(key="chat_format_new-chat").set_value("HTML").run()
		self.assertFalse(app.exception)
		self.assertEqual(app.selectbox(key="chat_format_new-chat").value, "HTML")

	def test_rerun_replaces_answer_for_original_question(self):
		script = self.script.replace(
			"frame = pd.DataFrame({'_source_file': ['Kinsta.csv']})",
			"frame = pd.DataFrame({'_source_file': ['Kinsta.csv']})\ndataset = pd.DataFrame([{'Sito': 'Kinsta'}])\nselected_files = []\nanalysis_time_range = 'All data'\nst.session_state.messages = [{'role':'user','content':'Original question','id':'question-1'}, {'role':'assistant','content':'Old answer','sources':[], 'id':'answer-1'}]\nst.session_state.rerun_calls = []",
		)
		script += "\nget_indexed_collection = lambda: object()\ngenerate_chat_answer = lambda question, collection, **kwargs: (st.session_state.rerun_calls.append(question) or 'Updated answer', [])\nrender_chat()\n"
		app = AppTest.from_string(script).run()
		self.assertFalse(app.exception)
		app.button(key="rerun_chat_question-1").click().run()
		self.assertFalse(app.exception)
		self.assertEqual(
			app.session_state["rerun_calls"], ["Original question"],
			f"warnings={[item.value for item in app.warning]}; errors={[item.value for item in app.error]}; pending={app.session_state.get('_chat_rerun_request')}",
		)
		self.assertEqual(len(app.session_state["messages"]), 2)
		self.assertEqual(app.session_state["messages"][1]["content"], "Updated answer")

	def test_article_context_uses_summary_by_default_and_only_requested_fields(self):
		import pandas as pd
		import re
		names = {"clean_text", "select_article_context_fields", "article_context"}
		nodes = [node for node in self.module.body if isinstance(node, ast.FunctionDef) and node.name in names]
		namespace = {"pd": pd, "re": re}
		exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
		row = {
			"Title": "Example article", "Topics": "AI; SEO", "Summary": "Concise grounded summary.",
			"Sales pitch": "Unused pitch", "Main Pain Point": "Specific challenge",
			"Solution": "Specific solution",
		}
		default_context = namespace["article_context"](row, "What is happening across the sites?")
		self.assertIn("Summary: Concise grounded summary.", default_context)
		self.assertNotIn("Unused pitch", default_context)
		specific_context = namespace["article_context"](row, "What pain points and solutions are mentioned?")
		self.assertIn("Main Pain Point: Specific challenge", specific_context)
		self.assertIn("Solution: Specific solution", specific_context)
		self.assertNotIn("Summary:", specific_context)
		self.assertNotIn("Unused pitch", specific_context)

	def test_editorial_question_uses_complete_csv_metrics_without_rag(self):
		import pandas as pd
		import re
		frame = pd.DataFrame({
			"_source_file": ["Alpha.csv", "Alpha.csv", "Alpha.csv", "Beta.csv"],
			"Sito": ["Alpha", "Alpha", "Alpha", "Beta"],
			"Date published": pd.to_datetime(["2026-01-05", "2026-01-20", "2026-02-03", "2026-01-11"]),
			"Word count": [1000, 3000, 3000, 5000],
			"Topics": ["AI; SEO", "AI", "SEO", "Hosting"],
		})
		prompt = "How does publishing frequency differ across competitors over time, and what changes in editorial priorities and average article length are visible?"
		client_calls = []
		client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: (
			client_calls.append(kwargs) or SimpleNamespace(
				text="Alpha published 2 articles in January and 1 in February; its average length is 2333 words. AI and SEO lead the topics [Dataset metrics].",
				candidates=[],
			)
		)))
		namespace = {
			"pd": pd, "re": re, "types": google_genai_types,
			"dataset": frame, "GEMINI_API_KEY": "mock", "CHAT_MODEL": "mock",
			"CHAT_RESULT_COUNT": 12, "MAX_GENERATION_OUTPUT_TOKENS": 8192,
			"MAX_GENERATION_CONTINUATIONS": 2, "MAX_RETRIEVAL_COSINE_DISTANCE": 0.65,
			"datetime": datetime, "timezone": timezone,
			"EDITORIAL_METRICS_SCHEMA_VERSION": 1,
			"EDITORIAL_METRIC_FIELDS": ("Topics", "Vertical", "Funnel stage", "Buyer persona", "Sales pitch", "Strategic initiative", "Features Mentioned", "Main Pain Point", "Solution", "Target audience", "Post type"),
			"EDITORIAL_MULTI_VALUE_FIELDS": {"Topics", "Vertical", "Funnel stage", "Buyer persona", "Strategic initiative", "Features Mentioned", "Target audience"},
			"EMBEDDING_MODEL": "mock", "clean_text": lambda value: str(value or ""),
			"article_document": lambda row: "full indexed document",
			"embed_texts": lambda *args: (_ for _ in ()).throw(AssertionError("RAG embeddings should not run for CSV metrics")),
			"get_genai_client": lambda key: client,
		}
		exec(compile(ast.Module(body=self.nodes, type_ignores=[]), "app.py", "exec"), namespace)
		result = namespace["generate_chat_answer"](prompt, None, frame=frame)
		self.assertEqual(
			result[0],
			"Alpha published 2 articles in January and 1 in February; its average length is 2333 words. AI and SEO lead the topics [Dataset metrics].",
		)
		self.assertEqual(result[1], [])
		contents = client_calls[0]["contents"]
		self.assertIn("Monthly article counts · Alpha: 2026-01=2, 2026-02=1", contents)
		self.assertIn("Monthly article counts · Beta: 2026-01=1, 2026-02=0", contents)
		self.assertIn("Average article length · Alpha: 2333.3 words", contents)
		self.assertIn("Top editorial topics · Alpha · 2026: AI (2), SEO (2)", contents)
		self.assertNotIn("RETRIEVED ARTICLE CONTEXT", contents)
		self.assertEqual(len(client_calls), 1)

	def test_generation_continues_on_max_tokens_and_flags_exhausted_retries(self):
		names = {"generate_complete_response", "response_reached_output_limit"}
		nodes = [node for node in self.module.body if isinstance(node, ast.FunctionDef) and node.name in names]
		namespace = {
			"types": google_genai_types, "CHAT_MODEL": "mock",
			"MAX_GENERATION_OUTPUT_TOKENS": 8192, "MAX_GENERATION_CONTINUATIONS": 2,
		}
		exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), namespace)
		responses = [
			SimpleNamespace(text="Partial first section.", candidates=[SimpleNamespace(finish_reason=google_genai_types.FinishReason.MAX_TOKENS)]),
			SimpleNamespace(text="Completed second section.", candidates=[SimpleNamespace(finish_reason=google_genai_types.FinishReason.STOP)]),
		]
		calls = []
		client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: calls.append(kwargs) or responses.pop(0)))
		answer, incomplete = namespace["generate_complete_response"](client, "Original question and evidence")
		self.assertEqual(answer, "Partial first section.\nCompleted second section.")
		self.assertFalse(incomplete)
		self.assertEqual(len(calls), 2)
		self.assertIn("Do not repeat earlier text", calls[1]["contents"])
		self.assertEqual(calls[0]["config"].max_output_tokens, 8192)

		responses = [
			SimpleNamespace(text=f"Part {index}.", candidates=[SimpleNamespace(finish_reason=google_genai_types.FinishReason.MAX_TOKENS)])
			for index in range(3)
		]
		client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: responses.pop(0)))
		answer, incomplete = namespace["generate_complete_response"](client, "Long question")
		self.assertTrue(incomplete)
		self.assertEqual(answer, "Part 0.\nPart 1.\nPart 2.")

	@unittest.skipUnless(shutil.which("node"), "Node.js is needed to test browser storage JavaScript")
	def test_storage_isolation_and_failures(self):
		assignment = next(node for node in self.module.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "saved_chat_storage" for target in node.targets))
		javascript = next(ast.literal_eval(keyword.value) for keyword in assignment.value.keywords if keyword.arg == "js")
		script = "import assert from 'node:assert/strict';\n" + javascript.replace("export default function", "const mount = function") + """;
const store = new Map();
globalThis.window = {addEventListener() {}, removeEventListener() {}};
globalThis.localStorage = {getItem: key => store.get(key) || null, setItem: (key, value) => store.set(key, value)};
let snapshot;
let updates = 0;
function run(kind, operation = null, acknowledged = null) {
    mount({data: {kind, storage_key: `content-intelligence.saved-${kind}s.v1`, operation, snapshot: acknowledged},
        parentElement: {}, setStateValue: (name, value) => {snapshot = value; updates++;}});
    return snapshot;
}
const chat = {id:'chat',question:'Question',answer:'Answer',sources:[]};
const report = {id:'report',title:'Report',content:'Report body',scope:{files:['Kinsta.csv']}};
run('chat', {id:'chat-save', action:'save', item_id:chat.id, item:chat});
run('report', {id:'report-save', action:'save', item_id:report.id, item:report});
assert.equal(run('report').items[0].content, 'Report body');
const acknowledged = snapshot;
const before = updates;
run('report', null, acknowledged);
assert.equal(updates, before);
assert.equal(run('report', {id:'save-again',action:'save',item_id:report.id,item:report}).items.length, 1);
assert.equal(run('chat').items[0].question, 'Question');
assert.equal(run('report', {id:'delete',action:'delete',item_id:report.id}).items.length, 0);
assert.equal(run('chat').items.length, 1);
store.set('content-intelligence.saved-reports.v1', '{bad JSON');
assert.ok(run('report').error);
assert.equal(store.get('content-intelligence.saved-reports.v1'), '{bad JSON');
store.set('content-intelligence.saved-reports.v1', '[]');
localStorage.setItem = () => {throw new Error('Quota exceeded');};
assert.match(run('report', {id:'quota',action:'save',item_id:report.id,item:report}).error, /Quota/);
assert.equal(store.get('content-intelligence.saved-reports.v1'), '[]');
"""
		result = subprocess.run([shutil.which("node"), "--input-type=module"], input=script, text=True, capture_output=True)
		self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
	unittest.main()