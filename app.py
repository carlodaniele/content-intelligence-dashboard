import os
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

import chromadb
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v2 as components
from streamlit.errors import StreamlitSecretNotFoundError
from google import genai
from google.genai import types
from suggested_questions import load_question_config, normalize_question, save_question_config


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
QUESTION_CONFIG_PATH = Path(os.getenv("SUGGESTED_QUESTIONS_CONFIG", str(ROOT_DIR / "suggested_questions.json"))).expanduser()
COLLECTION_NAME = "website_content"
CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 64
CHAT_RESULT_COUNT = 8
CONTENT_SCHEMA_VERSION = 2
REQUIRED_COLUMNS = [
	"Date published",
	"URL",
	"Title",
	"Post type",
	"Topics",
	"Vertical",
	"Buyer persona",
	"Sales pitch",
	"Strategic initiative",
	"Features Mentioned",
	"Main Pain Point",
	"Solution",
	"Word count",
	"Funnel stage",
	"Target audience",
	"Summary",
]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

saved_chat_storage = components.component(
	"saved_chat_storage",
	js="""
	export default function(component) {
		const { data, parentElement, setStateValue } = component;
		const storageKey = data.storage_key || 'content-intelligence.saved-chats.v1';
		const synchronize = (applyOperation) => {
			let snapshot;
			try {
				let items = JSON.parse(localStorage.getItem(storageKey) || '[]');
				if (!Array.isArray(items) || items.some(item =>
					!item || typeof item.id !== 'string' || (data.kind === 'report'
						? typeof item.title !== 'string' || typeof item.content !== 'string'
						: typeof item.question !== 'string' || typeof item.answer !== 'string' || !Array.isArray(item.sources)))) {
					throw new Error('Saved items have an invalid format.');
				}
				const operation = applyOperation ? data.operation : null;
				if (operation) {
					items = items.filter(item => item.id !== operation.item_id);
					if (operation.action === 'save') items.unshift(operation.item);
					localStorage.setItem(storageKey, JSON.stringify(items));
				}
				snapshot = { items, operation_id: data.operation?.id || null, error: null };
			} catch (error) {
				snapshot = { items: null, operation_id: data.operation?.id || null, error: error.message };
			}
			const serialized = JSON.stringify(snapshot);
			if (parentElement.savedChatSnapshot !== serialized) {
				parentElement.savedChatSnapshot = serialized;
				if (JSON.stringify(data.snapshot) !== serialized) setStateValue('snapshot', snapshot);
			}
		};
		synchronize(true);
		const onStorage = event => {
			if (event.key === storageKey || event.key === null) synchronize(false);
		};
		window.addEventListener('storage', onStorage);
		return () => window.removeEventListener('storage', onStorage);
	}
	""",
)

question_preferences_storage = components.component(
	"question_preferences_storage",
	js=r"""
	export default function(component) {
		const { data, parentElement, setStateValue } = component;
		const storageKey = 'content-intelligence.question-preferences.v1';
		const synchronize = (applyOperation) => {
			let snapshot;
			try {
				let preferences = JSON.parse(localStorage.getItem(storageKey) ||
					'{"version":1,"favorites":[],"hidden_shared_ids":[]}');
				if (!preferences || preferences.version !== 1 || !Array.isArray(preferences.favorites)
					|| !Array.isArray(preferences.hidden_shared_ids)
					|| preferences.hidden_shared_ids.some(id => typeof id !== 'string')
					|| preferences.favorites.some(item => !item || typeof item.id !== 'string'
						|| typeof item.text !== 'string' || !item.text.trim() || item.text.length > 1000)
					|| new Set(preferences.favorites.map(item => item.id)).size !== preferences.favorites.length) {
					throw new Error('Question preferences have an invalid format.');
				}
				const operation = applyOperation ? data.operation : null;
				if (operation) {
					if (operation.action === 'add') {
						if (!preferences.favorites.some(item => item.id === operation.question.id)) {
							if (preferences.favorites.some(item => item.text.toLowerCase() === operation.question.text.toLowerCase())) {
								throw new Error('This favorite question already exists.');
							}
							preferences.favorites.push(operation.question);
						}
					} else if (operation.action === 'delete') {
						preferences.favorites = preferences.favorites.filter(item => item.id !== operation.question.id);
					} else if (operation.action === 'hide') {
						preferences.hidden_shared_ids = [...new Set([...preferences.hidden_shared_ids, operation.question.id])];
					} else if (operation.action === 'restore') {
						preferences.hidden_shared_ids = [];
					}
					localStorage.setItem(storageKey, JSON.stringify(preferences));
				}
				snapshot = { preferences, operation_id: data.operation?.id || null, error: null };
			} catch (error) {
				snapshot = { preferences: null, operation_id: data.operation?.id || null, error: error.message };
			}
			const serialized = JSON.stringify(snapshot);
			if (parentElement.questionPreferencesSnapshot !== serialized) {
				parentElement.questionPreferencesSnapshot = serialized;
				if (JSON.stringify(data.snapshot) !== serialized) setStateValue('snapshot', snapshot);
			}
		};
		synchronize(true);
		const onStorage = event => {
			if (event.key === storageKey || event.key === null) synchronize(false);
		};
		window.addEventListener('storage', onStorage);
		return () => window.removeEventListener('storage', onStorage);
	}
	""",
)

st.set_page_config(page_title="Content Intelligence", page_icon="📚", layout="wide")

try:
	app_active = st.secrets.get("APP_ACTIVE", False)
except StreamlitSecretNotFoundError:
	app_active = False

if app_active is not True:
	st.info("Siamo spiacenti, l'applicazione è momentaneamente offline per manutenzione ordinaria. Riprova più tardi.")
	st.stop()

DATA_DIR.mkdir(parents=True, exist_ok=True)


def scan_csv_files():
	return sorted(
		(path for path in DATA_DIR.iterdir() if path.is_file() and path.suffix.lower() == ".csv"),
		key=lambda path: path.name.casefold(),
	)


def load_dataset(files):
	frames = []
	errors = []

	for path in files:
		try:
			frame = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")
			frame.columns = [str(column).strip() for column in frame.columns]
			missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
			if missing:
				errors.append(f"{path.name}: missing columns: {', '.join(missing)}")
				continue
			frame["Date published"] = pd.to_datetime(frame["Date published"], errors="coerce")
			frame["Sito"] = path.stem
			frame["_source_file"] = path.name
			frames.append(frame)
		except Exception as exc:
			errors.append(f"{path.name}: could not read CSV ({exc})")

	if not frames:
		return pd.DataFrame(columns=REQUIRED_COLUMNS + ["Sito", "_source_file"]), errors
	return pd.concat(frames, ignore_index=True), errors


def clean_text(value):
	if value is None or pd.isna(value):
		return ""
	return str(value).strip()


def metadata_value(value):
	if value is None or pd.isna(value):
		return ""
	if isinstance(value, pd.Timestamp):
		return value.isoformat()
	if hasattr(value, "item"):
		value = value.item()
	if isinstance(value, (bool, int, float, str)):
		return value if not isinstance(value, str) else value.strip()
	return str(value).strip()


def article_length_series(frame):
	values = frame["Word count"].astype("string").str.replace(",", "", regex=False)
	numbers = values.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
	return pd.to_numeric(numbers, errors="coerce")


def article_document(row, max_field_chars=None):
	lines = []
	for column in REQUIRED_COLUMNS:
		value = clean_text(row.get(column))
		if value:
			if max_field_chars is not None:
				value = value[:max_field_chars]
			lines.append(f"{column}: {value}")
	return "\n".join(lines)


@st.cache_resource
def get_chroma_client():
	return chromadb.PersistentClient(path=str(CHROMA_DIR))


@st.cache_resource
def get_genai_client(api_key):
	return genai.Client(api_key=api_key)


def embed_texts(client, texts, task_type):
	embeddings = []
	for start in range(0, len(texts), EMBED_BATCH_SIZE):
		batch = texts[start : start + EMBED_BATCH_SIZE]
		response = client.models.embed_content(
			model=EMBEDDING_MODEL,
			contents=batch,
			config=types.EmbedContentConfig(task_type=task_type),
		)
		batch_embeddings = getattr(response, "embeddings", None) or []
		if len(batch_embeddings) != len(batch):
			raise RuntimeError("Gemini did not return an embedding for every article.")
		embeddings.extend(embedding.values for embedding in batch_embeddings)
	return embeddings


def ingest_dataset(frame, gemini_client, progress_callback):
	if frame.empty:
		raise ValueError("There are no valid rows to index in the detected CSV files.")

	documents = []
	metadatas = []
	ids = []
	for row_number, row in frame.iterrows():
		documents.append(article_document(row))

		metadata = {"Sito": clean_text(row.get("Sito"))}
		for column in frame.columns:
			if column not in {"Summary", "Sito", "_source_file"}:
				metadata[column] = metadata_value(row.get(column))
		metadatas.append(metadata)
		source_file = clean_text(row.get("_source_file"))
		ids.append(f"{source_file}::{row_number}")

	all_embeddings = []
	total_batches = (len(documents) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
	for batch_number, start in enumerate(range(0, len(documents), EMBED_BATCH_SIZE), start=1):
		batch_embeddings = embed_texts(
			gemini_client,
			documents[start : start + EMBED_BATCH_SIZE],
			"RETRIEVAL_DOCUMENT",
		)
		all_embeddings.extend(batch_embeddings)
		progress_callback(batch_number / total_batches)

	chroma_client = get_chroma_client()
	try:
		chroma_client.delete_collection(COLLECTION_NAME)
	except Exception:
		pass
	collection = chroma_client.create_collection(
		name=COLLECTION_NAME,
		metadata={"hnsw:space": "cosine", "content_schema_version": CONTENT_SCHEMA_VERSION},
	)

	for start in range(0, len(documents), EMBED_BATCH_SIZE):
		end = start + EMBED_BATCH_SIZE
		collection.add(
			ids=ids[start:end],
			embeddings=all_embeddings[start:end],
			documents=documents[start:end],
			metadatas=metadatas[start:end],
		)
	return collection.count()


def get_indexed_collection():
	try:
		collection = get_chroma_client().get_collection(COLLECTION_NAME)
		if (collection.metadata or {}).get("content_schema_version") != CONTENT_SCHEMA_VERSION:
			return None
		return collection if collection.count() else None
	except Exception:
		return None


def valid_http_url(value):
	parsed = urlparse(clean_text(value))
	return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def render_sources(sources, key_prefix="source"):
	if not sources:
		return
	with st.expander("Sources Used"):
		for index, source in enumerate(sources, start=1):
			title = clean_text(source.get("Title")) or "Untitled article"
			site = clean_text(source.get("Sito"))
			st.markdown(f"**{index}. {title}** · {site}")
			if valid_http_url(source.get("URL")):
				st.link_button("Open article", clean_text(source["URL"]), key=f"{key_prefix}_{index}_{site}_{title[:24]}")
			else:
				st.caption("URL is unavailable or invalid.")


def report_context(frame):
	lines = [f"Articles analyzed: {len(frame)}", f"Websites analyzed: {frame['Sito'].nunique()}"]
	lengths = article_length_series(frame)
	published = pd.to_datetime(frame["Date published"], errors="coerce", utc=True)

	for site, group in frame.groupby("Sito", sort=True):
		site_lengths = article_length_series(group).dropna()
		site_dates = pd.to_datetime(group["Date published"], errors="coerce", utc=True).dropna()
		lines.append(f"\n## Website: {site}")
		lines.append(f"Articles: {len(group)}")
		if not site_lengths.empty:
			lines.append(f"Average word count: {site_lengths.mean():.1f}; median: {site_lengths.median():.1f}")
		if not site_dates.empty:
			lines.append(f"Publication date range: {site_dates.min().date()} - {site_dates.max().date()}")

		stages = group["Funnel stage"].dropna().astype(str).str.strip().str.upper().value_counts(normalize=True)
		if not stages.empty:
			lines.append("Funnel distribution (%): " + ", ".join(f"{name} {value * 100:.1f}%" for name, value in stages.items()))
		for column in ("Topics", "Post type", "Vertical", "Buyer persona", "Strategic initiative", "Features Mentioned", "Target audience"):
			counts = group[column].dropna().astype(str).str.strip()
			counts = counts[counts.ne("")].value_counts().head(6)
			if not counts.empty:
				lines.append(f"Most frequent {column} values: " + "; ".join(f"{name} ({count})" for name, count in counts.items()))

		dated = group.assign(_report_date=pd.to_datetime(group["Date published"], errors="coerce", utc=True))
		examples = dated.sort_values("_report_date", na_position="first").tail(3)
		for _, article in examples.iterrows():
			lines.append("Article example:\n" + article_document(article, max_field_chars=350))

	if lengths.notna().any():
		lines.append(f"\nOverall average word count: {lengths.mean():.1f}")
	if published.notna().any():
		lines.append(f"Overall publication date range: {published.min().date()} - {published.max().date()}")
	return "\n".join(lines)


def render_dashboard(frame):
	if frame.empty:
		st.info("Add valid CSV files to the `data/` folder to view the dashboard.")
		return

	site_count = frame["Sito"].nunique()
	article_count = len(frame)
	topic_count = frame["Topics"].dropna().nunique()
	metric_columns = st.columns(3)
	metric_columns[0].metric("Websites", site_count)
	metric_columns[1].metric("Articles", article_count)
	metric_columns[2].metric("Unique topics", topic_count)

	funnel = frame[["Sito", "Funnel stage"]].dropna().copy().rename(columns={"Sito": "Website"})
	funnel["Funnel stage"] = (
		funnel["Funnel stage"].astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
		.replace({"TOP OF FUNNEL": "TOFU", "MIDDLE OF FUNNEL": "MOFU", "BOTTOM OF FUNNEL": "BOFU"})
	)
	funnel = funnel[funnel["Funnel stage"].isin(["TOFU", "MOFU", "BOFU"])]
	if not funnel.empty:
		funnel_counts = funnel.groupby(["Website", "Funnel stage"]).size().reset_index(name="Articles")
		funnel_counts["Percentage"] = (
			funnel_counts["Articles"] / funnel_counts.groupby("Website")["Articles"].transform("sum") * 100
		)
		funnel_figure = px.bar(
			funnel_counts,
			x="Website",
			y="Percentage",
			color="Funnel stage",
			category_orders={"Funnel stage": ["TOFU", "MOFU", "BOFU"]},
			barmode="stack",
			color_discrete_sequence=px.colors.qualitative.Vivid,
			title="Funnel Stage Distribution by Website",
			labels={"Percentage": "Articles (%)", "Website": "Website"},
			hover_data={"Articles": True, "Percentage": ":.1f"},
		)
		funnel_figure.update_layout(yaxis_range=[0, 100], yaxis_ticksuffix="%", legend_title_text="Funnel stage")
		st.plotly_chart(funnel_figure, width="stretch")
	else:
		st.info("No funnel stage data is available.")
	st.divider()

	sites = sorted(frame["Sito"].dropna().astype(str).unique(), key=str.casefold)
	dimension = st.selectbox("Analyze by", ["Topics", "Post type", "Vertical", "Buyer persona", "Strategic initiative", "Features Mentioned", "Target audience"], key="topic_dimension")
	selected_site = st.selectbox("Website for most frequent values", ["All websites"] + sites)
	topic_data = frame.copy()
	if selected_site != "All websites":
		topic_data = topic_data[topic_data["Sito"] == selected_site]
	topic_values = topic_data[dimension].dropna().astype(str).str.strip()
	topic_values = topic_values[topic_values.ne("")].value_counts().head(15).sort_values()
	if not topic_values.empty:
		topic_frame = topic_values.rename_axis(dimension).reset_index(name="Articles")
		topic_figure = px.bar(
			topic_frame,
			x="Articles",
			y=dimension,
			orientation="h",
			color="Articles",
			color_continuous_scale="Viridis",
			title=f"Most Frequent {dimension}",
		)
		topic_figure.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Articles")
		st.plotly_chart(topic_figure, width="stretch")
	else:
		st.info(f"No {dimension.lower()} values are available for this selection.")
	st.divider()

	lengths = frame[["Sito"]].copy().rename(columns={"Sito": "Website"})
	lengths["Article length"] = article_length_series(frame)
	lengths = lengths.dropna(subset=["Article length"])
	if not lengths.empty:
		length_figure = px.box(
			lengths,
			x="Website",
			y="Article length",
			color="Website",
			points="outliers",
			color_discrete_sequence=px.colors.qualitative.Prism,
			title="Article Length Comparison",
		)
		length_figure.update_layout(showlegend=False, xaxis_title="Website", yaxis_title="Word count")
		st.plotly_chart(length_figure, width="stretch")
	else:
		st.info("The `Word count` column does not contain usable numeric values.")
	st.divider()

	dated_articles = frame.dropna(subset=["Date published"]).copy()
	if not dated_articles.empty:
		dated_articles["Publication month"] = (
			dated_articles["Date published"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
		)
		monthly_publications = (
			dated_articles.groupby(["Publication month", "Sito"], as_index=False)
			.size()
			.rename(columns={"Sito": "Website", "size": "Articles"})
		)
		publication_figure = px.line(
			monthly_publications,
			x="Publication month",
			y="Articles",
			color="Website",
			markers=True,
			color_discrete_sequence=px.colors.qualitative.Vivid,
			title="Editorial Publishing Velocity",
			labels={"Publication month": "Month ending", "Articles": "Articles published"},
		)
		month_end_ticks = monthly_publications["Publication month"].drop_duplicates().sort_values()
		last_month_end = month_end_ticks.iloc[-1]
		publication_figure.update_xaxes(
			tickmode="array",
			tickvals=month_end_ticks.tolist(),
			ticktext=month_end_ticks.dt.strftime("%b %Y").tolist(),
		)
		publication_figure.update_layout(
			title=f"Editorial Publishing Velocity (through {last_month_end:%B %d, %Y})",
			legend_title_text="Website",
		)
		st.plotly_chart(publication_figure, width="stretch")
	else:
		st.info("No valid publication dates are available for the selected range.")


def queue_saved_chat_operation(action, item):
	queue_saved_item_operation(action, item, "chat")


def queue_saved_report_operation(action, item):
	queue_saved_item_operation(action, item, "report")


def queue_saved_item_operation(action, item, kind):
	st.session_state[f"_saved_{kind}_operation"] = {
		"id": str(uuid4()),
		"action": action,
		"item_id": item["id"],
		"item": item if action == "save" else None,
	}


def render_saved_chats_sidebar():
	render_saved_items_sidebar("chat", "question", show_saved_chat)


def render_saved_reports_sidebar():
	render_saved_items_sidebar("report", "title", show_saved_report)


def render_saved_items_sidebar(kind, title_field, viewer):
	items_key = f"saved_{kind}s"
	ready_key = f"{items_key}_ready"
	operation_key = f"_saved_{kind}_operation"
	snapshot_key = f"_last_saved_{kind}_snapshot"
	notice_key = f"_saved_{kind}_notice"
	open_key = f"open_saved_{kind}"
	noun = "conversation" if kind == "chat" else "report"
	st.session_state.setdefault(items_key, [])
	st.session_state.setdefault(ready_key, False)
	with st.sidebar:
		st.divider()
		st.subheader("Saved conversations" if kind == "chat" else "Saved reports")
		result = saved_chat_storage(
			data={"operation": st.session_state.get(operation_key), "snapshot": st.session_state.get(snapshot_key),
				"kind": kind, "storage_key": f"content-intelligence.saved-{kind}s.v1"},
			default={"snapshot": None},
			key=f"saved_{kind}_browser_storage",
			on_snapshot_change=lambda: None,
		)
		if snapshot := result.snapshot:
			st.session_state[snapshot_key] = snapshot
			st.session_state[ready_key] = not snapshot.get("error")
			if snapshot.get("error"):
				st.warning(f"Browser storage is unavailable: {snapshot['error']}")
			else:
				st.session_state[items_key] = snapshot["items"]
			operation = st.session_state.get(operation_key)
			if operation and snapshot.get("operation_id") == operation["id"]:
				st.session_state.pop(operation_key)
				if snapshot.get("error"):
					st.session_state[notice_key] = {"error": f"Could not update saved {noun}: {snapshot['error']}"}
				else:
					st.session_state[notice_key] = {"success": f"{noun.capitalize()} {'saved' if operation['action'] == 'save' else 'deleted'}."}
				st.rerun()
		if notice := st.session_state.pop(notice_key, None):
			if isinstance(notice, str):
				st.success(notice)
			elif "error" in notice:
				st.error(notice["error"])
			else:
				st.success(notice["success"])
		if not st.session_state[items_key]:
			st.caption(f"No saved {noun}s.")
		for item in st.session_state[items_key]:
			open_column, delete_column = st.columns([5, 1])
			with open_column:
				if st.button(item[title_field], key=f"open_saved_{kind}_{item['id']}", width="stretch", icon=":material/bookmark:"):
					st.session_state[open_key] = item["id"]
			with delete_column:
				st.button(
					"", key=f"delete_saved_{kind}_{item['id']}", icon=":material/delete:",
					help=f"Delete saved {noun}", disabled=not st.session_state[ready_key] or bool(st.session_state.get(operation_key)),
					on_click=queue_saved_item_operation, args=("delete", item, kind),
				)
	if selected_id := st.session_state.pop(open_key, None):
		selected_item = next((item for item in st.session_state[items_key] if item["id"] == selected_id), None)
		if selected_item:
			viewer(selected_item)


def export_scope_details(item):
	scope = item.get("scope", {})
	return (
		f"Created: {item.get('created_at', '')}\n"
		f"Websites: {', '.join(scope.get('files', []))}\n"
		f"Analysis time range: {scope.get('time_range', 'Not recorded')}"
	)


def conversation_export(item, file_format):
	source_lines = []
	for index, source in enumerate(item["sources"], start=1):
		source_lines.append(
			f"[Source {index}] {clean_text(source.get('Title'))} | {clean_text(source.get('Sito'))} | {clean_text(source.get('URL'))}"
		)
	details = export_scope_details(item)
	if file_format == "HTML":
		sources_html = "".join(f"<li>{escape(line)}</li>" for line in source_lines)
		return (
			'<!doctype html><html lang="en"><head><meta charset="utf-8">'
			'<meta name="viewport" content="width=device-width, initial-scale=1">'
			'<title>Content Strategy Conversation</title><style>'
			'body{font-family:Georgia,serif;max-width:900px;margin:40px auto;padding:0 24px;line-height:1.6;}'
			'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;}li{overflow-wrap:anywhere;}'
			'</style></head><body><h1>Content Strategy Conversation</h1>'
			f'<pre>{escape(details)}</pre><h2>Question</h2><pre>{escape(item["question"])}</pre>'
			f'<h2>Answer</h2><pre>{escape(item["answer"])}</pre><h2>Sources</h2><ul>{sources_html}</ul>'
			'</body></html>'
		)
	if file_format == "Markdown":
		return f"# Content Strategy Conversation\n\n{details}\n\n## Question\n\n{item['question']}\n\n## Answer\n\n{item['answer']}\n\n## Sources\n\n" + "\n\n".join(source_lines)
	return f"Content Strategy Conversation\n\n{details}\n\nQuestion\n{item['question']}\n\nAnswer\n{item['answer']}\n\nSources\n" + "\n".join(source_lines)


def remove_chat_exchange(exchange_id):
	messages = st.session_state.get("messages", [])
	for index, message in enumerate(messages):
		if message["role"] == "assistant" and message.get("id") == exchange_id:
			start = index
			while start > 0 and messages[start]["role"] != "user":
				start -= 1
			end = index + 1
			while end < len(messages) and messages[end]["role"] != "user":
				end += 1
			st.session_state.messages = messages[:start] + messages[end:]
			return


def render_conversation_actions(item, allow_save=True, key_prefix="chat"):
	with st.container(horizontal=True):
		if allow_save:
			is_saved = any(saved["id"] == item["id"] for saved in st.session_state.get("saved_chats", []))
			st.button(
				"Saved" if is_saved else "Save", icon=":material/bookmark:", key=f"{key_prefix}_save_{item['id']}",
				disabled=is_saved or not st.session_state.get("saved_chats_ready", False) or bool(st.session_state.get("_saved_chat_operation")),
				on_click=queue_saved_chat_operation, args=("save", item),
			)
			st.button(
				"", icon=":material/delete:", key=f"{key_prefix}_delete_{item['id']}",
				help="Remove question and answer from chat history",
				on_click=remove_chat_exchange, args=(item["id"],),
			)
		render_export_controls(item, conversation_export, key_prefix, "content_strategy")


def render_export_controls(item, export_function, key_prefix, filename_prefix):
	file_format = st.selectbox("Export format", ["Markdown", "Text", "HTML"], key=f"{key_prefix}_format_{item['id']}", width=160, label_visibility="collapsed")
	extension, mime = {"Markdown": ("md", "text/markdown"), "Text": ("txt", "text/plain"), "HTML": ("html", "text/html")}[file_format]
	st.download_button(
		"Download", icon=":material/download:", data=export_function(item, file_format),
		file_name=f"{filename_prefix}_{item['id']}.{extension}", mime=mime,
		key=f"{key_prefix}_download_{item['id']}", on_click="ignore",
	)


def create_report_item(content, frame):
	created = datetime.now(timezone.utc)
	return {
		"id": str(uuid4()),
		"title": f"Content Strategy Report - {created:%Y-%m-%d %H:%M UTC}",
		"content": content,
		"created_at": created.isoformat(),
		"scope": {"files": sorted(frame["_source_file"].dropna().astype(str).unique(), key=str.casefold),
			"time_range": analysis_time_range, "articles": len(frame)},
	}


def report_export(item, file_format):
	details = export_scope_details(item) + f"\nArticles analyzed: {item.get('scope', {}).get('articles', 'Not recorded')}"
	if file_format == "HTML":
		return (
			'<!doctype html><html lang="en"><head><meta charset="utf-8">'
			'<meta name="viewport" content="width=device-width, initial-scale=1">'
			f'<title>{escape(item["title"])}</title><style>'
			'body{font-family:Georgia,serif;max-width:900px;margin:40px auto;padding:0 24px;line-height:1.6;}'
			'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;}'
			f'</style></head><body><h1>{escape(item["title"])}</h1><pre>{escape(details)}</pre>'
			f'<pre>{escape(item["content"])}</pre></body></html>'
		)
	if file_format == "Markdown":
		return f"# {item['title']}\n\n{details}\n\n{item['content']}"
	return f"{item['title']}\n\n{details}\n\n{item['content']}"


def render_report_actions(item, allow_save=True, key_prefix="report"):
	with st.container(horizontal=True):
		if allow_save:
			is_saved = any(saved["id"] == item["id"] for saved in st.session_state.get("saved_reports", []))
			st.button(
				"Saved" if is_saved else "Save", icon=":material/bookmark:", key=f"{key_prefix}_save_{item['id']}",
				disabled=is_saved or not st.session_state.get("saved_reports_ready", False) or bool(st.session_state.get("_saved_report_operation")),
				on_click=queue_saved_report_operation, args=("save", item),
			)
		render_export_controls(item, report_export, key_prefix, "content_strategy_report")


@st.dialog("Saved report", width="large")
def show_saved_report(item):
	st.subheader(item["title"])
	st.caption(export_scope_details(item))
	st.markdown(item["content"])
	render_report_actions(item, allow_save=False, key_prefix="saved_report")


@st.dialog("Saved conversation", width="large")
def show_saved_chat(item):
	with st.chat_message("user"):
		st.markdown(item["question"])
	with st.chat_message("assistant"):
		st.markdown(item["answer"])
		render_sources(item["sources"], key_prefix=f"saved_{item['id']}")
	scope = item.get("scope", {})
	st.caption(f"{item.get('created_at', '')} · {', '.join(scope.get('files', []))} · {scope.get('time_range', '')}")
	render_conversation_actions(item, allow_save=False, key_prefix="saved")


def generate_chat_answer(prompt, collection):
	allowed_document_ids = {
		f"{clean_text(row['_source_file'])}::{row_number}"
		for row_number, row in dataset.iterrows()
	}
	if not allowed_document_ids:
		return None
	indexed_count = collection.count()
	if not indexed_count:
		return None
	client = get_genai_client(GEMINI_API_KEY)
	query_embedding = embed_texts(client, [prompt], "RETRIEVAL_QUERY")[0]
	mentioned_sites = [
		site
		for site in sorted(dataset["Sito"].dropna().astype(str).unique(), key=len, reverse=True)
		if re.search(rf"(?<!\w){re.escape(site)}(?!\w)", prompt, flags=re.IGNORECASE)
	]
	candidates = []
	if mentioned_sites:
		base_count, remainder = divmod(CHAT_RESULT_COUNT, len(mentioned_sites))
		for site_index, site in enumerate(mentioned_sites):
			per_site = max(1, base_count + (site_index < remainder))
			site_candidates = []
			site_results = collection.query(
				query_embeddings=[query_embedding],
				n_results=indexed_count,
				where={"Sito": site},
				include=["documents", "metadatas", "distances"],
			)
			for document_id, document, metadata, distance in zip(
				site_results.get("ids", [[]])[0],
				site_results.get("documents", [[]])[0],
				site_results.get("metadatas", [[]])[0],
				site_results.get("distances", [[]])[0],
			):
				if document_id in allowed_document_ids:
					site_candidates.append((distance, document, metadata))
			site_candidates.sort(key=lambda item: item[0])
			candidates.extend(site_candidates[:per_site])
		candidates.sort(key=lambda item: item[0])
		documents = [item[1] for item in candidates[:CHAT_RESULT_COUNT]]
		metadatas = [item[2] for item in candidates[:CHAT_RESULT_COUNT]]
	else:
		results = collection.query(
			query_embeddings=[query_embedding],
			n_results=indexed_count,
			include=["documents", "metadatas", "distances"],
		)
		filtered_results = [
			(document, metadata)
			for document_id, document, metadata in zip(
				results.get("ids", [[]])[0],
				results.get("documents", [[]])[0],
				results.get("metadatas", [[]])[0],
			)
			if document_id in allowed_document_ids
		][:CHAT_RESULT_COUNT]
		documents = [item[0] for item in filtered_results]
		metadatas = [item[1] for item in filtered_results]
	if not documents:
		return None
	sources = []
	context_blocks = []
	for index, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
		metadata = metadata or {}
		sources.append(metadata)
		context_blocks.append(
			f"[Source {index}] Website: {metadata.get('Sito', '')}; "
			f"Title: {metadata.get('Title', '')}; URL: {metadata.get('URL', '')}\n{document}"
		)

	answer_prompt = (
		"You are a content strategy analyst. Respond in US English with a detailed, practical comparison "
		"based only on the provided sources. Distinguish facts from inferences, note when the data is "
		"insufficient, and cite sources in the answer as [Source 1].\n\n"
		f"QUESTION:\n{prompt}\n\nRETRIEVED CONTEXT:\n" + "\n\n".join(context_blocks)
	)
	response = client.models.generate_content(model=CHAT_MODEL, contents=answer_prompt)
	return response.text or "Gemini did not return a text response.", sources


def queue_question_preference_operation(action, question=None):
	st.session_state["_question_preference_operation"] = {"id": str(uuid4()), "action": action, "question": question}


def render_question_preferences_storage():
	st.session_state.setdefault("question_preferences", {"version": 1, "favorites": [], "hidden_shared_ids": []})
	st.session_state.setdefault("question_preferences_ready", False)
	result = question_preferences_storage(
		data={"operation": st.session_state.get("_question_preference_operation"), "snapshot": st.session_state.get("_last_question_preferences_snapshot")},
		default={"snapshot": None}, key="question_preferences_browser_storage", on_snapshot_change=lambda: None,
	)
	if snapshot := result.snapshot:
		st.session_state["_last_question_preferences_snapshot"] = snapshot
		st.session_state.question_preferences_ready = not snapshot.get("error")
		if snapshot.get("error"):
			st.warning(f"Question preferences could not be saved or loaded: {snapshot['error']}")
		else:
			st.session_state.question_preferences = snapshot["preferences"]
		operation = st.session_state.get("_question_preference_operation")
		if operation and snapshot.get("operation_id") == operation["id"]:
			st.session_state.pop("_question_preference_operation")
			if snapshot.get("error"):
				st.session_state["_question_error"] = f"Could not save question preferences: {snapshot['error']}"
			else:
				st.session_state["_question_notice"] = "Question preferences saved."
			st.rerun()


def add_favorite_question(shared_questions):
	try:
		text = normalize_question(st.session_state.get("favorite_question_text", ""))
		preferences = st.session_state.question_preferences
		existing = preferences["favorites"] + [
			question for question in shared_questions
			if question["enabled"] and question["id"] not in preferences["hidden_shared_ids"]
		]
		if any(question["text"].casefold() == text.casefold() for question in existing):
			raise ValueError("This question already exists.")
		queue_question_preference_operation("add", {"id": str(uuid4()), "text": text})
		st.session_state.favorite_question_text = ""
	except ValueError as exc:
		st.session_state["_question_error"] = str(exc)


def shared_question_editing_allowed():
	if os.getenv("ALLOW_SHARED_QUESTION_EDITING", "").lower() == "true":
		return True
	try:
		return st.secrets.get("ALLOW_SHARED_QUESTION_EDITING", False) is True
	except StreamlitSecretNotFoundError:
		return False


@st.dialog("Shared question settings", width="large")
def show_question_settings(config):
	can_edit = shared_question_editing_allowed()
	rows = [{"id": question["id"], "Position": index + 1, "Question": question["text"], "Enabled": question["enabled"]}
		for index, question in enumerate(config["suggested_questions"])]
	with st.form("shared_question_settings_form"):
		edited = st.data_editor(
			pd.DataFrame(rows, columns=["id", "Position", "Question", "Enabled"]),
			key="shared_question_editor", hide_index=True, num_rows="dynamic", height=400, disabled=not can_edit,
			column_config={"id": None, "Position": st.column_config.NumberColumn(min_value=1, step=1),
				"Question": st.column_config.TextColumn(required=True, max_chars=1000, width="large"),
				"Enabled": st.column_config.CheckboxColumn(default=True, required=True)},
		)
		submitted = st.form_submit_button("Save shared questions", icon=":material/save:", disabled=not can_edit)
	if submitted and can_edit:
		try:
			questions = []
			for index, row in enumerate(edited.to_dict("records")):
				position = index + 1 if pd.isna(row.get("Position")) else int(row["Position"])
				questions.append((position, {"id": row["id"] if isinstance(row.get("id"), str) else str(uuid4()),
					"text": row["Question"], "enabled": row["Enabled"]}))
			updated = {"version": 1, "suggested_questions": [question for position, question in sorted(questions, key=lambda item: item[0])]}
			save_question_config(QUESTION_CONFIG_PATH, updated, config)
			st.session_state["_question_notice"] = "Shared questions saved."
			st.rerun()
		except (OSError, ValueError, TypeError) as exc:
			st.error(f"Could not save shared questions: {exc}")


def render_suggested_questions():
	try:
		config = load_question_config(QUESTION_CONFIG_PATH)
	except (OSError, ValueError) as exc:
		st.warning(f"Shared questions could not be loaded: {exc}")
		config = None
	shared_questions = config["suggested_questions"] if config else []
	preferences = st.session_state.question_preferences
	questions = [(question, "delete") for question in preferences["favorites"]]
	questions += [(question, "hide") for question in shared_questions
		if question["enabled"] and question["id"] not in preferences["hidden_shared_ids"]]
	busy = not st.session_state.question_preferences_ready or bool(st.session_state.get("_question_preference_operation"))
	with st.expander("Suggested questions", key="suggested_questions_open", on_change="rerun"):
		if notice := st.session_state.pop("_question_notice", None):
			st.success(notice)
		if error := st.session_state.pop("_question_error", None):
			st.error(error)
		if questions:
			with st.container(height=min(400, len(questions) * 80 - 16), border=False, key="suggested_questions_list"):
				for question, action in questions:
					question_column, delete_column = st.columns([5, 1], gap="small")
					with question_column:
						st.button(question["text"], key=f"suggested_question_{action}_{question['id']}",
							icon=":material/bookmark:" if action == "delete" else ":material/chat:", width="stretch",
							help=question["text"], on_click=queue_suggested_question, args=(question["text"],))
					with delete_column:
						st.button("", key=f"remove_question_{action}_{question['id']}", icon=":material/delete:",
							help="Remove favorite question" if action == "delete" else "Hide shared question in this browser",
							disabled=busy, on_click=queue_question_preference_operation, args=(action, question))
		else:
			st.caption("No suggested questions.")
		with st.form("add_favorite_question_form"):
			st.text_input("Favorite question", key="favorite_question_text", max_chars=1000)
			st.form_submit_button("Add favorite", icon=":material/add:", disabled=busy,
				on_click=add_favorite_question, args=(shared_questions,))
		with st.container(horizontal=True):
			if st.button("Shared settings", icon=":material/settings:", disabled=config is None):
				show_question_settings(config)
			if preferences["hidden_shared_ids"]:
				st.button("Restore shared questions", icon=":material/undo:", disabled=busy,
					on_click=queue_question_preference_operation, args=("restore",))


def queue_suggested_question(question):
	st.session_state.suggested_chat_prompt = question
	st.session_state.suggested_questions_open = False


def render_chat():
	st.session_state.setdefault("messages", [])
	st.html("""<style>
		div:has(> .st-key-chat_composer) {position: sticky; top: 3.75rem; z-index: 20;}
		.st-key-chat_composer {background: var(--background-color); padding: 0.5rem 0;}
		.st-key-suggested_questions_list [data-testid="stHorizontalBlock"] {flex-wrap: nowrap;}
		.st-key-suggested_questions_list [data-testid="stColumn"] {min-width: 0;}
		.st-key-suggested_questions_list button {height: 64px;}
		.st-key-suggested_questions_list button p {display: -webkit-box;
			-webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;}
	</style>""")
	with st.container(key="chat_composer"):
		typed_prompt = st.chat_input("Ask a comparative question about the content...", key="comparative_chat_input")
	render_suggested_questions()
	prompt = st.session_state.pop("suggested_chat_prompt", None) or typed_prompt
	if prompt:
		if not GEMINI_API_KEY:
			st.error("API key not found in the .env file. Check your configuration.")
		elif dataset.empty:
			st.warning("No articles match the selected time range.")
		elif (collection := get_indexed_collection()) is None:
			st.warning("The search index is missing or uses an outdated content schema. Start Data Ingestion from the sidebar first.")
		else:
			try:
				with st.spinner("Searching sources and preparing a comparative analysis..."):
					result = generate_chat_answer(prompt, collection)
				if result is None:
					st.warning("No indexed articles match the selected time range. Run data ingestion to refresh the index.")
				else:
					answer, sources = result
					st.session_state.messages.extend([
						{"role": "user", "content": prompt},
						{"role": "assistant", "content": answer, "sources": sources,
						 "id": str(uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
						 "scope": {"files": [path.name for path in selected_files], "time_range": analysis_time_range}},
					])
			except Exception as exc:
				st.error(f"Error while searching or generating a Gemini response: {exc}")

	exchanges = []
	for message in st.session_state.messages:
		if message["role"] == "user":
			exchanges.append([message])
		elif exchanges:
			exchanges[-1].append(message)
	for exchange in reversed(exchanges):
		for message in exchange:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])
				if message["role"] == "assistant":
					message.setdefault("id", str(uuid4()))
					render_sources(message.get("sources", []), key_prefix=message["id"])
					item = {
						"id": message["id"], "question": exchange[0]["content"], "answer": message["content"],
						"sources": message.get("sources", []), "created_at": message.get("created_at", ""),
						"scope": message.get("scope", {}),
					}
					render_conversation_actions(item)
		st.divider()


def render_report(frame):
	if st.button("Generate Content Strategy Report", type="primary"):
		if not GEMINI_API_KEY:
			st.error("API key not found in the .env file. Check your configuration.")
		elif frame.empty:
			st.warning("There is no valid CSV data to analyze.")
		else:
			try:
				with st.spinner("Gemini is analyzing data from all websites..."):
					client = get_genai_client(GEMINI_API_KEY)
					prompt = (
						"Act as a senior content strategist specializing in competitive analysis and SEO. Write an "
						"in-depth, practical report in US English based on the supplied data. Use exactly these main "
						"sections: ## Competitor Analysis, ## Content Gaps, and ## Positioning Opportunities. Compare all "
						"websites, quantify observations when possible, distinguish facts from hypotheses, and recommend "
						"actionable priorities ranked by impact and feasibility. Do not invent missing data.\n\n"
						f"AGGREGATED DATA AND ARTICLE SAMPLES:\n{report_context(frame)}"
					)
					response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
					st.session_state.strategy_report = create_report_item(response.text or "Gemini did not return a text report.", frame)
			except Exception as exc:
				st.error(f"Error while generating the report: {exc}")

	if report := st.session_state.get("strategy_report"):
		if isinstance(report, str):
			report = create_report_item(report, frame)
			st.session_state.strategy_report = report
		st.markdown(report["content"])
		render_report_actions(report)


files = scan_csv_files()
all_dataset, all_csv_errors = load_dataset(files)
with st.sidebar:
	st.subheader("Available competitors")
	selected_files = []
	if files:
		for csv_file in files:
			if st.checkbox(csv_file.stem, value=True, key=f"data_source_{csv_file.name}"):
				selected_files.append(csv_file)
	else:
		st.caption("No CSV files found in `data/`.")
	st.divider()
	analysis_time_range = st.selectbox(
		"Analysis Time Range",
		["All data", "Last 3 months", "Last 6 months", "Last year", "Last 2 years"],
		key="analysis_time_range",
	)

	ingest_clicked = st.button("Start Data Ingestion", type="primary", use_container_width=True)

selected_file_names = {path.name for path in selected_files}
selected_dataset = all_dataset.loc[all_dataset["_source_file"].isin(selected_file_names)].copy()
csv_errors = [
	error
	for error in all_csv_errors
	if any(error.startswith(f"{path.name}:") for path in selected_files)
]

if ingest_clicked:
	if not GEMINI_API_KEY:
		st.error("API key not found in the .env file. Check your configuration.")
	elif not selected_files:
		st.warning("Select at least one CSV source before starting ingestion.")
	elif csv_errors and selected_dataset.empty:
		st.error("No valid selected CSV files to index. Check the columns and file formatting.")
	else:
		progress_bar = st.progress(0, text="Preparing ingestion...")
		try:
			client = get_genai_client(GEMINI_API_KEY)
			indexed_count = ingest_dataset(
				selected_dataset,
				client,
				lambda value: progress_bar.progress(value, text="Generating embeddings..."),
			)
			progress_bar.empty()
			st.success(f"Ingestion complete: {indexed_count} articles indexed.")
		except Exception as exc:
			progress_bar.empty()
			st.error(f"Error during ingestion: {exc}")

max_date = selected_dataset["Date published"].max() if not selected_dataset.empty else pd.NaT
previous_month_end = pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)
period_end_date = previous_month_end if pd.isna(max_date) else min(max_date.normalize(), previous_month_end)

active_analysis_scope = (tuple(path.name for path in selected_files), analysis_time_range)
if st.session_state.get("_active_analysis_scope") != active_analysis_scope:
	st.session_state.messages = []
	st.session_state.pop("strategy_report", None)
	st.session_state["_active_analysis_scope"] = active_analysis_scope

dataset = selected_dataset.copy()
range_months = {
	"Last 3 months": 3,
	"Last 6 months": 6,
	"Last year": 12,
	"Last 2 years": 24,
}
if pd.isna(max_date):
	if analysis_time_range != "All data":
		dataset = selected_dataset.iloc[0:0].copy()
		st.warning("No valid publication dates were found. Select All data or check the source files.")
else:
	end_exclusive = period_end_date + pd.Timedelta(days=1)
	if analysis_time_range == "All data":
		dataset = selected_dataset.loc[
			selected_dataset["Date published"].isna()
			| selected_dataset["Date published"].lt(end_exclusive)
		].copy()
	else:
		start_date = period_end_date.replace(day=1) - pd.DateOffset(
			months=range_months[analysis_time_range] - 1
		)
		dataset = selected_dataset.loc[
			selected_dataset["Date published"].ge(start_date)
			& selected_dataset["Date published"].lt(end_exclusive)
		].copy()

with st.sidebar:
	render_question_preferences_storage()
render_saved_chats_sidebar()
render_saved_reports_sidebar()

st.title("Content Intelligence")
st.caption("Competitive content analysis, semantic search, and editorial strategy.")
if not GEMINI_API_KEY:
	st.error("API key not found in the .env file. Check your configuration.")
for csv_error in csv_errors:
	st.warning(csv_error)

dashboard_tab, chat_tab, report_tab = st.tabs(
	["📊 Quantitative Dashboard", "🤖 AI Assistant & Comparative Chat", "📄 Strategic Report Generator"],
	key="content_views", on_change="rerun",
)

with dashboard_tab:
	render_dashboard(dataset)

with chat_tab:
	render_chat()

with report_tab:
	render_report(dataset)
