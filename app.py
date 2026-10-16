import os
import re
import time
import hashlib
import gzip
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
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
import plotly.io as pio
import streamlit as st
import streamlit.components.v2 as components
from streamlit.errors import StreamlitSecretNotFoundError
from google import genai
from google.genai import types
from suggested_questions import load_question_config, normalize_question, save_question_config


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
EDITORIAL_METRICS_CACHE_PATH = CHROMA_DIR / "editorial_metrics.json.gz"
EDITORIAL_METRICS_SCHEMA_VERSION = 3
EDITORIAL_METRIC_FIELDS = (
	"Topics", "Vertical", "Funnel stage", "Buyer persona", "Strategic initiative",
	"Features Mentioned", "Target audience", "Post type",
)
EDITORIAL_MULTI_VALUE_FIELDS = {
	"Topics", "Vertical", "Funnel stage", "Buyer persona", "Strategic initiative",
	"Features Mentioned", "Target audience",
}
QUESTION_CONFIG_PATH = Path(os.getenv("SUGGESTED_QUESTIONS_CONFIG", str(ROOT_DIR / "suggested_questions.json"))).expanduser()
COLLECTION_NAME = "website_content"
CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 64
CHAT_RESULT_COUNT = 12
MAX_RETRIEVAL_COSINE_DISTANCE = 0.65
MAX_GENERATION_OUTPUT_TOKENS = 8192
MAX_GENERATION_CONTINUATIONS = 2
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


def csv_source_fingerprints(files):
	return {
		path.name: hashlib.sha256(path.read_bytes()).hexdigest()
		for path in sorted(files, key=lambda item: item.name.casefold())
	}


def new_editorial_metrics_bucket():
	return {"articles": 0, "word_count_sum": 0.0, "word_count_n": 0, "word_counts": [], "categories": {}}


def build_editorial_metrics_cache(frame, source_fingerprints):
	sites = {}
	for site, group in frame.groupby("Sito", sort=True, dropna=False):
		site_name = clean_text(site) or "Unknown"
		word_counts = article_length_series(group)
		monthly = {}
		for row_index, row in group.iterrows():
			published = pd.to_datetime(row.get("Date published"), errors="coerce")
			month_key = published.strftime("%Y-%m") if not pd.isna(published) else "undated"
			bucket = monthly.setdefault(month_key, new_editorial_metrics_bucket())
			bucket["articles"] += 1
			word_count = word_counts.loc[row_index]
			if not pd.isna(word_count):
				word_count = float(word_count)
				bucket["word_count_sum"] += word_count
				bucket["word_count_n"] += 1
				bucket["word_counts"].append(word_count)
			for field in EDITORIAL_METRIC_FIELDS:
				value = clean_text(row.get(field))
				values = (
				normalize_funnel_stage(value) if field == "Funnel stage"
				else split_category_cell(value) if field in EDITORIAL_MULTI_VALUE_FIELDS
				else ([value] if value else [])
			)
				for category in values:
					counts = bucket["categories"].setdefault(field, {})
					counts[category] = counts.get(category, 0) + 1
		monthly = dict(sorted(monthly.items(), key=lambda item: item[0]))
		sites[site_name] = {"article_count": len(group), "months": monthly}
	return {
		"schema_version": EDITORIAL_METRICS_SCHEMA_VERSION,
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"source_fingerprints": source_fingerprints,
		"row_count": len(frame),
		"sites": sites,
	}


def write_editorial_metrics_cache(cache, cache_path=EDITORIAL_METRICS_CACHE_PATH):
	cache_path = Path(cache_path)
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	temporary_path = None
	try:
		with tempfile.NamedTemporaryFile(
			mode="wb", dir=cache_path.parent,
			prefix=f".{cache_path.name}.", suffix=".tmp", delete=False,
		) as temporary_file:
			temporary_path = Path(temporary_file.name)
		with gzip.open(temporary_path, mode="wt", encoding="utf-8") as temporary_file:
			json.dump(cache, temporary_file, ensure_ascii=False, separators=(",", ":"))
		os.replace(temporary_path, cache_path)
	except Exception:
		if temporary_path and temporary_path.exists():
			temporary_path.unlink()
		raise
	return cache


def load_or_build_editorial_metrics_cache(frame, files, cache_path=EDITORIAL_METRICS_CACHE_PATH):
	fingerprints = csv_source_fingerprints(files)
	cache_path = Path(cache_path)
	try:
		with gzip.open(cache_path, mode="rt", encoding="utf-8") as cache_file:
			cache = json.load(cache_file)
		if (
			cache.get("schema_version") == EDITORIAL_METRICS_SCHEMA_VERSION
			and cache.get("source_fingerprints") == fingerprints
			and cache.get("row_count") == len(frame)
		):
			return cache
	except (OSError, ValueError, TypeError):
		pass
	cache = build_editorial_metrics_cache(frame, fingerprints)
	return write_editorial_metrics_cache(cache, cache_path)


def save_editorial_metrics_cache(frame, files, cache_path=EDITORIAL_METRICS_CACHE_PATH):
	cache = build_editorial_metrics_cache(frame, csv_source_fingerprints(files))
	return write_editorial_metrics_cache(cache, cache_path)


def editorial_metrics_by_site(metrics_cache, sites, analysis_time_range="All data", period_end_date=None):
	month_window = {"Last 3 months": 3, "Last 6 months": 6, "Last year": 12, "Last 2 years": 24}
	available_months = sorted({
		month
		for site in sites
		for month in metrics_cache.get("sites", {}).get(site, {}).get("months", {})
		if month != "undated"
	})
	if period_end_date is not None and not pd.isna(period_end_date):
		end_month = pd.Timestamp(period_end_date).to_period("M")
	elif available_months:
		end_month = pd.Period(available_months[-1], freq="M")
	else:
		end_month = None
	if end_month is None:
		common_month_keys = []
	else:
		if analysis_time_range == "All data":
			start_month = pd.Period(available_months[0], freq="M") if available_months else end_month
		else:
			start_month = end_month - (month_window[analysis_time_range] - 1)
		common_month_keys = [str(month) for month in pd.period_range(start_month, end_month, freq="M")]
	result = {}
	for site in sites:
		site_cache = metrics_cache.get("sites", {}).get(site)
		if not site_cache:
			continue
		month_keys = common_month_keys

		category_counts = {}
		monthly = []
		article_count = 0
		word_count_sum = 0.0
		word_count_n = 0
		word_counts = []
		for month_key in month_keys:
			bucket = site_cache["months"].get(month_key, new_editorial_metrics_bucket())
			article_count += int(bucket["articles"])
			word_count_sum += float(bucket["word_count_sum"])
			word_count_n += int(bucket["word_count_n"])
			word_counts.extend(bucket["word_counts"])
			for field, values in bucket["categories"].items():
				field_counts = category_counts.setdefault(field, {})
				for value, count in values.items():
					field_counts[value] = field_counts.get(value, 0) + int(count)
			monthly.append({
				"month": month_key,
				"articles": int(bucket["articles"]),
				"word_count_sum": float(bucket["word_count_sum"]),
				"word_count_n": int(bucket["word_count_n"]),
				"categories": bucket["categories"],
				"average_word_count": (
					float(bucket["word_count_sum"]) / int(bucket["word_count_n"])
					if bucket["word_count_n"] else None
				),
			})

		if analysis_time_range == "All data":
			undated = site_cache["months"].get("undated", new_editorial_metrics_bucket())
			article_count += int(undated["articles"])
			word_count_sum += float(undated["word_count_sum"])
			word_count_n += int(undated["word_count_n"])
			word_counts.extend(undated["word_counts"])
			for field, values in undated["categories"].items():
				field_counts = category_counts.setdefault(field, {})
				for value, count in values.items():
					field_counts[value] = field_counts.get(value, 0) + int(count)
		result[site] = {
			"article_count": article_count,
			"word_count_sum": word_count_sum,
			"word_count_n": word_count_n,
			"word_counts": word_counts,
			"average_word_count": word_count_sum / word_count_n if word_count_n else None,
			"monthly": monthly,
			"category_counts": category_counts,
		}
	return result


def article_document(row, max_field_chars=None):
	lines = []
	for column in REQUIRED_COLUMNS:
		value = clean_text(row.get(column))
		if value:
			if max_field_chars is not None:
				value = value[:max_field_chars]
			lines.append(f"{column}: {value}")
	return "\n".join(lines)


def split_category_cell(value):
	if value is None or pd.isna(value):
		return []
	return list(dict.fromkeys(part.strip() for part in str(value).split(";") if part.strip()))


def normalize_funnel_stage(value):
	aliases = {
		"TOFU": ["TOFU"],
		"TOP OF FUNNEL": ["TOFU"],
		"TOP OF FUNNEL (TOFU)": ["TOFU"],
		"AWARENESS": ["TOFU"],
		"AWARENESS/CONSIDERATION": ["TOFU", "MOFU"],
		"MOFU": ["MOFU"],
		"MIDDLE OF FUNNEL": ["MOFU"],
		"MIDDLE OF FUNNEL (MOFU)": ["MOFU"],
		"CONSIDERATION": ["MOFU"],
		"CONSIDERATION/DECISION": ["MOFU", "BOFU"],
		"BOFU": ["BOFU"],
		"BOTTOM OF FUNNEL": ["BOFU"],
		"BOTTOM OF FUNNEL (BOFU)": ["BOFU"],
		"DECISION": ["BOFU"],
		"AWARENESS/CONSIDERATION/DECISION": ["TOFU", "MOFU", "BOFU"],
	}
	normalized_values = []
	for value_part in split_category_cell(value):
		label = re.sub(r"\s*/\s*", "/", re.sub(r"\s+", " ", value_part.strip())).upper()
		normalized_values.extend(aliases.get(label, []))
	return list(dict.fromkeys(normalized_values))


def explode_category_values(values):
	return values.map(split_category_cell).explode().dropna().astype(str).str.strip()


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


def format_elapsed_duration(seconds):
	seconds = max(0, int(seconds))
	hours, remainder = divmod(seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def response_reached_output_limit(response):
	for candidate in getattr(response, "candidates", None) or []:
		reason = getattr(candidate, "finish_reason", None)
		reason_name = getattr(reason, "name", None) or str(reason).rsplit(".", 1)[-1]
		if str(reason_name).upper().endswith("MAX_TOKENS"):
			return True
	return False


def generate_complete_response(client, prompt):
	config = types.GenerateContentConfig(max_output_tokens=MAX_GENERATION_OUTPUT_TOKENS)
	response = client.models.generate_content(model=CHAT_MODEL, contents=prompt, config=config)
	answer = response.text or ""
	for _ in range(MAX_GENERATION_CONTINUATIONS):
		if not response_reached_output_limit(response):
			return answer, False
		continuation_prompt = (
			"The previous response reached the output-token limit. Continue from exactly where it stopped. "
			"Do not repeat earlier text. Keep the same language, use only the original evidence, and finish "
			"all sections and conclusions.\n\n"
			f"ORIGINAL QUESTION AND EVIDENCE:\n{prompt}\n\nPARTIAL RESPONSE TO CONTINUE:\n{answer}"
		)
		response = client.models.generate_content(
			model=CHAT_MODEL, contents=continuation_prompt, config=config,
		)
		continuation = response.text or ""
		if continuation:
			answer = f"{answer.rstrip()}\n{continuation.lstrip()}"
	return answer, response_reached_output_limit(response)


def run_operation_with_progress(operation, label, status_callback, ingestion_started, poll_interval=1):
	operation_started = time.monotonic()
	with ThreadPoolExecutor(max_workers=1) as executor:
		future = executor.submit(operation)
		while True:
			try:
				result = future.result(timeout=poll_interval)
				break
			except FutureTimeoutError:
				if status_callback:
					status_callback(
						f"{label} · running {format_elapsed_duration(time.monotonic() - operation_started)}"
						f" · total {format_elapsed_duration(time.monotonic() - ingestion_started)}"
					)
	if status_callback:
		status_callback(
			f"{label} · complete in {format_elapsed_duration(time.monotonic() - operation_started)}"
			f" · total {format_elapsed_duration(time.monotonic() - ingestion_started)}"
		)
	return result


def ingest_dataset(frame, gemini_client, progress_callback, status_callback=None):
	if frame.empty:
		raise ValueError("There are no valid rows to index in the detected CSV files.")

	ingestion_started = time.monotonic()
	preparation_started = time.monotonic()
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
	if status_callback:
		status_callback(
			f"Prepared {len(documents)} articles in "
			f"{format_elapsed_duration(time.monotonic() - preparation_started)}"
		)

	all_embeddings = []
	total_batches = (len(documents) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
	for batch_number, start in enumerate(range(0, len(documents), EMBED_BATCH_SIZE), start=1):
		progress_callback(0.9 * (batch_number - 1) / total_batches)
		batch_embeddings = run_operation_with_progress(
			lambda start=start: embed_texts(
				gemini_client,
				documents[start : start + EMBED_BATCH_SIZE],
				"RETRIEVAL_DOCUMENT",
			),
			f"Generating embeddings · batch {batch_number}/{total_batches}",
			status_callback,
			ingestion_started,
		)
		all_embeddings.extend(batch_embeddings)
		progress_callback(0.9 * batch_number / total_batches)

	chroma_client = get_chroma_client()
	try:
		chroma_client.delete_collection(COLLECTION_NAME)
	except Exception:
		pass
	collection = chroma_client.create_collection(
		name=COLLECTION_NAME,
		metadata={"hnsw:space": "cosine", "content_schema_version": CONTENT_SCHEMA_VERSION},
	)

	storage_batches = (len(documents) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
	for batch_number, start in enumerate(range(0, len(documents), EMBED_BATCH_SIZE), start=1):
		end = start + EMBED_BATCH_SIZE
		progress_callback(0.9 + 0.1 * (batch_number - 1) / storage_batches)
		run_operation_with_progress(
			lambda start=start, end=end: collection.add(
				ids=ids[start:end],
				embeddings=all_embeddings[start:end],
				documents=documents[start:end],
				metadatas=metadatas[start:end],
			),
			f"Saving indexed articles · batch {batch_number}/{storage_batches}",
			status_callback,
			ingestion_started,
		)
		progress_callback(0.9 + 0.1 * batch_number / storage_batches)
	if status_callback:
		status_callback(f"Ingestion complete · total {format_elapsed_duration(time.monotonic() - ingestion_started)}")
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

		stages = group["Funnel stage"].map(normalize_funnel_stage).explode().dropna().value_counts(normalize=True)
		if not stages.empty:
			lines.append("Funnel distribution (%): " + ", ".join(f"{name} {value * 100:.1f}%" for name, value in stages.items()))
		for column in ("Topics", "Post type", "Vertical", "Buyer persona", "Strategic initiative", "Features Mentioned", "Target audience"):
			counts = explode_category_values(group[column]).value_counts().head(6)
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


def topic_frequency_context(frame):
	lines = ["Complete article counts by exact Topics value and website:"]
	for site, group in frame.groupby("Sito", sort=True):
		counts = explode_category_values(group["Topics"]).value_counts()
		article_count = group["Topics"].map(split_category_cell).map(bool).sum()
		lines.append(f"\nWebsite: {site}; articles with a populated Topics value: {int(article_count)}")
		lines.extend(f"- {topic}: {count}" for topic, count in counts.items())
	return "\n".join(lines)


def editorial_metrics_context(frame, prompt, metrics_cache=None, analysis_time_range="All data", period_end_date=None):
	field_patterns = [
		("Topics", r"\btopics?\b|\bthemes?\b|\bargomenti\b|\btemi\b"),
		("Vertical", r"\bverticals?\b|\bindustr(?:y|ies)\b|\bsectors?\b|\bsettori\b"),
		("Funnel stage", r"\bfunnel stages?\b|\btofu\b|\bmofu\b|\bbofu\b|\bfase funnel\b"),
		("Strategic initiative", r"\bstrategic initiatives?\b|\binitiatives?\b|\biniziative strategiche\b"),
		("Target audience", r"\btarget audiences?\b|\baudiences?\b|\bpubblico target\b"),
		("Post type", r"\bpost types?\b|\bcontent formats?\b|\btipo di post\b"),
	]
	requested_fields = [field for field, pattern in field_patterns if re.search(pattern, prompt, flags=re.IGNORECASE)]
	frequency_requested = bool(re.search(
		r"publishing frequency|publication frequency|publishing cadence|publication cadence|editorial velocity|"
		r"articles? per (?:month|week|year)|posts? per (?:month|week|year)|how often|frequency|cadence|over time|"
		r"frequenza di pubblicazione|cadenza di pubblicazione|con quale frequenza|articoli per mese",
		prompt, flags=re.IGNORECASE,
	))
	length_requested = bool(re.search(
		r"article length|average length|average word count|word count|words per article|longer|shorter|"
		r"lunghezza degli articoli|lunghezza media|conteggio parole|parole per articolo",
		prompt, flags=re.IGNORECASE,
	))
	priorities_requested = bool(re.search(
		r"editorial priorit(?:y|ies)|content priorit(?:y|ies)|topic trends?|topic changes?|themes? over time|"
		r"what (?:topics|themes) (?:are|do)|priorità editoriali|priorità dei contenuti|temi editoriali",
		prompt, flags=re.IGNORECASE,
	))
	if priorities_requested and "Topics" not in requested_fields:
		requested_fields.append("Topics")
	if length_requested:
		requested_fields.append("Word count")
	if not (frequency_requested or requested_fields):
		return ""
	if metrics_cache is None:
		metrics_cache = build_editorial_metrics_cache(frame, {})

	sites = sorted(frame["Sito"].dropna().astype(str).unique(), key=str.casefold)
	summaries = editorial_metrics_by_site(metrics_cache, sites, analysis_time_range, period_end_date)
	lines = [
		"DETERMINISTIC EDITORIAL METRICS, calculated from the versioned CSV metrics sidecar and all rows in the selected scope.",
		"Cite these complete-dataset calculations as [Dataset metrics]. Months with no listed articles have a count of zero.",
	]

	if frequency_requested:
		for site, summary in summaries.items():
			series = ", ".join(f"{bucket['month']}={bucket['articles']}" for bucket in summary["monthly"])
			lines.append(f"Monthly article counts · {site}: {series or 'no dated articles'}")

	if length_requested:
		for site, summary in summaries.items():
			values = summary["word_counts"]
			if not values:
				lines.append(f"Average article length · {site}: no valid word counts.")
				continue
			median = float(pd.Series(values).median())
			lines.append(
				f"Average article length · {site}: {summary['average_word_count']:.1f} words "
				f"(median {median:.1f}; n={summary['word_count_n']})."
			)
			yearly = {}
			for bucket in summary["monthly"]:
				year = bucket["month"][:4]
				entry = yearly.setdefault(year, {"sum": 0.0, "n": 0})
				entry["sum"] += bucket["word_count_sum"]
				entry["n"] += bucket["word_count_n"]
			if yearly:
				lines.append("Annual average article length · " + site + ": " + ", ".join(
					f"{year}={values['sum'] / values['n']:.1f}"
					for year, values in sorted(yearly.items()) if values["n"]
				))

	if priorities_requested:
		for site, summary in summaries.items():
			yearly_topics = {}
			for bucket in summary["monthly"]:
				year = bucket["month"][:4]
				for topic, count in bucket["categories"].get("Topics", {}).items():
					yearly_topics.setdefault(year, {})[topic] = yearly_topics.setdefault(year, {}).get(topic, 0) + count
			for year, topic_counts in sorted(yearly_topics.items()):
				top_topics = sorted(topic_counts.items(), key=lambda item: (-item[1], item[0].casefold()))[:5]
				lines.append(f"Top editorial topics · {site} · {year}: " + ", ".join(f"{topic} ({count})" for topic, count in top_topics))

	for field in requested_fields:
		limit = 10
		for site, summary in summaries.items():
			counts = summary["category_counts"].get(field, {})
			if not counts:
				lines.append(f"{field} · {site}: no populated values.")
				continue
			ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))
			prompt_lower = prompt.casefold()
			explicit = [(value, count) for value, count in ordered if len(value) > 3 and value.casefold() in prompt_lower]
			selected = list(dict.fromkeys(explicit + ordered))[:limit]
			formatted = [
				f"{value[:180] + ('…' if len(value) > 180 else '')} ({count})"
				for value, count in selected
			]
			lines.append(f"Top {field} values · {site} (unique={len(counts)}): " + "; ".join(formatted))
	return "\n".join(lines)


def is_csv_metrics_only_question(prompt, metrics_context):
	return bool(metrics_context) and not re.search(
		r"\bwhy\b|\bcause\b|\breason\b|\bbackground\b|\bqualitative context\b|"
		r"\bperché\b|\bcause\b",
		prompt, flags=re.IGNORECASE,
	)


def build_editorial_metric_charts(metrics_cache, selected_sites, analysis_time_range="All data", period_end_date=None):
	if isinstance(metrics_cache, pd.DataFrame):
		frame = metrics_cache
		selected_sites = [selected_sites] if isinstance(selected_sites, str) else selected_sites
		metrics_cache = build_editorial_metrics_cache(frame, {})
	else:
		selected_sites = [selected_sites] if isinstance(selected_sites, str) else selected_sites
	summaries = editorial_metrics_by_site(metrics_cache, selected_sites, analysis_time_range, period_end_date)
	frequency_rows = []
	length_rows = []
	article_total = 0
	word_count_sum = 0.0
	word_count_n = 0
	for site, summary in summaries.items():
		article_total += summary["article_count"]
		word_count_sum += summary["word_count_sum"]
		word_count_n += summary["word_count_n"]
		for bucket in summary["monthly"]:
			month = pd.Period(bucket["month"], freq="M").to_timestamp()
			frequency_rows.append({"Month": month, "Competitor": site, "Articles": bucket["articles"]})
			if bucket["average_word_count"] is not None:
				length_rows.append({"Month": month, "Competitor": site, "Average words": bucket["average_word_count"]})
	figures = {}
	if frequency_rows:
		frequency_figure = px.line(
			pd.DataFrame(frequency_rows), x="Month", y="Articles", color="Competitor", markers=True,
			title="Monthly Publishing Frequency",
			labels={"Articles": "Articles published"},
			color_discrete_sequence=px.colors.qualitative.Vivid,
		)
		figures["Monthly Publishing Frequency"] = frequency_figure
	if length_rows:
		length_figure = px.line(
			pd.DataFrame(length_rows), x="Month", y="Average words", color="Competitor", markers=True,
			title="Average Article Length by Month",
			labels={"Average words": "Words per article"},
			color_discrete_sequence=px.colors.qualitative.Prism,
		)
		figures["Average Article Length by Month"] = length_figure
	return figures, {
		"articles": article_total,
		"average_word_count": word_count_sum / word_count_n if word_count_n else None,
		"word_count_n": word_count_n,
		"competitors": len(summaries),
		"summaries": summaries,
	}


def build_editorial_category_chart(metrics_cache, selected_sites, dimension, analysis_time_range="All data", period_end_date=None):
	_, analytics = build_editorial_metric_charts(metrics_cache, selected_sites, analysis_time_range, period_end_date)
	counts = {}
	for summary in analytics["summaries"].values():
		for category, count in summary["category_counts"].get(dimension, {}).items():
			counts[category] = counts.get(category, 0) + count
	if not counts:
		return None
	items = sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))[:15]
	category_frame = pd.DataFrame([
		{"Category": value[:80] + ("…" if len(value) > 80 else ""), "Full category": value, "Articles": count}
		for value, count in items
	])
	figure = px.bar(
		category_frame, x="Articles", y="Category", orientation="h", color="Articles",
		color_continuous_scale="Viridis", hover_data={"Full category": True},
		title=f"Most Frequent {dimension}",
	)
	figure.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Articles")
	return figure


def build_dashboard_topic_chart(frame, dimension="Topics", selected_site="All websites"):
	topic_data = frame if selected_site == "All websites" else frame[frame["Sito"] == selected_site]
	topic_values = explode_category_values(topic_data[dimension]).value_counts().head(15).sort_values()
	if topic_values.empty:
		return None
	topic_frame = topic_values.rename_axis(dimension).reset_index(name="Articles")
	topic_figure = px.bar(
		topic_frame, x="Articles", y=dimension, orientation="h", color="Articles",
		color_continuous_scale="Viridis", title=f"Most Frequent {dimension}",
	)
	topic_figure.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Articles")
	return topic_figure


def build_dashboard_charts(frame, dimension="Topics", selected_site="All websites", include_topics=True):
	figures = {}
	if frame.empty:
		return figures

	funnel = frame[["Sito", "Funnel stage"]].dropna(subset=["Sito"]).copy().rename(columns={"Sito": "Website"})
	funnel["Funnel stage"] = funnel["Funnel stage"].map(normalize_funnel_stage)
	funnel = funnel.explode("Funnel stage").dropna(subset=["Funnel stage"])
	funnel = funnel[funnel["Funnel stage"].isin(["TOFU", "MOFU", "BOFU"])]
	if not funnel.empty:
		funnel_counts = funnel.groupby(["Website", "Funnel stage"]).size().reset_index(name="Articles")
		funnel_counts["Percentage"] = (
			funnel_counts["Articles"] / funnel_counts.groupby("Website")["Articles"].transform("sum") * 100
		)
		funnel_figure = px.bar(
			funnel_counts, x="Website", y="Percentage", color="Funnel stage",
			category_orders={"Funnel stage": ["TOFU", "MOFU", "BOFU"]}, barmode="stack",
			color_discrete_sequence=px.colors.qualitative.Vivid,
			title="Funnel Stage Distribution by Website",
			labels={"Percentage": "Articles (%)", "Website": "Website"},
			hover_data={"Articles": True, "Percentage": ":.1f"},
		)
		funnel_figure.update_layout(yaxis_range=[0, 100], yaxis_ticksuffix="%", legend_title_text="Funnel stage")
		figures["Funnel Stage Distribution by Website"] = funnel_figure

	if include_topics:
		if topic_figure := build_dashboard_topic_chart(frame, dimension, selected_site):
			figures[f"Most Frequent {dimension}"] = topic_figure

	lengths = frame[["Sito"]].copy().rename(columns={"Sito": "Website"})
	lengths["Article length"] = article_length_series(frame)
	lengths = lengths.dropna(subset=["Article length"])
	if not lengths.empty:
		length_figure = px.box(
			lengths, x="Website", y="Article length", color="Website", points="outliers",
			color_discrete_sequence=px.colors.qualitative.Prism, title="Article Length Comparison",
		)
		length_figure.update_layout(showlegend=False, xaxis_title="Website", yaxis_title="Word count")
		figures["Article Length Comparison"] = length_figure

	dated_articles = frame.dropna(subset=["Date published"]).copy()
	if not dated_articles.empty:
		dated_articles["Publication month"] = (
			dated_articles["Date published"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
		)
		monthly_publications = (
			dated_articles.groupby(["Publication month", "Sito"], as_index=False)
			.size().rename(columns={"Sito": "Website", "size": "Articles"})
		)
		publication_figure = px.line(
			monthly_publications, x="Publication month", y="Articles", color="Website", markers=True,
			color_discrete_sequence=px.colors.qualitative.Vivid,
			title="Editorial Publishing Velocity",
			labels={"Publication month": "Month ending", "Articles": "Articles published"},
		)
		month_end_ticks = monthly_publications["Publication month"].drop_duplicates().sort_values()
		last_month_end = month_end_ticks.iloc[-1]
		publication_figure.update_xaxes(
			tickmode="array", tickvals=month_end_ticks.tolist(), ticktext=month_end_ticks.dt.strftime("%b %Y").tolist(),
		)
		publication_figure.update_layout(
			title=f"Editorial Publishing Velocity (through {last_month_end:%B %d, %Y})",
			legend_title_text="Website",
		)
		figures["Editorial Publishing Velocity"] = publication_figure
	return figures


def render_dashboard(frame):
	if frame.empty:
		st.info("Add valid CSV files to the `data/` folder to view the dashboard.")
		return

	metric_columns = st.columns(3)
	metric_columns[0].metric("Websites", frame["Sito"].nunique())
	metric_columns[1].metric("Articles", len(frame))
	metric_columns[2].metric("Unique topics", frame["Topics"].dropna().nunique())

	figures = build_dashboard_charts(frame, include_topics=False)

	if figure := figures.get("Funnel Stage Distribution by Website"):
		st.plotly_chart(figure, width="stretch")
	else:
		st.info("No funnel stage data is available.")
	st.divider()

	sites = sorted(frame["Sito"].dropna().astype(str).unique(), key=str.casefold)
	dimension = st.selectbox(
		"Analyze by",
		["Topics", "Post type", "Vertical", "Buyer persona", "Strategic initiative", "Features Mentioned", "Target audience"],
		key="topic_dimension",
	)
	selected_site = st.selectbox("Website for most frequent values", ["All websites"] + sites)
	if figure := build_dashboard_topic_chart(frame, dimension, selected_site):
		st.plotly_chart(figure, width="stretch")
	else:
		st.info(f"No {dimension.lower()} values are available for this selection.")
	st.divider()

	if figure := figures.get("Article Length Comparison"):
		st.plotly_chart(figure, width="stretch")
	else:
		st.info("The `Word count` column does not contain usable numeric values.")
	st.divider()

	if figure := figures.get("Editorial Publishing Velocity"):
		st.plotly_chart(figure, width="stretch")
	else:
		st.info("No valid publication dates are available for the selected range.")



def render_analytics(frame, metrics_cache, analysis_time_range, period_end_date):
	st.subheader("Editorial performance")
	if frame.empty:
		st.info("No articles match the selected competitors and time range.")
		return
	sites = sorted(frame["Sito"].dropna().astype(str).unique(), key=str.casefold)
	selected_competitor = st.selectbox("Competitor", ["All competitors"] + sites, key="analytics_competitor")
	selected_sites = sites if selected_competitor == "All competitors" else [selected_competitor]
	figures, summary = build_editorial_metric_charts(
		metrics_cache, selected_sites, analysis_time_range, period_end_date,
	)
	with st.container(horizontal=True):
		st.metric("Articles in scope", f"{summary['articles']:,}")
		st.metric("Competitors in scope", summary["competitors"])
		average = summary["average_word_count"]
		st.metric("Average article length", f"{average:,.0f} words" if average is not None else "N/A")
	st.caption(f"Analysis range: {analysis_time_range} · Cache generated: {metrics_cache.get('generated_at', 'unknown')}")
	for chart_title, figure in figures.items():
		st.plotly_chart(figure, width="stretch", alt=chart_title, key=f"analytics_{chart_title}")

	dimension = st.selectbox("Content dimension", EDITORIAL_METRIC_FIELDS, key="analytics_dimension")
	category_figure = build_editorial_category_chart(
		metrics_cache, selected_sites, dimension, analysis_time_range, period_end_date,
	)
	if category_figure is not None:
		st.plotly_chart(category_figure, width="stretch", alt=f"Most frequent {dimension}", key="analytics_category_chart")
	else:
		st.info(f"No populated values for {dimension} in the selected scope.")


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


def create_report_item(content, frame, title=None, charts=None):
	created = datetime.now(timezone.utc)
	return {
		"id": str(uuid4()),
		"title": title or f"Content Strategy Report - {created:%Y-%m-%d %H:%M UTC}",
		"content": content,
		"charts": charts or {},
		"created_at": created.isoformat(),
		"scope": {"files": sorted(frame["_source_file"].dropna().astype(str).unique(), key=str.casefold),
			"time_range": analysis_time_range, "articles": len(frame)},
	}


def report_export(item, file_format):
	details = export_scope_details(item) + f"\nArticles analyzed: {item.get('scope', {}).get('articles', 'Not recorded')}"
	if file_format == "HTML":
		charts_html = "".join(
			pio.to_html(pio.from_json(chart_json), full_html=False, include_plotlyjs="cdn" if index == 0 else False)
			for index, chart_json in enumerate(item.get("charts", {}).values())
		)
		return (
			'<!doctype html><html lang="en"><head><meta charset="utf-8">'
			'<meta name="viewport" content="width=device-width, initial-scale=1">'
			f'<title>{escape(item["title"])}</title><style>'
			'body{font-family:Georgia,serif;max-width:900px;margin:40px auto;padding:0 24px;line-height:1.6;}'
			'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;}'
			f'</style></head><body><h1>{escape(item["title"])}</h1><pre>{escape(details)}</pre>'
			f'<pre>{escape(item["content"])}</pre>{charts_html}</body></html>'
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


def render_report_visuals(item, key_prefix):
	for index, (title, chart_json) in enumerate(item.get("charts", {}).items()):
		st.subheader(title)
		st.plotly_chart(pio.from_json(chart_json), width="stretch", key=f"{key_prefix}_chart_{index}")


@st.dialog("Saved report", width="large")
def show_saved_report(item):
	st.subheader(item["title"])
	st.caption(export_scope_details(item))
	st.markdown(item["content"])
	render_report_visuals(item, "saved_report")
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


def select_article_context_fields(prompt):
	field_patterns = [
		("Topics", r"\btopics?\b|\bthemes?\b|\bargomenti\b|\btemi\b"),
		("Vertical", r"\bverticals?\b|\bindustr(?:y|ies)\b|\bsectors?\b|\bsettori\b"),
		("Buyer persona", r"\bbuyer personas?\b|\bpersonas?\b|\bpubblici target\b"),
		("Sales pitch", r"\bsales pitches?\b|\bvalue propositions?\b|\bpitches?\b|\bproposta di vendita\b"),
		("Strategic initiative", r"\bstrategic initiatives?\b|\binitiatives?\b|\biniziative strategiche\b"),
		("Features Mentioned", r"\bfeatures?\b|\bcapabilities\b|\btools?\b|\bintegrations?\b|\bfunzionalità\b"),
		("Main Pain Point", r"\bpain points?\b|\bchallenges?\b|\bproblems?\b|\bissues?\b|\bpunti dolenti\b|\bproblemi\b"),
		("Solution", r"\bsolutions?\b|\bfix(?:es|ed|ing)?\b|\bresolve\b|\baddress(?:es|ed|ing)?\b|\bsoluzioni\b"),
		("Word count", r"\bword count\b|\barticle length\b|\bconteggio parole\b"),
		("Funnel stage", r"\bfunnel stages?\b|\btofu\b|\bmofu\b|\bbofu\b|\bfase funnel\b"),
		("Target audience", r"\btarget audiences?\b|\baudiences?\b|\bwho is it for\b|\bpubblico target\b|\baudience target\b"),
		("Date published", r"\bpublication date\b|\bdate published\b|\bdata di pubblicazione\b"),
		("Post type", r"\bpost types?\b|\bcontent formats?\b|\btipo di post\b"),
	]
	matched_fields = [field for field, pattern in field_patterns if re.search(pattern, prompt, flags=re.IGNORECASE)]
	if not matched_fields:
		return ["Title", "Topics", "Summary"]
	fields = ["Title", *matched_fields]
	if re.search(r"\b(explain|describe|summari[sz]e|compare|comparison|why|how|analy[sz]e|analysis|recommend)\b", prompt, flags=re.IGNORECASE):
		fields.append("Summary")
	return list(dict.fromkeys(fields))


def article_context(row, prompt):
	return "\n".join(
		f"{field}: {clean_text(row.get(field))}"
		for field in select_article_context_fields(prompt)
		if clean_text(row.get(field))
	)


def is_feature_search_query(prompt):
	return bool(re.search(
		r"\bfeatures?\b|\bcapabilities\b|\bintegrations?\b|\btools?\b|\bproduct functionality\b|"
		r"\bfunzionalità\b|\bintegrazioni\b|\bstrumenti\b|\bfeature introdotte\b|\bnew features?\b",
		prompt, flags=re.IGNORECASE,
	))


def prompt_month_window(prompt):
	numbers = {
		"one": 1, "two": 2, "three": 3, "four": 4, "six": 6, "twelve": 12,
		"uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "sei": 6, "dodici": 12,
	}
	patterns = (
		r"\b(?:last|past|previous)\s+(\d+|one|two|three|four|six|twelve)\s+months?\b",
		r"\b(?:in the last|within the last)\s+(\d+|one|two|three|four|six|twelve)\s+months?\b",
		r"\b(?:negli ultimi|nelle ultime|ultimi|ultime)\s+(\d+|uno|una|due|tre|quattro|sei|dodici)\s+mesi\b",
	)
	for pattern in patterns:
		match = re.search(pattern, prompt, flags=re.IGNORECASE)
		if match:
			value = match.group(1).casefold()
			return int(value) if value.isdigit() else numbers[value]
	return None


def filter_frame_for_prompt_month_window(frame, prompt, reference_end_date=None):
	month_count = prompt_month_window(prompt)
	if month_count is None or "Date published" not in frame:
		return frame
	dates = pd.to_datetime(frame["Date published"], errors="coerce")
	valid_dates = dates.dropna()
	if valid_dates.empty:
		return frame.iloc[0:0].copy()
	end_date = pd.to_datetime(reference_end_date, errors="coerce") if reference_end_date is not None else valid_dates.max()
	if pd.isna(end_date):
		end_date = valid_dates.max()
	end_date = min(end_date.normalize(), valid_dates.max().normalize())
	start_date = end_date.replace(day=1) - pd.DateOffset(months=month_count - 1)
	end_exclusive = end_date + pd.Timedelta(days=1)
	return frame.loc[dates.ge(start_date) & dates.lt(end_exclusive)].copy()


def is_current_relevant_result(document_id, document, distance, current_documents):
	return (
		document_id in current_documents
		and document == current_documents[document_id]
		and distance is not None
		and distance <= MAX_RETRIEVAL_COSINE_DISTANCE
	)


def generate_chat_answer(prompt, collection, frame=None, report_mode=False, metrics_context=None, reference_end_date=None):
	source_frame = frame if frame is not None else dataset
	feature_search = is_feature_search_query(prompt)
	if feature_search:
		source_frame = filter_frame_for_prompt_month_window(source_frame, prompt, reference_end_date)
	metrics_context = metrics_context if metrics_context is not None else editorial_metrics_context(source_frame, prompt)
	metrics_only = is_csv_metrics_only_question(prompt, metrics_context) and not feature_search
	if metrics_only:
		client = get_genai_client(GEMINI_API_KEY)
		topic_counts_context = ""
		include_topic_counts = report_mode and bool(re.search(
			r"\btopic counts?\b|\btopics? (?:by|per|for each|frequency|distribution|count)|"
			r"\btheme counts?\b|\bconteggio (?:dei )?topics?\b|\bfrequenza dei temi\b",
			prompt, flags=re.IGNORECASE,
		))
		if include_topic_counts:
			topic_counts_context = "\n\n" + topic_frequency_context(source_frame)
		answer_prompt = (
			"You are a data analyst. Answer in English or Italian to match the user's question; use US English for an "
			"English question and default to US English for other languages. Use only the complete calculated CSV metrics below. "
			"These metrics cover all selected rows, not a retrieved sample. Do not claim that publication dates, "
			"article lengths, or topic trends are unavailable when the corresponding calculation is supplied. "
			"Do not invent explanations for trends. Cite calculations as [Dataset metrics] and exact topic-frequency "
			f"counts as [Dataset counts].\n\nQUESTION:\n{prompt}\n\n{metrics_context}{topic_counts_context}"
		)
		answer, output_incomplete = generate_complete_response(client, answer_prompt)
		if output_incomplete:
			answer += "\n\n> The response may still be incomplete because it reached the output limit. Use Re-run to request another attempt."
		if (
			not answer.strip()
			or "[dataset metrics]" not in answer.lower()
			or (include_topic_counts and "[dataset counts]" not in answer.lower())
		):
			return "The selected CSV data does not provide enough verified metrics to answer this question.", []
		return answer, []

	current_rows = {}
	retrieved_documents = []
	client = get_genai_client(GEMINI_API_KEY)
	if not metrics_only:
		for row_number, row in source_frame.iterrows():
			current_rows[f"{clean_text(row['_source_file'])}::{row_number}"] = row
		current_documents = {document_id: article_document(row) for document_id, row in current_rows.items()}
		if not current_documents or collection is None:
			return None
		indexed_count = collection.count()
		if not indexed_count:
			return None
		query_embedding = embed_texts(client, [prompt], "RETRIEVAL_QUERY")[0]
		mentioned_sites = [
			site
			for site in sorted(source_frame["Sito"].dropna().astype(str).unique(), key=len, reverse=True)
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
					if is_current_relevant_result(document_id, document, distance, current_documents):
						site_candidates.append((distance, document_id, metadata))
				site_candidates.sort(key=lambda item: item[0])
				candidates.extend(site_candidates[:per_site])
			candidates.sort(key=lambda item: item[0])
			retrieved_documents = [(item[1], item[2]) for item in candidates[:CHAT_RESULT_COUNT]]
		else:
			results = collection.query(
				query_embeddings=[query_embedding],
				n_results=indexed_count,
				include=["documents", "metadatas", "distances"],
			)
			retrieved_documents = [
				(document_id, metadata)
				for document_id, document, metadata, distance in zip(
					results.get("ids", [[]])[0],
					results.get("documents", [[]])[0],
					results.get("metadatas", [[]])[0],
					results.get("distances", [[]])[0],
				)
				if is_current_relevant_result(document_id, document, distance, current_documents)
			][:CHAT_RESULT_COUNT]
		if not retrieved_documents:
			return None
	sources = []
	context_blocks = []
	for index, (document_id, metadata) in enumerate(retrieved_documents, start=1):
		metadata = metadata or {}
		sources.append(metadata)
		article_text = (
			current_documents[document_id]
			if feature_search
			else article_context(current_rows[document_id], prompt)
		)
		context_blocks.append(
			f"[Source {index}] Website: {metadata.get('Sito', '')}; "
			f"Title: {metadata.get('Title', '')}; URL: {metadata.get('URL', '')}\n"
			f"{article_text}"
		)
	additional_context_parts = []
	if metrics_context:
		additional_context_parts.append(metrics_context)
	include_topic_counts = report_mode and bool(re.search(
		r"\btopic counts?\b|\btopics? (?:by|per|for each|frequency|distribution|count)|"
		r"\btheme counts?\b|\bconteggio (?:dei )?topics?\b|\bfrequenza dei temi\b",
		prompt, flags=re.IGNORECASE,
	))
	if include_topic_counts:
		additional_context_parts.append(topic_frequency_context(source_frame))
	additional_context = "\n\n".join(additional_context_parts)

	if report_mode:
		answer_guidance = (
			"You are a content strategy analyst writing a concise, report-ready answer. "
			"Respond in English or Italian to match the user's question. For any other language, default to US English. "
			"Do not browse the web or use outside knowledge. Treat the supplied dataset and retrieved "
			"sources and computed CSV metrics as the only facts available. If the evidence does not directly answer the question, "
			"say that the selected data is insufficient instead of guessing. Cite exact dataset counts as "
			"[Dataset counts], calculated editorial metrics as [Dataset metrics], and qualitative facts with valid [Source N] references. "
			"Do not provide article-title lists. Prefer useful counts or percentages by website. "
			"When the complete Topics-value frequency table is supplied, use it for counts about topics; do not infer "
			"whole-dataset counts from the retrieved source sample. If the table cannot support an exact "
			"count, say so clearly. Do not present unsupported assumptions as facts."
		)
	else:
		answer_guidance = (
			"You are a content strategy analyst. Respond in English or Italian to match the user's question, with a "
			"detailed, practical comparison. For any other language, default to US English. "
			"Use only provided article sources and computed CSV metrics. Distinguish facts from inferences, "
			"do not claim that CSV metrics are unavailable when a computed metrics block is supplied, "
			"and cite source-based claims as [Source 1] and computed metrics as [Dataset metrics]."
		)
	context_label = "\n\nRETRIEVED ARTICLE CONTEXT:\n" + "\n\n".join(context_blocks) if context_blocks else ""
	answer_prompt = (
		f"{answer_guidance}\n\nQUESTION:\n{prompt}"
		+ context_label
		+ ("\n\n" + additional_context if additional_context else "")
	)
	answer, output_incomplete = generate_complete_response(client, answer_prompt)
	if output_incomplete:
		answer += "\n\n> The response may still be incomplete because it reached the output limit. Use Re-run to request another attempt."
		return answer, sources
	source_citations = [int(index) for index in re.findall(r"\[Source\s+(\d+)\]", answer)]
	dataset_metrics_cited = bool(metrics_context) and "[dataset metrics]" in answer.lower()
	dataset_counts_cited = include_topic_counts and "[dataset counts]" in answer.lower()
	if (
		not answer.strip()
		or any(index < 1 or index > len(sources) for index in source_citations)
		or (not source_citations and not dataset_metrics_cited and not dataset_counts_cited)
	):
		return "The selected data does not provide enough verified evidence to answer this question.", []
	return answer, sources


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


def queue_chat_rerun(question_id, question):
	st.session_state["_chat_rerun_request"] = {"question_id": question_id, "question": question}


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
	rerun_request = st.session_state.pop("_chat_rerun_request", None)
	prompt = (
		rerun_request["question"] if rerun_request
		else st.session_state.pop("suggested_chat_prompt", None) or typed_prompt
	)
	if prompt:
		metrics_context = editorial_metrics_context(
			dataset, prompt, editorial_metrics_cache, analysis_time_range, period_end_date,
		)
		metrics_only = is_csv_metrics_only_question(prompt, metrics_context)
		collection = None if metrics_only else get_indexed_collection()
		if not GEMINI_API_KEY:
			st.error("API key not found in the .env file. Check your configuration.")
		elif dataset.empty:
			st.warning("No articles match the selected time range.")
		elif collection is None and not metrics_only:
			st.warning("The search index is missing or uses an outdated content schema. Start Data Ingestion from the sidebar first.")
		else:
			try:
				with st.spinner("Calculating CSV metrics and preparing an answer..." if metrics_only else "Searching sources and preparing a comparative analysis..."):
					result = generate_chat_answer(
						prompt, collection, metrics_context=metrics_context, reference_end_date=period_end_date,
					)
				if result is None:
					if is_feature_search_query(prompt) and prompt_month_window(prompt):
						st.warning("No indexed articles match the requested date window. Try a wider period or verify the selected CSV data.")
					else:
						st.warning("No indexed articles match the selected time range. Run Data Ingestion to refresh the index.")
				else:
					answer, sources = result
					assistant_message = {
						"role": "assistant", "content": answer, "sources": sources,
						"id": str(uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
						"scope": {"files": [path.name for path in selected_files], "time_range": analysis_time_range},
					}
					if rerun_request:
						question_index = next((
							index for index, message in enumerate(st.session_state.messages)
							if message.get("role") == "user" and message.get("id") == rerun_request["question_id"]
						), None)
						if question_index is None:
							st.warning("This question is no longer in the active conversation.")
						else:
							answer_index = question_index + 1
							if answer_index < len(st.session_state.messages) and st.session_state.messages[answer_index].get("role") == "assistant":
								st.session_state.messages[answer_index] = assistant_message
							else:
								st.session_state.messages.insert(answer_index, assistant_message)
					else:
						st.session_state.messages.extend([
							{"role": "user", "content": prompt, "id": str(uuid4())},
							assistant_message,
						])
			except Exception as exc:
				st.error(f"Error while searching or generating a Gemini response: {exc}")

	exchanges = []
	for message in st.session_state.messages:
		if message["role"] == "user":
			message.setdefault("id", str(uuid4()))
			exchanges.append([message])
		elif exchanges:
			exchanges[-1].append(message)
	for exchange in reversed(exchanges):
		for message in exchange:
			with st.chat_message(message["role"]):
				if message["role"] == "user":
					question_column, action_column = st.columns([12, 1], vertical_alignment="top", gap="small")
					with question_column:
						st.markdown(message["content"])
					with action_column:
						st.button(
							"", icon=":material/replay:",
							key=f"rerun_chat_{message['id']}",
							help="Generate a new answer to this question",
							on_click=queue_chat_rerun, args=(message["id"], message["content"]),
						)
				else:
					st.markdown(message["content"])
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
					report_text, output_incomplete = generate_complete_response(client, prompt)
					if output_incomplete:
						report_text += "\n\n> This report may be incomplete because it reached the output limit. Generate it again to retry."
					st.session_state.strategy_report = create_report_item(report_text or "Gemini did not return a text report.", frame)
			except Exception as exc:
				st.error(f"Error while generating the report: {exc}")

	if report := st.session_state.get("strategy_report"):
		if isinstance(report, str):
			report = create_report_item(report, frame)
			st.session_state.strategy_report = report
		st.markdown(report["content"])
		render_report_actions(report)


def custom_report_content(title, answers, frame, chart_titles):
	lines = [
		f"# {title}",
		"",
		f"Websites analyzed: {frame['Sito'].nunique()}",
		f"Articles analyzed: {len(frame)}",
		f"Analysis time range: {analysis_time_range}",
		"",
		"## Custom analysis",
	]
	for index, answer in enumerate(answers, start=1):
		lines.extend(["", f"### Question {index}: {answer['question']}", "", answer["answer"]])
		sources = answer.get("sources", [])
		if sources:
			lines.extend(["", "Sources:"])
			for source_index, source in enumerate(sources, start=1):
				title_text = clean_text(source.get("Title")) or "Untitled article"
				site = clean_text(source.get("Sito")) or "Website"
				url = str(source.get("URL", "")).replace("(", "%28").replace(")", "%29")
				label = f"Source {source_index}: {site} - {title_text}"
				lines.append(f"- [{label}]({url})" if url else f"- {label}")
	if chart_titles:
		lines.extend(["", "## Included charts", ""])
		lines.extend(f"- {chart_title}" for chart_title in chart_titles)
	return "\n".join(lines)


def remove_custom_report_question(question_id):
	st.session_state.custom_report_answers = [
		answer for answer in st.session_state.get("custom_report_answers", []) if answer["id"] != question_id
	]
	st.session_state.pop(f"custom_report_include_{question_id}", None)


def render_custom_report(frame):
	st.session_state.setdefault("custom_report_answers", [])
	st.subheader("Ask questions")
	st.caption("Answers use the selected websites and time range. Topic counts are computed from all selected articles.")

	with st.form("custom_report_question_form", clear_on_submit=True):
		question = st.text_area(
			"Question", max_chars=1000, key="custom_report_question_input",
			placeholder="How does each competitor use AI in its content strategy?",
		)
		question_submitted = st.form_submit_button("Generate answer", icon=":material/auto_awesome:")

	if question_submitted:
		question_text = question.strip()
		metrics_context = editorial_metrics_context(
			frame, question_text, editorial_metrics_cache, analysis_time_range, period_end_date,
		)
		metrics_only = is_csv_metrics_only_question(question_text, metrics_context)
		collection = None if metrics_only else get_indexed_collection()
		if not question.strip():
			st.warning("Enter a question first.")
		elif not GEMINI_API_KEY:
			st.error("API key not found in the .env file. Check your configuration.")
		elif frame.empty:
			st.warning("No articles match the selected websites and time range.")
		elif collection is None and not metrics_only:
			st.warning("The search index is missing or outdated. Start Data Ingestion from the sidebar first.")
		else:
			try:
				with st.spinner("Calculating CSV metrics and preparing the report answer..." if metrics_only else "Gemini is preparing a report-ready answer..."):
					result = generate_chat_answer(
						question_text, collection, frame=frame, report_mode=True,
						metrics_context=metrics_context, reference_end_date=period_end_date,
					)
				if result is None:
					st.warning("No indexed articles match this question. Run data ingestion to refresh the index.")
				else:
					answer, sources = result
					st.session_state.custom_report_answers.append({
						"id": str(uuid4()), "question": question.strip(), "answer": answer, "sources": sources,
						"include": True, "created_at": datetime.now(timezone.utc).isoformat(),
					})
			except Exception as exc:
				st.error(f"Error while preparing the answer: {exc}")

	answers = st.session_state.custom_report_answers
	selected_answers = []
	for index, answer in enumerate(answers, start=1):
		with st.container(border=True):
			st.markdown(f"**Question {index}:** {answer['question']}")
			st.markdown(answer["answer"])
			render_sources(answer.get("sources", []), key_prefix=f"custom_report_{answer['id']}")
			include = st.checkbox(
				"Include in report", value=answer.get("include", True),
				key=f"custom_report_include_{answer['id']}",
			)
			answer["include"] = include
			if include:
				selected_answers.append(answer)
			st.button(
				"Remove question", icon=":material/delete:", key=f"custom_report_remove_{answer['id']}",
				on_click=remove_custom_report_question, args=(answer["id"],),
			)

	st.divider()
	st.subheader("Charts")
	dimensions = ["Topics", "Post type", "Vertical", "Buyer persona", "Strategic initiative", "Features Mentioned", "Target audience"]
	dimension = st.selectbox("Break down by", dimensions, key="custom_report_dimension")
	sites = sorted(frame["Sito"].dropna().astype(str).unique(), key=str.casefold) if not frame.empty else []
	selected_site = st.selectbox("Website for frequency chart", ["All websites"] + sites, key="custom_report_site")
	figures = build_dashboard_charts(frame, dimension, selected_site)
	chart_titles = list(figures)
	selected_chart_titles = st.multiselect(
		"Include charts", chart_titles, default=chart_titles, key="custom_report_charts",
	)
	if not chart_titles:
		st.info("No charts are available for the selected data.")

	st.text_input("Report title", key="custom_report_title", max_chars=120)
	if st.button("Generate custom report", type="primary", icon=":material/description:", disabled=not selected_answers):
		title = st.session_state.custom_report_title.strip() or "Custom Content Strategy Report"
		selected_figures = {name: figures[name].to_json() for name in selected_chart_titles}
		content = custom_report_content(title, selected_answers, frame, selected_chart_titles)
		st.session_state.custom_strategy_report = create_report_item(
			content, frame, title=title, charts=selected_figures,
		)

	if report := st.session_state.get("custom_strategy_report"):
		st.divider()
		st.subheader(report["title"])
		st.caption(export_scope_details(report))
		st.markdown(report["content"])
		render_report_visuals(report, "custom_report")
		render_report_actions(report, key_prefix="custom_report")


files = scan_csv_files()
all_dataset, all_csv_errors = load_dataset(files)
editorial_metrics_cache = load_or_build_editorial_metrics_cache(all_dataset, files)
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
		progress_status = st.empty()
		ingestion_started = time.monotonic()
		try:
			client = get_genai_client(GEMINI_API_KEY)
			indexed_count = ingest_dataset(
				selected_dataset,
				client,
				lambda value: progress_bar.progress(value, text="Indexing selected articles..."),
				progress_status.caption,
			)
			progress_status.caption("Refreshing cached editorial metrics...")
			editorial_metrics_cache = save_editorial_metrics_cache(all_dataset, files)
			progress_bar.empty()
			progress_status.empty()
			st.success(f"Ingestion complete: {indexed_count} articles indexed.")
		except Exception as exc:
			progress_bar.empty()
			progress_status.error(
				f"Ingestion failed after {format_elapsed_duration(time.monotonic() - ingestion_started)}."
			)
			st.error(f"Error during ingestion: {exc}")

max_date = selected_dataset["Date published"].max() if not selected_dataset.empty else pd.NaT
previous_month_end = pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)
period_end_date = previous_month_end if pd.isna(max_date) else min(max_date.normalize(), previous_month_end)

active_analysis_scope = (tuple(path.name for path in selected_files), analysis_time_range)
if st.session_state.get("_active_analysis_scope") != active_analysis_scope:
	st.session_state.messages = []
	st.session_state.pop("strategy_report", None)
	st.session_state.custom_report_answers = []
	st.session_state.pop("custom_strategy_report", None)
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

dashboard_tab, chat_tab, report_tab, custom_report_tab, analytics_tab = st.tabs(
	["📊 Quantitative Dashboard", "🤖 AI Assistant & Comparative Chat", "📄 Strategic Report Generator", "🧩 Build custom report", "📈 Analytics"],
	key="content_views", on_change="rerun",
)

with dashboard_tab:
	render_dashboard(dataset)

with chat_tab:
	render_chat()

with report_tab:
	render_report(dataset)

with custom_report_tab:
	render_custom_report(dataset)

with analytics_tab:
	render_analytics(dataset, editorial_metrics_cache, analysis_time_range, period_end_date)
