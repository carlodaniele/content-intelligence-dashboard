import os
import re
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

import chromadb
import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "website_content"
CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 64
CHAT_RESULT_COUNT = 6
REQUIRED_COLUMNS = [
	"URL",
	"Title",
	"Date published",
	"Category",
	"Tags",
	"Topic",
	"Lenght",
	"Funnel stage",
	"Target audience",
	"Summary",
]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Content Intelligence", page_icon="📚", layout="wide")
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
	values = frame["Lenght"].astype("string").str.replace(",", "", regex=False)
	numbers = values.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
	return pd.to_numeric(numbers, errors="coerce")


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
		title = clean_text(row.get("Title"))
		summary = clean_text(row.get("Summary"))
		documents.append(f"Title: {title}\nSummary: {summary}")

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
		metadata={"hnsw:space": "cosine"},
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
		return collection if collection.count() else None
	except Exception:
		return None


def valid_http_url(value):
	parsed = urlparse(clean_text(value))
	return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def render_sources(sources):
	if not sources:
		return
	with st.expander("Sources Used"):
		for index, source in enumerate(sources, start=1):
			title = clean_text(source.get("Title")) or "Untitled article"
			site = clean_text(source.get("Sito"))
			st.markdown(f"**{index}. {title}** · {site}")
			if valid_http_url(source.get("URL")):
				st.link_button("Open article", clean_text(source["URL"]), key=f"source_{index}_{site}_{title[:24]}")
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
			lines.append(f"Average article length: {site_lengths.mean():.1f}; median: {site_lengths.median():.1f}")
		if not site_dates.empty:
			lines.append(f"Publication date range: {site_dates.min().date()} - {site_dates.max().date()}")

		stages = group["Funnel stage"].dropna().astype(str).str.strip().str.upper().value_counts(normalize=True)
		if not stages.empty:
			lines.append("Funnel distribution (%): " + ", ".join(f"{name} {value * 100:.1f}%" for name, value in stages.items()))
		for column in ("Topic", "Category", "Target audience"):
			counts = group[column].dropna().astype(str).str.strip()
			counts = counts[counts.ne("")].value_counts().head(6)
			if not counts.empty:
				lines.append(f"Most frequent {column} values: " + "; ".join(f"{name} ({count})" for name, count in counts.items()))

		dated = group.assign(_report_date=pd.to_datetime(group["Date published"], errors="coerce", utc=True))
		examples = dated.sort_values("_report_date", na_position="first").tail(3)
		for _, article in examples.iterrows():
			title = clean_text(article.get("Title"))
			summary = clean_text(article.get("Summary"))[:350]
			lines.append(f"Example: {title} - {summary}")

	if lengths.notna().any():
		lines.append(f"\nOverall average article length: {lengths.mean():.1f}")
	if published.notna().any():
		lines.append(f"Overall publication date range: {published.min().date()} - {published.max().date()}")
	return "\n".join(lines)


def render_dashboard(frame):
	if frame.empty:
		st.info("Add valid CSV files to the `data/` folder to view the dashboard.")
		return

	site_count = frame["Sito"].nunique()
	article_count = len(frame)
	topic_count = frame["Topic"].dropna().nunique()
	metric_columns = st.columns(3)
	metric_columns[0].metric("Websites", site_count)
	metric_columns[1].metric("Articles", article_count)
	metric_columns[2].metric("Unique topics", topic_count)

	funnel = frame[["Sito", "Funnel stage"]].dropna().copy().rename(columns={"Sito": "Website"})
	funnel["Funnel stage"] = funnel["Funnel stage"].astype(str).str.strip().str.upper()
	funnel = funnel[funnel["Funnel stage"].ne("")]
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
	dimension = st.selectbox("Analyze by", ["Topic", "Category"], key="topic_dimension")
	selected_site = st.selectbox("Website for most frequent topics", ["All websites"] + sites)
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
			title=f"Most Frequent {dimension}s",
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
		length_figure.update_layout(showlegend=False, xaxis_title="Website", yaxis_title="Article length")
		st.plotly_chart(length_figure, width="stretch")
	else:
		st.info("The `Lenght` column does not contain usable numeric values.")
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
	st.divider()

	treemap_data = frame[["Sito", "Category", "Topic"]].copy()
	treemap_data["Category"] = treemap_data["Category"].fillna("").astype(str).str.strip()
	treemap_data["Topic"] = treemap_data["Topic"].fillna("").astype(str).str.strip()
	treemap_data["Category"] = treemap_data["Category"].replace("", "Uncategorized")
	treemap_data["Topic"] = treemap_data["Topic"].replace("", "Unspecified topic")
	treemap_data["Articles"] = 1
	treemap_figure = px.treemap(
		treemap_data,
		path=["Sito", "Category", "Topic"],
		values="Articles",
		color="Articles",
		color_continuous_scale="Viridis",
		title="Content Hierarchy by Website, Category, and Topic",
		labels={"Sito": "Website", "Category": "Category", "Topic": "Topic", "Articles": "Articles"},
	)
	treemap_figure.update_layout(margin=dict(t=55, l=0, r=0, b=0))
	st.plotly_chart(treemap_figure, width="stretch")


def render_chat():
	if "messages" not in st.session_state:
		st.session_state.messages = []

	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])
			if message["role"] == "assistant":
				render_sources(message.get("sources", []))

	if prompt := st.chat_input("Ask a comparative question about the content..."):
		if not GEMINI_API_KEY:
			st.error("API key not found in the .env file. Check your configuration.")
			return

		collection = get_indexed_collection()
		if collection is None:
			st.warning("Database not indexed yet. Start ingestion from the sidebar first.")
			return
		if dataset.empty:
			st.warning("No articles match the selected time range.")
			return

		allowed_document_ids = {
			f"{clean_text(row['_source_file'])}::{row_number}"
			for row_number, row in dataset.iterrows()
		}

		st.session_state.messages.append({"role": "user", "content": prompt})
		with st.chat_message("user"):
			st.markdown(prompt)

		try:
			with st.chat_message("assistant"):
				with st.spinner("Searching sources and preparing a comparative analysis..."):
					client = get_genai_client(GEMINI_API_KEY)
					query_embedding = embed_texts(client, [prompt], "RETRIEVAL_QUERY")[0]
					mentioned_sites = [
						site
						for site in sorted(dataset["Sito"].dropna().astype(str).unique(), key=len, reverse=True)
						if re.search(rf"(?<!\w){re.escape(site)}(?!\w)", prompt, flags=re.IGNORECASE)
					]
					candidates = []
					if mentioned_sites:
						per_site = max(1, CHAT_RESULT_COUNT // len(mentioned_sites))
						for site in mentioned_sites:
							site_candidates = []
							if not collection.count():
								continue
							site_results = collection.query(
								query_embeddings=[query_embedding],
								n_results=collection.count(),
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
							n_results=collection.count(),
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
						st.warning("No indexed articles match the selected time range. Run data ingestion to refresh the index.")
						return
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
					answer = response.text or "Gemini did not return a text response."
					st.markdown(answer)
					render_sources(sources)
			st.session_state.messages.append(
				{"role": "assistant", "content": answer, "sources": sources}
			)
		except Exception as exc:
			error_message = f"Error while searching or generating a Gemini response: {exc}"
			st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})
			st.error(error_message)


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
					st.session_state.strategy_report = response.text or "Gemini did not return a text report."
			except Exception as exc:
				st.error(f"Error while generating the report: {exc}")

	if report := st.session_state.get("strategy_report"):
		st.markdown(report)
		st.download_button(
			"Download Markdown report",
			data=report,
			file_name="content_strategy_report.md",
			mime="text/markdown",
		)


files = scan_csv_files()
all_dataset, all_csv_errors = load_dataset(files)
with st.sidebar:
	st.subheader("Detected CSV files")
	selected_files = []
	if files:
		for csv_file in files:
			if st.checkbox(csv_file.name, value=True, key=f"data_source_{csv_file.name}"):
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

st.title("Content Intelligence")
st.caption("Competitive content analysis, semantic search, and editorial strategy.")
if not GEMINI_API_KEY:
	st.error("API key not found in the .env file. Check your configuration.")
for csv_error in csv_errors:
	st.warning(csv_error)

dashboard_tab, chat_tab, report_tab = st.tabs(
	["📊 Quantitative Dashboard", "🤖 AI Assistant & Comparative Chat", "📄 Strategic Report Generator"]
)

with dashboard_tab:
	render_dashboard(dataset)

with chat_tab:
	render_chat()

with report_tab:
	render_report(dataset)
