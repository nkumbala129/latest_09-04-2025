import streamlit as st
import json
import re
import requests
import snowflake.connector
import pandas as pd
from snowflake.snowpark import Session
from typing import Any, Dict, List, Optional, Tuple
import plotly.express as px  # Added for interactive visualizations

# Snowflake/Cortex Configuration
HOST = "GNB14769.snowflakecomputing.com"
DATABASE = "CORTEX_SEARCH_TUTORIAL_DB"
SCHEMA = "PUBLIC"
STAGE = "CC_STAGE"
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000  # in milliseconds
CORTEX_SEARCH_SERVICES = "CORTEX_SEARCH_TUTORIAL_DB.PUBLIC.BAYREN2"

# Single semantic model
SEMANTIC_MODEL = '@"CORTEX_SEARCH_TUTORIAL_DB"."PUBLIC"."MULTIFAMILYSTAGE"/Green_Residences.yaml'

# Streamlit Page Config
st.set_page_config(
    page_title="Welcome to Cortex AI Assistant",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.CONN = None
    st.session_state.snowpark_session = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
# Initialize chart selection persistence
if "chart_x_axis" not in st.session_state:
    st.session_state.chart_x_axis = None
if "chart_y_axis" not in st.session_state:
    st.session_state.chart_y_axis = None
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Bar Chart"
# Initialize query and results persistence
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "current_sql" not in st.session_state:
    st.session_state.current_sql = None
if "current_summary" not in st.session_state:
    st.session_state.current_summary = None

# Hide Streamlit branding
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Authentication logic
if not st.session_state.authenticated:
    st.title("Welcome to Snowflake Cortex AI")
    st.markdown("Please login to interact with your data")

    st.session_state.username = st.text_input("Enter Snowflake Username:", value=st.session_state.username)
    st.session_state.password = st.text_input("Enter Password:", type="password")

    if st.button("Login"):
        try:
            conn = snowflake.connector.connect(
                user=st.session_state.username,
                password=st.session_state.password,
                account="GNB14769",
                host=HOST,
                port=443,
                warehouse="CORTEX_SEARCH_TUTORIAL_WH",
                role="DEV_BR_CORTEX_AI_ROLE",
                database=DATABASE,
                schema=SCHEMA,
            )
            st.session_state.CONN = conn

            snowpark_session = Session.builder.configs({
                "connection": conn
            }).create()
            st.session_state.snowpark_session = snowpark_session

            with conn.cursor() as cur:
                cur.execute(f"USE DATABASE {DATABASE}")
                cur.execute(f"USE SCHEMA {SCHEMA}")
                cur.execute("ALTER SESSION SET TIMEZONE = 'UTC'")
                cur.execute("ALTER SESSION SET QUOTED_IDENTIFIERS_IGNORE_CASE = TRUE")

            st.session_state.authenticated = True
            st.success("Authentication successful! Redirecting...")
            st.rerun()

        except Exception as e:
            st.error(f"Authentication failed: {e}")
else:
    session = st.session_state.snowpark_session

    # Utility Functions
    def run_snowflake_query(query):
        try:
            if not query:
                st.warning("⚠️ No SQL query generated.")
                return None
            df = session.sql(query)
            data = df.collect()
            if not data:
                return None
            columns = df.schema.names
            result_df = pd.DataFrame(data, columns=columns)
            return result_df
        except Exception as e:
            st.error(f"❌ SQL Execution Error: {str(e)}")
            return None

    def is_structured_query(query: str):
        structured_patterns = [
            r'\b(county|from|where|group by|order by|join|sum|count|avg|max|min|least|highest|which)\b',
            r'\b(total|revenue|sales|profit|projects|jurisdiction|month|year|energy savings|kwh)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in structured_patterns)

    def is_complete_query(query: str):
        complete_patterns = [r'\b(generate|write|create|describe|explain)\b']
        return any(re.search(pattern, query.lower()) for pattern in complete_patterns)

    def is_summarize_query(query: str):
        summarize_patterns = [r'\b(summarize|summary|condense)\b']
        return any(re.search(pattern, query.lower()) for pattern in summarize_patterns)

    def is_question_suggestion_query(query: str):
        suggestion_patterns = [
            r'\b(what|which|how)\b.*\b(questions|type of questions|queries)\b.*\b(ask|can i ask|pose)\b',
            r'\b(give me|show me|list)\b.*\b(questions|examples|sample questions)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in suggestion_patterns)

    def complete(prompt, model="mistral-large"):
        try:
            prompt = prompt.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{prompt}') AS response"
            result = session.sql(query).collect()
            return result[0]["RESPONSE"]
        except Exception as e:
            st.error(f"❌ COMPLETE Function Error: {str(e)}")
            return None

    def summarize(text):
        try:
            text = text.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
            result = session.sql(query).collect()
            return result[0]["SUMMARY"]
        except Exception as e:
            st.error(f"❌ SUMMARIZE Function Error: {str(e)}")
            return None

    def parse_sse_response(response_text: str) -> List[Dict]:
        """Parse SSE response into a list of JSON objects."""
        events = []
        lines = response_text.strip().split("\n")
        current_event = {}
        for line in lines:
            if line.startswith("event:"):
                current_event["event"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                if data_str != "[DONE]":  # Skip the [DONE] marker
                    try:
                        data_json = json.loads(data_str)
                        current_event["data"] = data_json
                        events.append(current_event)
                        current_event = {}  # Reset for next event
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Failed to parse SSE data: {str(e)} - Data: {data_str}")
        return events

    def snowflake_api_call(query: str, is_structured: bool = False):
        payload = {
            "model": "mistral-large",
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            "tools": []
        }
        if is_structured:
            payload["tools"].append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
            payload["tool_resources"] = {"analyst1": {"semantic_model_file": SEMANTIC_MODEL}}
        else:
            payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
            payload["tool_resources"] = {"search1": {"name": CORTEX_SEARCH_SERVICES, "max_results": 1}}

        try:
            resp = requests.post(
                url=f"https://{HOST}{API_ENDPOINT}",
                json=payload,
                headers={
                    "Authorization": f'Snowflake Token="{st.session_state.CONN.rest.token}"',
                    "Content-Type": "application/json",
                },
                timeout=API_TIMEOUT // 1000
            )
            if st.session_state.debug_mode:  # Show debug info only if toggle is enabled
                st.write(f"API Response Status: {resp.status_code}")
                st.write(f"API Raw Response: {resp.text}")
            if resp.status_code < 400:
                if not resp.text.strip():
                    st.error("❌ API returned an empty response.")
                    return None
                return parse_sse_response(resp.text)
            else:
                raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"❌ API Request Failed: {str(e)}")
            return None

    def summarize_unstructured_answer(answer):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|")\s', answer)
        return "\n".join(f"• {sent.strip()}" for sent in sentences[:6])

    def process_sse_response(response, is_structured):
        sql = ""
        search_results = []
        if not response:
            return sql, search_results
        try:
            for event in response:
                if event.get("event") == "message.delta" and "data" in event:
                    delta = event["data"].get("delta", {})
                    content = delta.get("content", [])
                    for item in content:
                        if item.get("type") == "tool_results":
                            tool_results = item.get("tool_results", {})
                            if "content" in tool_results:
                                for result in tool_results["content"]:
                                    if result.get("type") == "json":
                                        result_data = result.get("json", {})
                                        if is_structured and "sql" in result_data:
                                            sql = result_data.get("sql", "")
                                        elif not is_structured and "searchResults" in result_data:
                                            search_results = [sr["text"] for sr in result_data["searchResults"]]
        except Exception as e:
            st.error(f"❌ Error Processing Response: {str(e)}")
        return sql.strip(), search_results

    def generate_result_summary(results):
        """Generate a meaningful summary of the query results using summarize and complete functions."""
        try:
            # Convert DataFrame to a readable string format for summarization
            results_text = results.to_string(index=False)
            initial_summary = summarize(results_text)
            if not initial_summary:
                return "⚠️ Unable to generate an initial summary."
            # Pass the initial summary to complete for a meaningful explanation
            prompt = f"Provide a concise, meaningful summary of the following query results:\n\n{initial_summary}"
            meaningful_summary = complete(prompt)
            if meaningful_summary:
                return meaningful_summary
            else:
                return "⚠️ Unable to generate a meaningful summary."
        except Exception as e:
            return f"⚠️ Summary generation failed: {str(e)}"

    # Visualization Function
    def display_chart_tab(df: pd.DataFrame, prefix: str = "chart"):
        """Allows user to select chart options and displays a chart with unique widget keys."""
        if len(df.columns) < 2:
            st.write("Not enough columns to chart.")
            return

        all_cols = list(df.columns)
        col1, col2, col3 = st.columns(3)

        # Use st.session_state.get(...) to obtain the previously selected values without reassigning them.
        default_x = st.session_state.get(f"{prefix}_x", all_cols[0])
        try:
            x_index = all_cols.index(default_x)
        except ValueError:
            x_index = 0
        x_col = col1.selectbox("X axis", all_cols, index=x_index, key=f"{prefix}_x")

        remaining_cols = [c for c in all_cols if c != x_col]
        default_y = st.session_state.get(f"{prefix}_y", remaining_cols[0])
        try:
            y_index = remaining_cols.index(default_y)
        except ValueError:
            y_index = 0
        y_col = col2.selectbox("Y axis", remaining_cols, index=y_index, key=f"{prefix}_y")

        chart_options = ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Chart", "Histogram Chart"]
        default_type = st.session_state.get(f"{prefix}_type", "Line Chart")
        try:
            type_index = chart_options.index(default_type)
        except ValueError:
            type_index = 0
        chart_type = col3.selectbox("Chart Type", chart_options, index=type_index, key=f"{prefix}_type")

        # Display the chart based on the selected type.
        if chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_line")
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_bar")
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_col, values=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_pie")
        elif chart_type == "Scatter Chart":
            fig = px.scatter(df, x=x_col, y=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_scatter")
        elif chart_type == "Histogram Chart":
            fig = px.histogram(df, x=x_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_hist")

    # UI Logic
    with st.sidebar:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] [data-testid="stButton"] > button {
            background-color: #29B5E8 !important;
            color: white !important;
            font-weight: bold !important;
            width: 100% !important;
            border-radius: 0px !important;
            margin: 0 !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        logo_container = st.container()
        button_container = st.container()
        about_container = st.container()
        help_container = st.container()

        with logo_container:
            logo_url = "https://www.snowflake.com/wp-content/themes/snowflake/assets/img/logo-blue.svg"
            st.image(logo_url, width=200)

        with button_container:
            st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)

        with about_container:
            st.markdown("### About")
            st.write(
                "This application uses **Snowflake Cortex Analyst** to interpret "
                "your natural language questions and generate data insights. "
                "Simply ask a question below to see relevant answers and visualizations."
            )

        with help_container:
            st.markdown("### Help & Documentation")
            st.write(
                "- [User Guide](https://docs.snowflake.com/en/guides-overview-ai-features)  \n"
                "- [Snowflake Cortex Analyst Docs](https://docs.snowflake.com/)  \n"
                "- [Contact Support](https://www.snowflake.com/en/support/)"
            )

    st.title("Cortex AI Assistant")

    # Display the fixed semantic model
    semantic_model_filename = SEMANTIC_MODEL.split("/")[-1]
    st.markdown(f"Semantic Model: `{semantic_model_filename}`")

    st.sidebar.subheader("Sample Questions")
    sample_questions = [
        "What is the Green Residences program?",
        "Show total energy savings by county.",
        "Which county has the highest kWh savings?",
        "How many homes were retrofitted in Alameda County?",
        "What are the thermal savings in San Francisco?",
        "How many active projects are there",
        "What is the average kWh savings across Sonoma and Napa?",
        "Describe the energy savings technologies used in Green Residences."
    ]

    query = st.chat_input("Ask your question...")

    for sample in sample_questions:
        if st.sidebar.button(sample, key=sample):
            query = sample

    # Display persisted query and results if no new query is submitted
    if not query and st.session_state.current_query:
        with st.chat_message("user"):
            st.markdown(st.session_state.current_query)
        with st.chat_message("assistant"):
            if st.session_state.current_results is not None and not st.session_state.current_results.empty:
                st.markdown("**🛠️ Generated SQL Query:**")
                with st.expander("View SQL Query", expanded=False):  # Collapsible SQL
                    st.code(st.session_state.current_sql, language="sql")
                st.markdown("**📝 Summary of Query Results:**")
                st.write(st.session_state.current_summary)
                st.markdown(f"**📊 Query Results ({len(st.session_state.current_results)} rows):**")
                st.dataframe(st.session_state.current_results)
                st.markdown("**📈 Visualization:**")
                display_chart_tab(st.session_state.current_results, prefix=f"chart_{hash(st.session_state.current_query)}")

    if query:
        # Reset chart selections for new query
        st.session_state.chart_x_axis = None
        st.session_state.chart_y_axis = None
        st.session_state.chart_type = "Bar Chart"

        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Cool..Fetching the data..."):
                is_structured = is_structured_query(query)
                is_complete = is_complete_query(query)
                is_summarize = is_summarize_query(query)
                is_suggestion = is_question_suggestion_query(query)  # New check

                if is_suggestion:  # Handle question suggestion request
                    st.markdown("**Here are some questions you can ask me:**")
                    for i, q in enumerate(sample_questions, 1):
                        st.write(f"{i}. {q}")
                    st.info("Feel free to ask any of these or come up with your own related to energy savings, Green Residences, or other programs!")
                    # Store the query but not results, as this is informational
                    st.session_state.current_query = query
                    st.session_state.current_results = None
                    st.session_state.current_sql = None
                    st.session_state.current_summary = None

                elif is_complete:
                    response = complete(query)
                    if response:
                        st.markdown("**✍️ Generated Response:**")
                        st.write(response)
                    else:
                        st.warning("⚠️ Failed to generate a response.")
                elif is_summarize:
                    summary = summarize(query)
                    if summary:
                        st.markdown("**📝 Summary:**")
                        st.write(summary)
                    else:
                        st.warning("⚠️ Failed to generate a summary.")
                elif is_structured:
                    response = snowflake_api_call(query, is_structured=True)
                    sql, _ = process_sse_response(response, is_structured=True)
                    if sql:
                        results = run_snowflake_query(sql)
                        if results is not None and not results.empty:
                            summary = generate_result_summary(results)
                            st.markdown("**🛠️ Generated SQL Query:**")
                            with st.expander("View SQL Query", expanded=False):  # Collapsible SQL
                                st.code(sql, language="sql")
                            st.markdown("**📝 Summary of Query Results:**")
                            st.write(summary)
                            st.markdown(f"**📊 Query Results ({len(results)} rows):**")
                            st.dataframe(results)
                            st.markdown("**📈 Visualization:**")
                            display_chart_tab(results, prefix=f"chart_{hash(query)}")
                            # Store results in session state
                            st.session_state.current_query = query
                            st.session_state.current_results = results
                            st.session_state.current_sql = sql
                            st.session_state.current_summary = summary
                        else:
                            st.warning("⚠️ No data found.")
                    else:
                        st.warning("⚠️ No SQL generated.")
                else:
                    response = snowflake_api_call(query, is_structured=False)
                    _, search_results = process_sse_response(response, is_structured=False)
                    if search_results:
                        raw_result = search_results[0]
                        summary = summarize(raw_result)
                        if summary:
                            st.markdown("**🔍 Here is the Answer to your query:**")
                            st.write(summary)
                            last_sentence = summary.split(".")[-2] if "." in summary else summary
                            st.success(f"✅ Key Insight: {last_sentence.strip()}")
                        else:
                            st.markdown("**🔍 Key Information (Unsummarized):**")
                            st.write(summarize_unstructured_answer(raw_result))
                    else:
                        st.warning("⚠️ No relevant search results found.")
