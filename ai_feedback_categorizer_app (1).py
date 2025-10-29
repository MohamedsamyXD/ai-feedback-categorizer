"""
Streamlit app: AI Feedback Categorizer (Updated for OpenAI v1.0+)
Features:
- Upload Excel/CSV with feedback
- Create/Edit/Delete categories (persisted to categories.json)
- Use OpenAI Embeddings (v1.0+ syntax) to pick the closest category from the list
- Download the categorized file

Requirements:
pip install streamlit openai pandas numpy openpyxl

Run:
export OPENAI_API_KEY="sk-..."
streamlit run ai_feedback_categorizer_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from io import BytesIO
from openai import OpenAI

CATEGORIES_FILE = "categories.json"
EMBEDDING_MODEL = "text-embedding-3-small"

# ----------------- Helpers -----------------

def load_categories():
    if os.path.exists(CATEGORIES_FILE):
        with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_categories(categories):
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)


def compute_embeddings(client, texts):
    """Compute embeddings for a list of texts using OpenAI v1+ API."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def cosine_sim(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def categorize_feedbacks(client, feedbacks, categories):
    if len(categories) == 0:
        raise ValueError("No categories available")

    category_texts = [c.get("name", "") + (" - " + c.get("desc", "") if c.get("desc") else "") for c in categories]
    cat_embs = compute_embeddings(client, category_texts)
    fb_embs = compute_embeddings(client, feedbacks)

    results = []
    for emb in fb_embs:
        sims = [cosine_sim(emb, cemb) for cemb in cat_embs]
        idx = int(np.argmax(sims))
        results.append({
            "category": categories[idx]["name"],
            "score": sims[idx]
        })
    return results

# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="AI Feedback Categorizer", layout="wide")
st.title("AI Feedback Categorizer — choose closest category from your list")

# API key input
st.sidebar.header("OpenAI API Key")
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

key_input = st.sidebar.text_input("Paste your OpenAI API key (or set OPENAI_API_KEY env var)", value=st.session_state.openai_api_key, type="password")
if key_input:
    st.session_state.openai_api_key = key_input

st.sidebar.markdown("---")
st.sidebar.markdown("App stores editable categories in `categories.json` next to this script.")

if "categories" not in st.session_state:
    st.session_state.categories = load_categories()

st.header("Manage categories")
col1, col2 = st.columns([2,3])
with col1:
    new_name = st.text_input("New category name", key="new_cat_name")
    new_desc = st.text_area("Optional description (helps AI match)", key="new_cat_desc", height=80)
    if st.button("Add category"):
        if not new_name.strip():
            st.error("Category name can't be empty")
        else:
            st.session_state.categories.append({"name": new_name.strip(), "desc": new_desc.strip()})
            save_categories(st.session_state.categories)
            st.success(f"Added category: {new_name.strip()}")

with col2:
    if len(st.session_state.categories) == 0:
        st.info("No categories yet — add some on the left.")
    else:
        st.write("**Existing categories (click a row to edit/delete):**")
        df_cats = pd.DataFrame(st.session_state.categories)
        selected = st.selectbox("Select category", options=df_cats.index, format_func=lambda i: df_cats.loc[i, 'name'])
        edit_name = st.text_input("Edit name", value=df_cats.loc[selected, 'name'], key="edit_name")
        edit_desc = st.text_area("Edit description", value=df_cats.loc[selected, 'desc'], key="edit_desc", height=80)
        if st.button("Save changes"):
            st.session_state.categories[selected]['name'] = edit_name.strip()
            st.session_state.categories[selected]['desc'] = edit_desc.strip()
            save_categories(st.session_state.categories)
            st.success("Category updated")
        if st.button("Delete category"):
            cat_nm = st.session_state.categories[selected]['name']
            st.session_state.categories.pop(selected)
            save_categories(st.session_state.categories)
            st.success(f"Deleted {cat_nm}")

st.markdown("---")

st.header("Upload feedback file (Excel or CSV)")
uploaded = st.file_uploader("Upload .xlsx/.csv", type=["xlsx", "csv"]) 

if uploaded is not None:
    try:
        if uploaded.type == "text/csv" or uploaded.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    cols = df.columns.tolist()
    feedback_col = st.selectbox("Which column contains the feedback/comments?", options=cols)
    out_name = st.text_input("Output filename (will be .xlsx)", value=(os.path.splitext(uploaded.name)[0] + "_categorized.xlsx"))

    if st.button("Run categorization"):
        if len(st.session_state.categories) == 0:
            st.error("Please add at least one category before running categorization.")
        elif not st.session_state.openai_api_key:
            st.error("Please provide your OpenAI API key in the sidebar or as OPENAI_API_KEY environment variable.")
        else:
            client = OpenAI(api_key=st.session_state.openai_api_key)
            with st.spinner("Computing embeddings and categorizing... this may take a moment"):
                try:
                    feedbacks = df[feedback_col].astype(str).fillna("").tolist()
                    res = categorize_feedbacks(client, feedbacks, st.session_state.categories)
                    df['Category'] = [r['category'] for r in res]
                    df['CategoryScore'] = [r['score'] for r in res]

                    st.success("Categorization complete")
                    st.dataframe(df.head(200))

                    towrite = BytesIO()
                    df.to_excel(towrite, index=False, engine='openpyxl')
                    towrite.seek(0)
                    st.download_button(label="Download categorized file (.xlsx)", data=towrite, file_name=out_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                except Exception as e:
                    st.error(f"Error during categorization: {e}")
else:
    st.info("Upload an Excel or CSV file to get started.")

st.markdown("---")
st.caption("Built with Streamlit + OpenAI embeddings (v1.0+). The app stores categories in categories.json so you can keep, edit, and reuse them.")
