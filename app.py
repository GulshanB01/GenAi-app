import streamlit as st
import os
import json
from pathlib import Path
from io import BytesIO

import easyocr
reader = easyocr.Reader(['en'], gpu=False)

import numpy as np
from rank_bm25 import BM25Okapi
from PyPDF2 import PdfReader
from PIL import Image
import cv2

import sys
RUNNING_IN_CLOUD = "streamlit" in sys.modules

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- CONFIG ----------------

DATA_DIR = Path("data")
USERS_PATH = DATA_DIR / "users.json"
FEEDBACK_PATH = DATA_DIR / "user_feedback.jsonl"
UPLOAD_DIR = DATA_DIR / "uploaded"

# Use base model for now; later you can switch to "models/finetuned_t5"
MODEL_NAME = "google/flan-t5-small"

# how much text we allow in context
MAX_CONTEXT_CHARS = 2000

# starting credits for each new user
DEFAULT_CREDITS = 10

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="AI Notes Q&A (T5)", layout="wide")


# ---------------- MODEL LOADING ----------------

@st.cache_resource
def load_t5_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    return tokenizer, model, device

tokenizer, t5_model, device = load_t5_model()


# ---------------- USER / AUTH ----------------

def load_users():
    if not USERS_PATH.exists():
        return {}
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def init_session_state():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "credits" not in st.session_state:
        st.session_state.credits = 0
    if "docs" not in st.session_state:
        st.session_state.docs = []      # list of chunks (strings)
    if "bm25" not in st.session_state:
        st.session_state.bm25 = None

init_session_state()

def login_signup_widget():
    users = load_users()
    st.sidebar.title("🔐 Login / Signup")

    mode = st.sidebar.radio("Mode", ["Login", "Signup"])

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    # Signup flow
    if mode == "Signup":
        if st.sidebar.button("Create Account"):
            if username in users:
                st.sidebar.error("Username already exists.")
            elif not username or not password:
                st.sidebar.error("Please enter username and password.")
            else:
                users[username] = {"password": password, "credits": DEFAULT_CREDITS}
                save_users(users)
                st.sidebar.success("Account created! Please login.")

    # Login flow
    else:  # Login
        if st.sidebar.button("Login"):
            if username not in users or users[username]["password"] != password:
                st.sidebar.error("Invalid username or password.")
            else:
                st.session_state.user = username
                st.session_state.credits = users[username].get("credits", DEFAULT_CREDITS)
                st.sidebar.success(f"Logged in as {username}")

    # Show user info + manual credit add + logout
    if st.session_state.user:
        st.sidebar.markdown(f"**User:** {st.session_state.user}")
        st.sidebar.markdown(f"**Credits:** {st.session_state.credits}")

        # --- Simple manual credit add (no payment gateway) ---
        with st.sidebar.expander("💰 Add credits"):
            st.write("For now, credits can be added manually (no payment gateway).")

            add_amount = st.number_input(
                "Credits to add",
                min_value=1,
                max_value=100,
                value=5,
                step=1,
                key="manual_add_credits"
            )
            if st.button("Add credits", key="btn_add_credits"):
                st.session_state.credits += int(add_amount)
                update_user_credits()
                st.success(f"Added {int(add_amount)} credits.")

        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.session_state.credits = 0
            st.session_state.docs = []
            st.session_state.bm25 = None

def update_user_credits():
    if st.session_state.user is None:
        return
    users = load_users()
    if st.session_state.user not in users:
        return
    users[st.session_state.user]["credits"] = st.session_state.credits
    save_users(users)

login_signup_widget()


# ---------------- FILE HANDLING ----------------

def extract_text_from_pdf(file: BytesIO) -> str:
    pdf_reader = PdfReader(file)
    texts = []
    for page in pdf_reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def extract_text_from_image(file: BytesIO) -> str:
    """Extract text from an image file using EasyOCR."""
    try:
        image = Image.open(file).convert("RGB")
        img_array = np.array(image)
        result = reader.readtext(img_array, detail=0)  # list of strings
        return "\n".join(result)
    except Exception as e:
        st.warning(f"OCR failed: {e}")
        return ""


def extract_text_from_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    file_bytes = uploaded_file.read()
    file_ext = uploaded_file.name.lower().split(".")[-1]

    # Optionally save original file
    with open(UPLOAD_DIR / uploaded_file.name, "wb") as f:
        f.write(file_bytes)

    file_obj = BytesIO(file_bytes)

    if file_ext == "pdf":
        return extract_text_from_pdf(file_obj)
    elif file_ext in ["txt", "md"]:
        return file_bytes.decode("utf-8", errors="ignore")
    elif file_ext in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(file_obj)
    else:
        st.warning("Unsupported file type. Use PDF/TXT/MD/PNG/JPG.")
        return ""

def simple_chunk_text(text: str, max_chars: int = 1000) -> list:
    text = text.replace("\r", " ")
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current += " " + p
        else:
            if current:
                chunks.append(current.strip())
            current = p
    if current:
        chunks.append(current.strip())
    return chunks


# ---------------- BM25 INDEX ----------------

def build_bm25_index():
    if not st.session_state.docs:
        st.session_state.bm25 = None
        return
    tokenized_docs = [doc.split() for doc in st.session_state.docs]
    st.session_state.bm25 = BM25Okapi(tokenized_docs)

def retrieve_top_chunks(query: str, top_k: int = 5):
    if st.session_state.bm25 is None or not st.session_state.docs:
        return []
    tokenized_query = query.split()
    scores = st.session_state.bm25.get_scores(tokenized_query)
    scores = np.array(scores)
    top_idxs = scores.argsort()[-top_k:][::-1]
    return [(st.session_state.docs[i], float(scores[i]), i) for i in top_idxs]


# ---------------- T5 QA ----------------

def t5_answer(question: str, context_chunks: list, max_new_tokens: int = 128) -> str:
    context = "\n\n".join(context_chunks)
    # clip context if too long
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[-MAX_CONTEXT_CHARS:]

    prompt = (
        "You are a helpful study assistant for college notes.\n"
        "Use ONLY the given context to answer the question.\n"
        "If the answer is not in the context, say: "
        "\"I couldn't find this in your notes.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def log_user_feedback(entry: dict):
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------- MAIN UI ----------------

st.title("Smart Notes AI Assistant")

if st.session_state.user is None:
    st.info("Please login or sign up from the sidebar to use the app.")
    st.stop()

st.write("Upload your PDFs / text files / images and ask questions. Each answer uses 1 credit.")

# Upload section
uploaded_files = st.file_uploader(
    "Upload notes (PDF / TXT / MD / PNG / JPG)",
    type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        text = extract_text_from_file(f)
        if not text.strip():
            continue
        chunks = simple_chunk_text(text, max_chars=1000)
        st.session_state.docs.extend(chunks)

    build_bm25_index()
    st.success(f"Indexed {len(st.session_state.docs)} chunks from uploaded files.")

st.caption(f"Indexed chunks: {len(st.session_state.docs)}")

query = st.text_input("🔎 Ask a question from your notes")
top_k = st.slider("Top-k chunks used as context", 1, 10, 5)

if st.button("Get Answer"):
    if st.session_state.credits <= 0:
        st.error("No credits left. Ask admin / dev to add more credits from sidebar.")
    elif not query.strip():
        st.error("Please enter a question.")
    elif not st.session_state.docs:
        st.error("Please upload some notes first.")
    else:
        retrieved = retrieve_top_chunks(query, top_k=top_k)
        if not retrieved:
            st.warning("No relevant chunks found. Try changing your question.")
        else:
            context_chunks = [r[0] for r in retrieved]

            with st.spinner("Thinking with T5..."):
                answer = t5_answer(query, context_chunks)

            # deduct credit
            st.session_state.credits -= 1
            update_user_credits()

            st.subheader("🧾 Answer")
            st.write(answer)

            st.subheader("📚 Sources (Top Chunks)")
            for idx, (text, score, doc_idx) in enumerate(retrieved, start=1):
                st.markdown(f"**[{idx}] Score {score:.2f}, Chunk #{doc_idx}**")
                st.write(text[:400] + ("..." if len(text) > 400 else ""))

            st.markdown("---")
            st.markdown("### 🧠 Feedback (helps the model learn later)")

            helpful = st.radio("Was this answer helpful?", ["Yes", "No", "Not sure"], index=2)
            user_correct = st.text_area("If not correct, write the correct answer (optional):")

            if st.button("Submit Feedback"):
                feedback_entry = {
                    "user": st.session_state.user,
                    "question": query,
                    "answer": answer,
                    "helpful": helpful,
                    "user_correct_answer": user_correct,
                    "context_chunks": context_chunks,
                }
                log_user_feedback(feedback_entry)
                st.success("Thanks! Your feedback has been saved.")
