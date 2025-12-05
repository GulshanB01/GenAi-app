
# Smart Notes AI Assistant

A web application that allows users to upload their notes (PDF, text files, or images) and ask questions directly from those documents. The system performs text extraction, document chunking, semantic retrieval, and answer generation using a fine-tuned T5 model.

The project is fully deployed on Streamlit Cloud and supports user authentication, credit-based usage, OCR extraction, and feedback logging.

---
# Live Demo (Hosted on Streamlit Cloud)

Access the app here:
https://smart-notes-ai.streamlit.app/

## Features

### 1. Document Upload

Supports the following formats:

* PDF
* TXT
* MD
* PNG, JPG, JPEG

### 2. OCR Support

Images with text are processed using EasyOCR for text extraction.

### 3. Chunking + Retrieval

* Extracted text is split into fixed-size chunks.
* BM25Okapi is used to retrieve the most relevant chunks for each query.

### 4. T5 Question Answering

* Uses a fine-tuned T5 model for generating answers.
* The model uses only the retrieved context from the user’s documents.

### 5. User Authentication

* Login and signup functionality.
* Each user has their own stored credits.
* Credits decrease with each answered query.

### 6. Credit System

* Credits are stored per user in `data/users.json`.
* Users can manually add credits from the UI (payment integration not included).

### 7. Feedback Logging

All user feedback is written to:

```
data/user_feedback.jsonl
```

This file can be used later to fine-tune the T5 model.

---

## Application Workflow

1. User logs in or signs up.
2. User uploads documents (PDF/text/image).
3. System extracts text and creates chunks.
4. BM25 ranks relevant chunks based on query.
5. Top-k chunks are passed to the T5 model.
6. Model generates an answer.
7. One credit is deducted per query.
8. User submits feedback (optional).

---

## Project Structure

```
├── app.py                     # Main Streamlit application
├── finetune_t5.py             # Script for T5 fine-tuning
├── requirements.txt           # Python dependencies
├── data/
│   ├── users.json             # User database (credentials + credits)
│   ├── user_feedback.jsonl    # Collected feedback for training
│   └── uploaded/              # Uploaded files by users
├── models/
│   └── finetuned_t5           # (Optional) Local fine-tuned model
```

---

## Technologies Used

### Backend / Model

* Python
* HuggingFace Transformers
* T5 (fine-tuned for question answering)
* PyTorch

### Document Processing

* PyPDF2
* EasyOCR
* PIL
* BM25 (rank_bm25 library)

### Frontend

* Streamlit
* Custom sidebar for authentication and credit management

---

## Installation (Local Setup)

### 1. Clone the repository

```
git clone https://github.com/<your-username>/<repo-name>
cd <repo-name>
```

### 2. Create a virtual environment

```
python -m venv .venv
source .venv/bin/activate         # Mac/Linux
.venv\Scripts\activate            # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. (Optional) Setup EasyOCR GPU (CPU works by default)

### 5. Run the application

```
streamlit run app.py
```

---

## Deployment

The project is deployed using Streamlit Community Cloud.

To deploy your own version:

1. Push your code to a GitHub repository.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Deploy a new app
4. Select your repository
5. Set the main file as `app.py`
6. Deploy

Streamlit Cloud automatically installs dependencies from `requirements.txt`.

---

## Fine-Tuning the T5 Model

1. Collect user feedback in `data/user_feedback.jsonl`
2. Run the fine-tuning script:

```
python finetune_t5.py
```

3. The script trains on your collected feedback.
4. Save the new model under:

```
models/finetuned_t5/
```

5. Update model path in `app.py`:

```python
MODEL_NAME = "models/finetuned_t5"
```
