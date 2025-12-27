# 📚 Programming Documentation Helper

A RAG (Retrieval-Augmented Generation) system that helps you query programming documentation using Gemini AI.

## 🚀 Features

- ✅ Upload PDF documentation
- ✅ Load documentation from URLs
- ✅ Ask natural language questions
- ✅ Get code examples and explanations
- ✅ Chat interface with history
- ✅ Persistent storage

## 📋 Prerequisites

- Python 3.8 or higher
- Gemini API key (free from Google AI Studio)

## 🛠️ Installation

### 1. Clone or create the project structure:

```
programming-doc-helper/
├── app.py
├── rag_system.py
├── document_loader.py
├── requirements.txt
├── .env
└── README.md
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Get your Gemini API key:

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your key

### 4. Configure environment:

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

## 🎯 Usage

### Start the application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Add Documentation:

**Option 1: Load from URL**
1. Click the "URL" tab in the sidebar
2. Paste a documentation URL (e.g., `https://docs.python.org/3/tutorial/`)
3. Click "Load URL"

**Option 2: Upload PDF**
1. Click the "PDF" tab in the sidebar
2. Upload a PDF file
3. Click "Load PDF"

### Ask Questions:

Type your question in the chat box at the bottom, for example:
- "How do I create a list in Python?"
- "Show me an example of a decorator"
- "What's the difference between async and await?"

## 📚 Recommended Documentation Sources

### Python
- https://docs.python.org/3/tutorial/
- https://docs.python.org/3/library/

### JavaScript
- https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide

### React
- https://react.dev/learn

### Django
- https://docs.djangoproject.com/en/stable/

### FastAPI
- https://fastapi.tiangolo.com/

## 🔧 Troubleshooting

### "GEMINI_API_KEY not found"
- Make sure you created the `.env` file
- Check that your API key is correct
- Restart the application

### "Error loading URL"
- Some websites block automated access
- Try downloading as PDF and uploading instead
- Check if the URL is accessible

### "No relevant information found"
- Upload more documentation
- Try rephrasing your question
- Make sure the documentation covers that topic

## 📝 Tips for Best Results

1. **Upload comprehensive documentation** - The more context, the better answers
2. **Be specific in questions** - "How do I sort a list in Python?" vs "sorting"
3. **Ask for examples** - "Show me an example of..." usually works well
4. **Check sources** - The system shows which documents it used

## 🛡️ Limitations

- Free Gemini API has rate limits (60 requests per minute)
- Large PDFs take time to process
- Some documentation websites may block scraping

## 📦 Project Structure

- `app.py` - Streamlit UI and main application logic
- `rag_system.py` - RAG implementation with ChromaDB and Gemini
- `document_loader.py` - PDF and URL content extraction
- `data/chroma_db/` - Vector database storage (auto-created)
- `data/uploaded_pdfs/` - Uploaded PDF files (auto-created)

## 📄 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
