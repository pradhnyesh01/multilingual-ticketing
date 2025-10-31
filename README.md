<img width="1470" height="835" alt="image" src="https://github.com/user-attachments/assets/e76b426c-3fc8-4ee1-b1a4-dc5317743bf0" />

# IT Support RAG Chatbot-> Vega 🤖

This project is a RAG (Retrieval-Augmented Generation) chatbot that answers IT support questions based on a knowledge base of past tickets.

## Features
-   **Multilingual:** Supports both English and German.
-   **Metadata Filtering:** Uses `language` and `type` to find relevant answers.
-   **Powered By:** LlamaIndex, Streamlit, and ChromaDB.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-rag-project.git](https://github.com/your-username/your-rag-project.git)
    cd your-rag-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your API Key:**
    -   Create a file named `.env`
    -   Add your key: `OPENAI_API_KEY=sk-xxxxxxxx`

4.  **Build the Knowledge Base (Run once):**
    This will load the CSV from the `data/` folder and build the local `chroma_db/` vector store.
    ```bash
    python src/ingest.py
    ```

5.  **Run the Chatbot App:**
    ```bash
    streamlit run src/app.py
    ```
    Now open your browser to the URL provided (e.g., `http://localhost:8501`).
