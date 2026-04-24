import gradio as gr
from embedding_vis import generate_embedding_viz

from ai_waiter_chatbot import (
    MENU_FILE,
    TinyRetriever,
    answer_with_rag,
    answer_without_rag,
    load_menu_items,
)

items = load_menu_items(MENU_FILE)
retriever = TinyRetriever(items)

def ask_waiter(user_query: str) -> tuple[str, str, str]:
    query = (user_query or "").strip()
    if not query:
        return "Please enter a menu question.", "Please enter a menu question.", ""

    # 1. Get standard responses
    no_rag = answer_without_rag(query)
    with_rag = answer_with_rag(query, retriever)
    
    # 2. Get retrieved documents for the visualization
    retrieved_docs = retriever.get_relevant_documents(query) 
    
    # 3. Generate the HTML visualization
    html_file_path = generate_embedding_viz(query, retrieved_docs)
    
    with open(html_file_path, 'r', encoding='utf-8') as f:
        viz_html = f.read()

    return no_rag, with_rag, viz_html

demo = gr.Interface(
    fn=ask_waiter,
    inputs=gr.Textbox(
        lines=2,
        label="Ask the AI Waiter",
        placeholder="Example: What pasta dishes do you have under $26?",
    ),
    outputs=[
        gr.Textbox(label="Response 1: Without RAG"),
        gr.Textbox(label="Response 2: With RAG"),
        gr.HTML(label="Knowledge Retrieval Graph (Interactive)")
    ],
    title="Cheesecake Factory AI Waiter (RAG vs No RAG)",
    description=(
        "Ask a question about the menu and compare responses. "
        "The first response is generic (no retrieval). "
        "The second response is retrieval-grounded. "
        "The graph shows which menu nodes were 'pulled' to answer your question."
    ),
)

if __name__ == "__main__":
    demo.launch()
