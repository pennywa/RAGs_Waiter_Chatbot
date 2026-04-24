import os
from pyvis.network import Network
import tempfile

def generate_embedding_viz(query, retrieved_docs):
    """
    Generates an interactive HTML network graph showing the relationship 
    between the user query and the retrieved menu items.
    """
    # Init network
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    
    # 1. Add the Center Node (The User's Question)
    net.add_node(0, label=f"Query: {query[:30]}...", title=query, color="#FF4B4B", size=30)
    
    # 2. Add Document Nodes (The Menu Chunks)
    for i, doc in enumerate(retrieved_docs, start=1):
        # Extract a snippet for the label
        content = doc.page_content
        label = content[:40].replace("\n", " ") + "..."
        
        # Add the node
        net.add_node(i, label=label, title=content, color="#1f77b4", size=20)
        
        # 3. Add Edges (Connecting Query to Results)
        # We assume if it was retrieved, it is related.
        net.add_edge(0, i, value=1, title="Similarity Match")

    # Physics makes it bouncy and interactive
    net.toggle_physics(True)
    
    # Save to a temporary file so Gradio/Hugging Face can read it
    output_path = "network.html"
    net.save_graph(output_path)
    
    return output_path
