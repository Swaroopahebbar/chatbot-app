import streamlit as st
import transformers
import torch
import pdfplumber

# Load the transformer model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Extract text from PDF
def extract_pdf_text(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Only add if there's text on the page
                text += page_text + "\n"
    return text

# Function for chatbot to respond based on PDF text
def chatbot_response(user_message: str, pdf_text: str) -> str:
    if "pdf" in user_message.lower() or "document" in user_message.lower():
        bot_input = f"User: {user_message}\nDocument: {pdf_text[:3000]}"  # Limiting text for performance
        outputs = pipeline(bot_input, max_new_tokens=512)
        return outputs[0]["generated_text"].strip()
    else:
        return "I can only answer questions based on the provided document."

# Streamlit UI for Chatbot
def run_chatbot():
    st.title("PDF Chatbot")
    uploaded_pdf = st.file_uploader("Upload a PDF Document", type=["pdf"])
    
    if uploaded_pdf:
        pdf_text = extract_pdf_text(uploaded_pdf)
        st.write("PDF loaded successfully! You can now ask questions related to the document.")
        
        user_message = st.text_input("Ask something:")
        
        if user_message:
            bot_response = chatbot_response(user_message, pdf_text)
            st.write(f"Bot: {bot_response}")
    
if __name__ == "__main__":
    run_chatbot()

