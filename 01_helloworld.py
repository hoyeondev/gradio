import gradio as gr
import os

# API 없이 사용하도록 하기
os.environ["OPENAI_API_KEY"] = "not-needed"


demo = gr.load_chat(
    "http://localhost:11434/v1/",
    model="llama3:8b"
)

demo.launch()