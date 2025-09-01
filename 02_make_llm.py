import gradio as gr
import requests
import random

# 랜덤 질문 리스트
questions = [
    "What did you eat for breakfast today?",
    "If you could travel anywhere, where would you go?",
    "Tell me about a time you felt proud of yourself.",
    "What do you usually do on weekends?",
    "Which do you prefer: books or movies? Why?"
]

# Ollama API 호출 함수
def ask_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/v1/chat/completions",
        json={
            "model": "llama3:8b",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# 대화 상태 저장
history = []

def study_mate(user_input):
    global history
    
    if not history:  # 첫 시작 → 랜덤 질문
        question = random.choice(questions)
        history.append(("🤖 Bot", question))
        return "\n".join([f"{speaker}: {msg}" for speaker, msg in history])
    
    # 사용자의 답변 기록
    history.append(("🧑 You", user_input))
    
    # Ollama에게 문법 교정 & 피드백 요청
    feedback_prompt = f"Correct this English sentence and give feedback:\n\n'{user_input}'"
    feedback = ask_ollama(feedback_prompt)
    
    # 다음 질문
    next_question = random.choice(questions)
    
    history.append(("🤖 Bot", feedback))
    history.append(("🤖 Bot", next_question))
    
    return "\n".join([f"{speaker}: {msg}" for speaker, msg in history])

with gr.Blocks() as demo:
    gr.Markdown("## 🗣️ 영어 회화 스터디 메이트 (Ollama LLM)")
    chatbot = gr.Textbox(label="Chat Log", lines=15)
    user_input = gr.Textbox(label="Your Answer")
    submit = gr.Button("Send")
    
    submit.click(study_mate, inputs=user_input, outputs=chatbot)

demo.launch()
