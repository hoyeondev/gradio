import gradio as gr
import requests

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

# 랜덤 영어 질문 생성
def get_random_question():
    prompt = "Generate one random English conversation question for practice. Keep it short and simple."
    return ask_ollama(prompt)

# 대화 저장
history = []

def study_mate(user_input):
    global history
    
    if not history:  # 첫 시작 → 질문 생성
        question = get_random_question()
        history.append(("🤖 Bot", question))
        chat_text = "\n".join([f"{speaker}: {msg}" for speaker, msg in history])
        return chat_text, gr.update(value="Send")   # ✅ 수정된 부분
    
    # 사용자의 답변 기록
    history.append(("🧑 You", user_input))
    
    # Ollama에게 문법 교정 & 피드백 요청
    feedback_prompt = f"Correct this English sentence and give feedback:\n\n'{user_input}'"
    feedback = ask_ollama(feedback_prompt)
    
    # 다음 질문 생성
    next_question = get_random_question()
    
    history.append(("🤖 Bot", feedback))
    history.append(("🤖 Bot", next_question))
    
    chat_text = "\n".join([f"{speaker}: {msg}" for speaker, msg in history])
    return chat_text, gr.update(value="Send")   # ✅ 수정된 부분

with gr.Blocks() as demo:
    gr.Markdown("## 🗣️ 영어 회화 스터디 메이트 (Ollama LLM)")
    chatbot = gr.Textbox(label="Chat Log", lines=15)
    user_input = gr.Textbox(label="Your Answer")
    submit = gr.Button("Start")  # 초기에는 Start
    
    submit.click(
        study_mate, 
        inputs=user_input, 
        outputs=[chatbot, submit]  # 버튼 라벨 업데이트 가능
    )

demo.launch(share=True)
