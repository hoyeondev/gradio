import torch # 딥러닝 모델을 만들고 실행하는 PyTorch 라이브러리
import numpy as np # 수치 연산을 위한 NumPy 라이브러리
from PIL import Image # 이미지 처리를 위한 Pillow 라이브러리
import gradio as gr # 모델을 웹 앱 형태로 쉽게 공유하는 Gradio 라이브러리
from pytorch_grad_cam import GradCAM # Grad-CAM을 계산하는 라이브러리
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget # 분류 모델의 타겟을 설정하는 함수
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image # CAM 시각화 및 이미지 전처리 함수
from efficientnet_pytorch import EfficientNet # EfficientNet 모델 구조를 불러오는 라이브러리

# CPU 또는 GPU 장비 설정 (GPU가 있다면 'cuda', 없으면 'cpu' 사용)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 예측 클래스 이름 정의 (정상, 폐렴)
CLASS_NAMES = ["정상", "폐렴"]

# 모델 생성 및 가중치 불러오기
# EfficientNet-b3 모델 구조를 불러오고 클래스 수를 2개로 설정
model = EfficientNet.from_name("efficientnet-b3", num_classes=2)
# 모델을 설정한 장비(CPU/GPU)로 이동
model = model.to(DEVICE)

# model_state_dict.pth 파일에서 미리 학습된 모델 가중치 불러오기
# map_location=DEVICE를 통해 가중치를 현재 장비에 맞게 매핑
ckpt = torch.load("model_state_dict.pth", map_location=DEVICE)

# 불러온 가중치를 모델에 적용
model.load_state_dict(ckpt)
# 모델을 평가(evaluation) 모드로 설정 (드롭아웃, 배치 정규화 등 비활성화)
model.eval()

# 모델 로딩 완료 메시지 출력
print("✅ Model loaded!")

# Grad-CAM을 적용할 마지막 레이어 설정
# EfficientNet의 마지막 블록을 타겟 레이어로 지정
target_layer = model._blocks[-1]

# GradCAM 객체 생성. 모델과 타겟 레이어를 지정
gradcam = GradCAM(model=model, target_layers=[target_layer])

# ====== 예측을 위한 클래스 정의 ======
class EfficientNetPredictor:
    # 클래스 초기화. 필요한 객체들을 인자로 받음
    def __init__(self, model, gradcam, device, class_names):
        self.model = model
        self.gradcam = gradcam
        self.device = device
        self.class_names = class_names

    # 이미지 예측 및 Grad-CAM 시각화 함수
    def predict_and_cam(self, pil_img):
        # 이미지를 NumPy 배열로 변환하고 224x224로 크기 조정
        # Grad-CAM 라이브러리의 전처리 함수를 사용하여 PyTorch 텐서로 변환
        img_tensor = preprocess_image(np.array(pil_img.resize((224,224))), 
                                      mean=[0.485, 0.456, 0.406], # 이미지 정규화를 위한 평균값
                                      std=[0.229, 0.224, 0.225]) # 이미지 정규화를 위한 표준편차
        # 텐서를 설정한 장비로 이동
        img_tensor = img_tensor.to(self.device)

        # 모델 예측 (기울기 계산 비활성화)
        with torch.no_grad():
            # 이미지 텐서를 모델에 입력하여 출력값(outputs) 계산
            outputs = self.model(img_tensor)
            # 출력값을 확률값으로 변환
            probs = torch.softmax(outputs, dim=1)
            # 가장 높은 확률을 가진 클래스의 인덱스 추출
            pred_class = torch.argmax(probs, dim=1).item()
        
        # Grad-CAM 생성
        # 예측된 클래스를 Grad-CAM의 타겟으로 설정
        targets = [ClassifierOutputTarget(pred_class)]
        # GradCAM 객체를 사용하여 CAM 이미지(히트맵) 생성
        grayscale_cam = self.gradcam(input_tensor=img_tensor, targets=targets)
        # 원본 이미지 위에 CAM 히트맵을 덧입혀서 시각화
        cam_image = show_cam_on_image(np.array(pil_img.resize((224,224)))/255.0, 
                                      grayscale_cam[0], use_rgb=True)
        # 예측 확률과 CAM 이미지 반환
        return probs.cpu().numpy()[0], cam_image

# Predictor 객체 생성
predictor = EfficientNetPredictor(model, gradcam, DEVICE, CLASS_NAMES)

# ====== Gradio 인터페이스를 위한 함수 정의 ======
def gradio_fn(img):
    # Gradio로부터 받은 NumPy 이미지를 PIL 이미지로 변환하고 RGB로 포맷 변경
    pil_img = Image.fromarray(img).convert("RGB")
    # predictor 클래스의 함수를 호출하여 예측 및 CAM 이미지 생성
    probs, cam_image = predictor.predict_and_cam(pil_img)

    # 예측 결과를 '클래스 이름: 확률' 형태의 딕셔너리로 변환하여 반환
    output_dict = {predictor.class_names[i]: float(probs[i]) for i in range(len(probs))}

    return output_dict, cam_image

# ====== Gradio 인터페이스 생성 ======
# gr.Interface 객체를 생성하여 웹 앱의 구조 정의
demo = gr.Interface(
    fn=gradio_fn, # 웹 앱이 호출할 함수
    inputs=gr.Image(type="numpy", label="Upload Chest X-ray"), # 입력 컴포넌트: NumPy 타입의 이미지 업로드 위젯
    outputs=[gr.Label(label="Prediction"), gr.Image(label="Grad-CAM")], # 출력 컴포넌트: 예측 결과 레이블과 Grad-CAM 이미지
    title="폐렴일까 정상일까?" # 웹 앱의 제목
)

# 웹 앱 실행
demo.launch(share=True)
