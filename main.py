import os
import csv
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
MODEL_PATH = "model"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 수어 라벨 목록
LABELS = ["안녕", "감사합니다", "사랑해", "도움", "예", "아니오", "화장실", "행복", "슬퍼", "반가워"]

# 모델 전역 변수
model = None
label_encoder = None

# ───────────────────────────────────────────
# 데이터 구조
# ───────────────────────────────────────────
class Landmark(BaseModel):
    x: float
    y: float
    z: float

class CollectData(BaseModel):
    label: str
    landmarks: List[Landmark]

class PredictData(BaseModel):
    landmarks: List[Landmark]

# ───────────────────────────────────────────
# 화면 보여주기
# ───────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_interface():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# ───────────────────────────────────────────
# 라벨 목록 반환
# ───────────────────────────────────────────
@app.get("/labels")
async def get_labels():
    counts = {}
    for label in LABELS:
        label_dir = os.path.join(DATA_DIR, label)
        file_path = os.path.join(label_dir, f"{label}.csv")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                counts[label] = sum(1 for _ in f)
        else:
            counts[label] = 0
    return {"labels": LABELS, "counts": counts}

# ───────────────────────────────────────────
# 데이터 수집
# ───────────────────────────────────────────
@app.post("/collect")
async def collect_landmarks(data: CollectData):
    if data.label not in LABELS:
        raise HTTPException(status_code=400, detail="Invalid label")

    label_dir = os.path.join(DATA_DIR, data.label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    file_path = os.path.join(label_dir, f"{data.label}.csv")
    row = []
    for lm in data.landmarks:
        row.extend([lm.x, lm.y, lm.z])

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    with open(file_path, "r") as f:
        count = sum(1 for _ in f)

    return {"status": "saved", "count": count}

# ───────────────────────────────────────────
# 모델 학습
# ───────────────────────────────────────────
@app.post("/train")
async def train_model():
    global model, label_encoder

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        import pickle
    except ImportError:
        return {"status": "error", "message": "sklearn 설치 필요: pip install scikit-learn"}

    X, y = [], []

    for label in LABELS:
        file_path = os.path.join(DATA_DIR, label, f"{label}.csv")
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 63:  # 21개 랜드마크 x 3(x,y,z)
                    X.append([float(v) for v in row])
                    y.append(label)

    if len(X) < 10:
        return {"status": "error", "message": "데이터가 부족합니다. 각 수어당 최소 30개 이상 수집하세요."}

    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    with open(os.path.join(MODEL_PATH, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_PATH, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    return {
        "status": "success",
        "accuracy": round(accuracy * 100, 2),
        "total_samples": len(X),
        "message": f"학습 완료! 정확도: {round(accuracy * 100, 2)}%"
    }

# ───────────────────────────────────────────
# 모델 로드
# ───────────────────────────────────────────
def load_model():
    global model, label_encoder
    try:
        import pickle
        model_file = os.path.join(MODEL_PATH, "model.pkl")
        encoder_file = os.path.join(MODEL_PATH, "label_encoder.pkl")
        if os.path.exists(model_file) and os.path.exists(encoder_file):
            with open(model_file, "rb") as f:
                model = pickle.load(f)
            with open(encoder_file, "rb") as f:
                label_encoder = pickle.load(f)
            print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 실패: {e}")

# ───────────────────────────────────────────
# 실시간 예측
# ───────────────────────────────────────────
@app.post("/predict")
async def predict(data: PredictData):
    if model is None or label_encoder is None:
        load_model()

    if model is None:
        return {"label": None, "confidence": 0, "message": "모델이 없습니다. 먼저 학습하세요."}

    if len(data.landmarks) != 21:
        return {"label": None, "confidence": 0, "message": "손 랜드마크 21개 필요"}

    row = []
    for lm in data.landmarks:
        row.extend([lm.x, lm.y, lm.z])

    X = np.array([row])
    proba = model.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    confidence = proba[pred_idx]

    if confidence < 0.6:
        return {"label": "인식 중...", "confidence": float(confidence), "message": "확신도 낮음"}

    label = label_encoder.inverse_transform([pred_idx])[0]
    return {
        "label": label,
        "confidence": float(confidence),
        "message": "success"
    }

# ───────────────────────────────────────────
# 모델 상태 확인
# ───────────────────────────────────────────
@app.get("/status")
async def status():
    model_exists = os.path.exists(os.path.join(MODEL_PATH, "model.pkl"))
    return {
        "model_loaded": model is not None,
        "model_exists": model_exists
    }

# ───────────────────────────────────────────
# 서버 실행
# ───────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)