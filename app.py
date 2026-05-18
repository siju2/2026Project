from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
import numpy as np
import os
import sys
import json
from datetime import datetime

# TensorFlow 임포트 예외 처리 로직 (라이브러리 부재 시 안내)
try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("❌ TensorFlow가 설치되지 않았습니다. 'sudo pip3 install tensorflow-cpu'를 실행하세요.")
    sys.exit(1)

app = Flask(__name__)
app.secret_key = 'sign_language_project_secret_9999'
socketio = SocketIO(app, cors_allowed_origins="*")

# 데이터, 모델, 통역 기록 저장 경로 설정
DATA_DIR = 'dataset'
MODEL_PATH = 'model.h5'
HISTORY_DIR = 'history'

# 폴더 자동 생성
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# 실시간 분석을 위한 전역 변수
sequence = []
actions = []
model = None

# --- [AI 자동 학습 로직] ---
def run_auto_training():
    """데이터셋을 기반으로 AI 모델을 자동으로 학습시키고 저장합니다."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        print("⚠️ 학습할 데이터가 없습니다. 먼저 수집 센터에서 데이터를 모아주세요.")
        return None

    labels_list = sorted(os.listdir(DATA_DIR))
    sequences, labels = [], []

    for idx, action in enumerate(labels_list):
        path = os.path.join(DATA_DIR, action)
        for file in os.listdir(path):
            res = np.load(os.path.join(path, file))
            sequences.append(res)
            labels.append(idx)

    if len(sequences) == 0: return None

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # 모델 설계 (두 손 기준 126개 좌표 입력)
    trained_model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)),
        LSTM(128, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(labels_list), activation='softmax')
    ])

    trained_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"🚀 AI 자동 학습 시작 (대상 단어: {labels_list})...")
    trained_model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
    trained_model.save(MODEL_PATH)
    print("✅ 학습 완료 및 model.h5 저장 성공!")
    return trained_model

# 서버 인프라 초기화 함수
def init_server():
    global model, actions
    if os.path.exists(DATA_DIR):
        actions = sorted(os.listdir(DATA_DIR))
    
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("🧠 기존 AI 모델을 성공적으로 로드했습니다.")
    else:
        model = run_auto_training()

# 초기화 런칭
init_server()

# --- [로그인 체크 세션 함수] ---
def is_logged_in():
    return 'user_id' in session

# --- [페이지 라우팅 영역] ---

@app.route('/')
def index():
    """메인 대시보드 페이지"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """로그인 페이지 처리"""
    if request.method == 'POST':
        session['user_id'] = request.form.get('email')
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    """로그아웃 세션 해제"""
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/translate')
def translate():
    """실시간 통역실 (자막, TTS 제어 기능 포함)"""
    if not is_logged_in(): 
        return redirect(url_for('login'))
    return render_template('translate_live.html')

@app.route('/collect')
def collect():
    """AI 데이터 수집 센터"""
    if not is_logged_in(): 
        return redirect(url_for('login'))
    return render_template('collect.html')

@app.route('/learn')
def learn():
    """3주차 추가: 기초 수어 학습 및 실시간 AI 채점 룸"""
    if not is_logged_in(): 
        return redirect(url_for('login'))
    return render_template('learn.html')

# --- [API 및 실시간 데이터 통신] ---

@app.route('/save-data', methods=['POST'])
def save_data():
    """데이터 수집 전용 데이터(.npy) 저장 API"""
    try:
        json_data = request.json
        action = json_data['action']
        sequence_data = json_data['data'] 
        path = os.path.join(DATA_DIR, action)
        os.makedirs(path, exist_ok=True)
        file_count = len(os.listdir(path))
        np.save(os.path.join(path, f'seq_{file_count}.npy'), np.array(sequence_data))
        return jsonify({"status": "success", "message": f"[{action}] 저장 완료!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/save-history', methods=['POST'])
def save_history():
    """3주차 추가: 클라이언트가 전송한 실시간 통역 로그를 파일로 백업합니다."""
    if not is_logged_in():
        return jsonify({"status": "error", "message": "로그인이 필요합니다."}), 401
    try:
        json_data = request.json
        history_data = json_data.get('history', [])
        user_id = session.get('user_id', 'unknown').split('@')[0]
        
        # 유니크한 타임스탬프 기반 파일명 설계
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"history_{user_id}_{timestamp}.json"
        file_path = os.path.join(HISTORY_DIR, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "user": session.get('user_id'),
                "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "logs": history_data
            }, f, ensure_ascii=False, indent=4)
            
        return jsonify({"status": "success", "message": "성공", "filename": file_name})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@socketio.on('video_frame')
def handle_video_frame(data):
    """실시간 웹캠 프레임 다중 손 좌표 분석 및 추론 엔진"""
    global sequence, model, actions
    try:
        multi_hand_landmarks = data.get('landmarks', [])
        
        # 126차원 양손 고정 데이터 패딩 전처리
        input_frame = []
        for i in range(2):
            if i < len(multi_hand_landmarks):
                res = np.array([[lm['x'], lm['y'], lm['z']] for lm in multi_hand_landmarks[i]]).flatten()
                input_frame.extend(res)
            else:
                input_frame.extend(np.zeros(21 * 3))
        
        sequence.append(input_frame)
        sequence = sequence[-30:] # 시퀀스 윈도우 크기 30 고정

        if len(sequence) == 30 and model is not None:
            # 예측 연산 실행
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            idx = np.argmax(res)
            confidence = float(res[idx])
            
            # 신뢰도가 75% 이상일 때만 확실하게 매칭값 전송
            if confidence > 0.75:
                emit('translate_result', {
                    'word': actions[idx], 
                    'confidence': confidence, 
                    'status': 'OK'
                })
    except Exception as e:
        print(f"Socket Error: {e}")

# --- [서버 가동 실행부] ---

if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Certbot SSL 보안 도메인 인증서 연동 경로
    cert_path = '/etc/letsencrypt/live/leedhproject.duckdns.org/fullchain.pem'
    key_path = '/etc/letsencrypt/live/leedhproject.duckdns.org/privkey.pem'

    if os.path.exists(cert_path) and os.path.exists(key_path):
        print("✅ HTTPS 보안 프로토콜 모드로 서버를 엽니다.")
        socketio.run(app, debug=False, host='0.0.0.0', port=8000, 
                     certfile=cert_path, keyfile=key_path)
    else:
        print("⚠️ SSL 인증서 누락으로 개발 전용 일반 HTTP 모드로 구동합니다.")
        socketio.run(app, debug=True, host='0.0.0.0', port=8000)