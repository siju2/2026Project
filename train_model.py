# train_model.py
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def run_training():
    DATA_PATH = 'dataset'
    if not os.path.exists(DATA_PATH) or len(os.listdir(DATA_PATH)) == 0:
        print("⚠️ 학습할 데이터가 없습니다. 먼저 데이터를 수집하세요.")
        return

    actions = np.array(os.listdir(DATA_PATH))
    sequences, labels = [], []

    for idx, action in enumerate(actions):
        action_path = os.path.join(DATA_PATH, action)
        files = os.listdir(action_path)
        for file in files:
            res = np.load(os.path.join(action_path, file))
            sequences.append(res)
            labels.append(idx)

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)),
        LSTM(128, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(actions.shape[0], activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("🚀 서버 시작 전 AI 모델 자동 학습 중...")
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1) # 서버 실행용이므로 epoch는 적당히 조절
    model.save('model.h5')
    print("✅ 학습 완료 및 model.h5 저장됨.")

if __name__ == "__main__":
    run_training()