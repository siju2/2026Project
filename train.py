import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. 데이터 로드 및 전처리
DATA_PATH = 'dataset' # 데이터가 저장된 폴더
actions = os.listdir(DATA_PATH) # dataset 폴더 안의 폴더명들을 단어 리스트로 가져옴
actions.sort()

data_list = []
label_list = []

print(f"학습을 시작할 단어 목록: {actions}")

for idx, action in enumerate(actions):
    action_path = os.path.join(DATA_PATH, action)
    files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    
    for file in files:
        res = np.load(os.path.join(action_path, file))
        # res의 모양이 (30, 63)인지 확인 (30프레임, 21개 관절*3좌표)
        if res.shape == (30, 63):
            data_list.append(res)
            label_list.append(idx)

X = np.array(data_list)
y = to_categorical(label_list).astype(int)

# 데이터를 학습용(80%)과 테스트용(20%)으로 나눔
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"총 학습 데이터 수: {len(X)}")

# 2. AI 모델 설계 (LSTM: 시간 흐름을 읽는 신경망)
model = Sequential([
    # 첫 번째 LSTM 층: 30프레임의 흐름을 읽음
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)),
    Dropout(0.2), # 과적합 방지 (너무 데이터에만 집착하지 않게 함)
    
    # 두 번째 LSTM 층: 더 복잡한 패턴 분석
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.2),
    
    # 일반 신경망 층
    Dense(64, activation='relu'),
    
    # 출력층: 단어 개수만큼 확률로 표시 (Softmax)
    Dense(len(actions), activation='softmax')
])

# 3. 학습 설정
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 모델 학습 시작
print("학습을 시작합니다... 잠시만 기다려주세요.")
model.fit(
    x_train, y_train, 
    epochs=100,             # 100번 반복 학습
    batch_size=16,          # 한 번에 16개씩 묶어서 공부
    validation_data=(x_test, y_test)
)

# 5. 모델 저장 (매우 중요!)
model.save('sign_model.h5')
print("\n🎉 학습 완료! 'sign_model.h5' 파일이 생성되었습니다.")