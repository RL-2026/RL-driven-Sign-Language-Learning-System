import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("🧠 분류기 학습 시작")
try:
    df = pd.read_csv('perfect_dataset.csv')
except FileNotFoundError:
    print("❌ perfect_dataset.csv 파일을 찾을 수 없습니다. 1단계를 먼저 실행하세요.")
    exit()

X = df.drop('label', axis=1)
y = df['label']

# 테스트 데이터 비율을 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# 정확도 확인
y_pred = model.predict(X_test)
print(f"🎯 학습 완료! (테스트 데이터 정확도: {accuracy_score(y_test, y_pred)*100:.2f}%)")

# 모델 저장
with open('sign_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("💾 sign_model.pkl 저장 완료!")