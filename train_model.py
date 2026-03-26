import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("🧠 [2단계] 26개 알파벳 분류기 학습 시작...")
df = pd.read_csv('perfect_dataset.csv')
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('sign_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f"✅ 학습 완료! (알파벳 개수: {len(model.classes_)})")