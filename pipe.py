import cv2
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

# MediaPipe Hands 솔루션 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 손 인식 모델 설정 (최대 2개의 손 인식)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 켜기 (0번 카메라)
cap = cv2.VideoCapture(0)

print("MediaPipe로 손 인식을 시작합니다. 종료하려면 'ESC' 키를 누르세요.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        break

    # MediaPipe는 RGB 이미지를 사용하므로 BGR에서 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지에서 손 랜드마크 추론
    results = hands.process(image_rgb)

    # 인식된 손이 있다면 화면에 그리기
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손가락 관절 선 그리기
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            # (참고) RL 환경으로 넘길 데이터 추출 방법:
            # 관절의 x, y, z 좌표는 0.0 ~ 1.0 사이의 정규화된 값으로 나옵니다.
            # for id, lm in enumerate(hand_landmarks.landmark):
            #     print(f"관절 번호: {id}, x: {lm.x}, y: {lm.y}, z: {lm.z}")

    # 화면에 출력
    cv2.imshow('MediaPipe Hand Tracking', image)

    # ESC 키(27)를 누르면 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()