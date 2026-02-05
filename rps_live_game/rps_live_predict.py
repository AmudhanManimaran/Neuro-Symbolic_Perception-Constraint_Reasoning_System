# rps_live_predict.py
import cv2
import time
from utils import (
    load_rps_model,
    preprocess_roi,
    predict_class,
    draw_game_info,
    get_bot_move,
    decide_winner
)

# Load model
model_path = "rps_mobilenetv2_final.keras"
model = load_rps_model(model_path)
print("[INFO] Model loaded!")

# Labels used in training
class_names = ['paper', 'rock', 'scissors']

# Webcam init
print("[INFO] Starting webcam. Press 'q' to quit.")
cap = cv2.VideoCapture(0)

user_score = 0
bot_score = 0
last_result = "Waiting..."
user_move = "None"
bot_move = "None"
next_play_time = time.time() + 5  # Start timer

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame not captured.")
        break

    # Define ROI
    x1, y1, x2, y2 = 50, 50, 350, 350
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    current_time = time.time()

    if current_time >= next_play_time:
        processed_img = preprocess_roi(roi)
        user_move = predict_class(model, processed_img, class_names)
        bot_move = get_bot_move(class_names)
        last_result = decide_winner(user_move, bot_move)

        if last_result == "You Win":
            user_score += 1
        elif last_result == "Bot Wins":
            bot_score += 1

        next_play_time = current_time + 5  # Wait 5 seconds before next round

    else:
        # Countdown display
        remaining = int(next_play_time - current_time)
        last_result = f"Next round in {remaining}s"

    draw_game_info(frame, user_move, bot_move, last_result, user_score, bot_score)
    cv2.imshow("Rock Paper Scissors - AI Bot", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Game exited.")
