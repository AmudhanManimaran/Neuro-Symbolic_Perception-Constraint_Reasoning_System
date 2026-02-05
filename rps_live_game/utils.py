# utils.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import random

# Load the trained model
def load_rps_model(model_path):
    return load_model(model_path)

# Preprocess input image for prediction
def preprocess_roi(roi):
    img = cv2.resize(roi, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Predict player's move
def predict_class(model, image, class_names):
    prediction = model.predict(image, verbose=0)
    return class_names[np.argmax(prediction)]

# Draw predictions and scores on frame
def draw_game_info(frame, user_move, comp_move, result, user_score, comp_score):
    cv2.putText(frame, f"You: {user_move}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.putText(frame, f"Bot: {comp_move}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.putText(frame, f"Result: {result}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.putText(frame, f"Score - You: {user_score}  Bot: {comp_score}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)

# Random move for bot
def get_bot_move(class_names):
    return random.choice(class_names)

# Decide winner
def decide_winner(user, bot):
    if user == bot:
        return "Draw"
    elif (user == "rock" and bot == "scissors") or \
         (user == "paper" and bot == "rock") or \
         (user == "scissors" and bot == "paper"):
        return "You Win"
    else:
        return "Bot Wins"
