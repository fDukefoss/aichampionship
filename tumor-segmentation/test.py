from utils import decode_request, encode_request, plot_prediction, dice_score
import requests
import cv2
import numpy as np
import base64
import os

image_path = "data/patients/imgs/patient_001.png"
mask_path = "data/patients/labels/segmentation_001.png"

img = cv2.imread(image_path)
gt_mask = cv2.imread(mask_path)

# Encode image
encoded = encode_request(img)
payload = {"img": encoded}

# Predict
res = requests.post("http://localhost:9051/predict", json=payload).json()
pred_encoded = res['img']

# Decode prediction
decoded = base64.b64decode(pred_encoded)
nparr = np.frombuffer(decoded, np.uint8)
predicted_mask = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Plot
plot_prediction(img, gt_mask, predicted_mask)
