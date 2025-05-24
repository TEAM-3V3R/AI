from flask import Blueprint, request, jsonify
import requests
import cv2
import numpy as np
import base64
#import time
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import os
from uuid import uuid4

sam_bp = Blueprint('sam', __name__)

checkpoint_path = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = None

def initialize_sam():
    global sam
    if sam is None:
        print("\n사용 장치:", device)
        if device.type == 'cuda':
            print(f"GPU 모델: {torch.cuda.get_device_name(0)}")
        
        print("\n모델 로딩 중...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        print("모델 로딩 완료")

initialize_sam()

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    image = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def process_image(image):
    height, width = image.shape[:2]
    target_size = 1000
    if height > width:
        scale = target_size / height
    else:
        scale = target_size / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def extract_objects(image, masks):
    objects = []
    for i, mask in enumerate(masks):
        binary_mask = mask['segmentation'].astype(np.uint8)
        y_indices, x_indices = np.where(binary_mask > 0)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        y_max = min(image.shape[0], y_max + padding)
        
        cut_out = image[y_min:y_max, x_min:x_max]
        _, buffer = cv2.imencode('.png', cv2.cvtColor(cut_out, cv2.COLOR_RGB2BGR))
        base64_str = base64.b64encode(buffer).decode('utf-8')

        objects.append({
            "uuid": str(uuid4()),
            "base64Image": base64_str
        })
    
    return objects

def generate_uuid():
    return str(uuid4())

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

@sam_bp.route('/sam', methods=['POST'])
def handle_sam():
    try:
        # start_time = time.time()
        
        # 1. 이미지 URL 수신
        data = request.get_json()
        if 'resultImage' not in data:
            return jsonify({'error': 'resultImage parameter is required'}), 400
            
        image_url = data['resultImage']
        
        # 2. 이미지 다운로드 및 전처리
        original_image = download_image(image_url)
        resized_image = process_image(original_image)
        
        # 3. SAM 세그멘테이션 실행
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(resized_image)
        
        # 4. 객체 추출 및 Base64 인코딩
        cutout_objects = extract_objects(resized_image, masks)
        
        original_uuid = str(uuid4())
        original_base64 = encode_image_to_base64(resized_image)

        # 5. 응답 생성
        # processing_time = time.time() - start_time
        result = [{
            "uuid": original_uuid,
            "base64Image": original_base64
        }] + cutout_objects
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
