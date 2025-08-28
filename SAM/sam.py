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

from PIL import Image
from io import BytesIO
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches

sam_bp = Blueprint('sam', __name__)

#checkpoint_path = "sam_vit_h_4b8939.pth"
#model_type = "vit_h"
model_type = "vit_b"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = None

def download_sam_checkpoint():
    ckpt_path = "sam_vit_b_01ec64.pth"
    if not os.path.exists(ckpt_path):
        try:
            print("SAM checkpoint 다운로드 중")
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            r = requests.get(url, stream=True)
            with open(ckpt_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("SAM checkpoin 다운로드 완료")
        except Exception as e:
            print(f"[ERROR] Checkpoint download failed: {e}")
            raise RuntimeError("SAM 모델 체크포인트 다운로드 실패")
    return ckpt_path


def initialize_sam():
    global sam
    device = torch.device("cuda:0")
    print(f"GPU 모델: {torch.cuda.get_device_name(0)}")

    if sam is None:    
        #checkpoint_path = download_sam_checkpoint()
        checkpoint_path = "/app/models/sam_vit_b_01ec64.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM Checkpoint not found at {checkpoint_path}")
        print("\n모델 로딩 중...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        print("모델 로딩 완료")

# initialize_sam()

def download_image(url):
    response = requests.get(url)
    response.raise_for_status()
    image = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def process_image(image):
    height, width = image.shape[:2]
    target_size = 512
    if height > width:
        scale = target_size / height
    else:
        scale = target_size / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


#def visualize_sam_result(image, masks):
#    plt.figure(figsize=(10, 10))
#    plt.imshow(image)
#    ax = plt.gca()
#
#    for i, mask in enumerate(masks):
#        m = mask['segmentation']
#        color = np.random.rand(3,)
#        ax.imshow(np.dstack((m * color[0], m * color[1], m * color[2], m * 0.35)))  # 반투명 색상
#
#        # 경계 박스 그리기
#        y, x = np.where(m)
#        if len(x) > 0 and len(y) > 0:
#            xmin, xmax = np.min(x), np.max(x)
#            ymin, ymax = np.min(y), np.max(y)
#            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                     linewidth=1.5, edgecolor=color, facecolor='none')
#            ax.add_patch(rect)
#
#    plt.axis('off')
#    plt.tight_layout()
#    plt.show()


def extract_objects(image, masks, margin=10):
    objects = []
        
    for mask in masks:
        binary_mask = mask['segmentation'].astype(np.uint8)

        image_rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        image_rgba[:, :, :3] = image
        image_rgba[:, :, 3] = binary_mask * 255

        image_pil = Image.fromarray(image_rgba)

        bbox = Image.fromarray(binary_mask * 255).getbbox()
        if not bbox:
            continue

        crop_box = (
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(image_pil.width, bbox[2] + margin),
            min(image_pil.height, bbox[3] + margin),
        )

        cropped_image = image_pil.crop(crop_box)

        buffer = BytesIO()
        cropped_image.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        objects.append({
            "uuid": str(uuid4()),
            "filename": str(uuid4()),
            "base64Image": base64_str
        })
    
    return objects

def generate_uuid():
    return str(uuid4())

def encode_image_to_base64(image):
    image_pil = Image.fromarray(image)
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@sam_bp.route('/sam', methods=['POST'])
def handle_sam():
    try:
        initialize_sam()
        
        # 1. 이미지 URL 수신
        data = request.get_json()
        if 'resultImage' not in data:
            return jsonify({'error': 'resultImage parameter is required'}), 400
            
        image_url = data['resultImage']
        
        # 2. 이미지 다운로드 및 전처리
        original_image = download_image(image_url)
        resized_image = process_image(original_image)
        
        # 3. SAM 세그멘테이션 실행
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=4,
            pred_iou_thresh=0.95,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=512
        )
        masks = mask_generator.generate(resized_image)

        
        # 🖼️ 결과 시각화
        #visualize_sam_result(resized_image, masks)
        
        # 4. 객체 추출 및 Base64 인코딩
        cutout_objects = extract_objects(resized_image, masks)
        
        original_uuid = str(uuid4())
        original_base64 = encode_image_to_base64(resized_image)

        # 5. 응답 생성
        result = [{
            "uuid": original_uuid,
            "base64Image": original_base64
        }] + cutout_objects
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
if __name__ == "__main__":
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(sam_bp)

    app.run(host="0.0.0.0", port=5001)
