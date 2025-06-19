from flask import Blueprint, request, jsonify
from fastsam import FastSAM
import requests
import cv2
import numpy as np
import base64
#from ultralytics import FastSAM
import os
from uuid import uuid4
from io import BytesIO
import torch
from PIL import Image

fastsam_bp = Blueprint('fastsam', __name__)
model = FastSAM("/app/models/FastSAM-s.pt", device="cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지에서 모든 객체 자동 분할
#results = model('input.jpg', device='cuda', imgsz=640, conf=0.4)
fastsam_model = None

def initialize_fastSam():
        # print("\n모델 로딩 중...")
        # sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        # sam.to(device=device)
        # print("모델 로딩 완료")
        
    global fastsam_model
    if fastsam_model is None:
        print(f"\n사용 장치: {device}")
        if device.type == 'cuda':
            print(f"GPU 모델: {torch.cuda.get_device_name(0)}")
        
        # 1. 모델 파일 찾기
        checkpoint_path = "/app/models/FastSAM-s.pt"
        # if checkpoint_path is None:
        #     raise RuntimeError("FastSAM 모델 파일을 찾을 수 없습니다. 배포 환경에 모델 파일이 있는지 확인해주세요.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"FastSAM Checkpoint not found at {checkpoint_path}")
        print("\n모델 로딩 중...")

        # 2. 모델 파일 검증
        # is_valid, message = verify_model_file(checkpoint_path)
        # if not is_valid:
        #     raise RuntimeError(f"모델 파일 검증 실패: {message}")
        
        # 3. 모델 로딩
        try:            
            print(f"\nFastSAM 모델 로딩 중... ({checkpoint_path})")
            fastsam_model = FastSAM(checkpoint_path, device=device)
            #fastsam_model = YOLO(checkpoint_path)
            
            # 모델이 FastSAM인지 확인
            model_info = str(fastsam_model.model)
            if 'FastSAM' not in model_info and 'SAM' not in model_info:
                print("Warning: 로딩된 모델이 FastSAM이 아닐 수 있습니다.")
            
            print("FastSAM 모델 로딩 완료")
            
        except Exception as e:
            print(f"[ERROR] FastSAM 모델 로딩 실패: {e}")
            raise RuntimeError(f"FastSAM 모델 초기화 실패: {e}")

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

# def run_fastsam_inference(image):
#     """FastSAM 추론 실행"""
#     try:
#         # FastSAM 추론 실행
#         results = fastsam_model(
#             image,
#             device=device,
#             retina_masks=True,
#             imgsz=1024,  # TODO : 해상도 512도 테스트 필요(1024 실패 시)
#             conf=0.4,
#             iou=0.9
#         )
        
#         # 결과에서 마스크 추출
#         if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
#             masks = results[0].masks.data.cpu().numpy()
#             return masks
#         else:
#             print("FastSAM에서 마스크를 생성하지 못했습니다.")
#             return None
            
#     except Exception as e:
#         print(f"[ERROR] FastSAM 추론 실패: {e}")
#         return None
def run_fastsam_inference(image: np.ndarray):
    # FastSAM은 PIL Image도, numpy array 받을 수 있음
    pil_img = Image.fromarray(image)
    results = fastsam_model(pil_img,
                            device="cuda",
                            retina_masks=True,
                            imgsz=1024,  # TODO : 해상도 512도 테스트 필요(1024 실패 시)
                            conf=0.4,
                            iou=0.9)
    if results and hasattr(results[0], "masks") and results[0].masks is not None:
        return results[0].masks.data.cpu().numpy()
    return []

def extract_objects_from_fastsam_masks(image, masks, margin=10):
    """FastSAM 마스크에서 객체 추출"""
    objects = []
    
    if masks is None or len(masks) == 0:
        return objects
    
    for i, mask in enumerate(masks):
        try:
            # 마스크를 uint8로 변환
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            # 마스크가 비어있으면 건너뛰기
            if np.sum(binary_mask) < 100:  # 최소 픽셀 수 확인
                continue
            
            # RGBA 이미지 생성
            image_rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            image_rgba[:, :, :3] = image
            image_rgba[:, :, 3] = binary_mask * 255
            
            # PIL 이미지로 변환
            image_pil = Image.fromarray(image_rgba)
            
            # 바운딩 박스 계산
            mask_pil = Image.fromarray(binary_mask * 255)
            bbox = mask_pil.getbbox()
            if not bbox:
                continue
            
            # 마진을 포함한 크롭 박스 계산
            crop_box = (
                max(0, bbox[0] - margin),
                max(0, bbox[1] - margin),
                min(image_pil.width, bbox[2] + margin),
                min(image_pil.height, bbox[3] + margin),
            )
            
            # 이미지 크롭
            cropped_image = image_pil.crop(crop_box)
            
            # Base64 인코딩
            buffer = BytesIO()
            cropped_image.save(buffer, format="PNG")
            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            objects.append({
                "uuid": str(uuid4()),
                "filename": str(uuid4()),
                "base64Image": base64_str
            })
            
        except Exception as e:
            print(f"[ERROR] 객체 {i} 추출 실패: {e}")
            continue
    
    return objects

def generate_uuid():
    return str(uuid4())

def encode_image_to_base64(image):
    image_pil = Image.fromarray(image)
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@fastsam_bp.route('/fastsam', methods=['POST'])
def handle_fastsam():
    try:
        initialize_fastSam()
        
        # 1. 이미지 URL 수신
        data = request.get_json()
        if 'resultImage' not in data:
            return jsonify({'error': 'resultImage parameter is required'}), 400
            
        #image_url = data['resultImage']
        
        # 2. 이미지 다운로드 및 전처리
        #original_image = download_image(image_url)
        original_image = download_image(data['resultImage'])
        resized_image = process_image(original_image)
        
        masks = run_fastsam_inference(resized_image)

        # mask_generator = run_fastsam_inference(
        #     model=FastSAM,
        #     points_per_side=4,
        #     pred_iou_thresh=0.95,
        #     stability_score_thresh=0.95,
        #     crop_n_layers=0,
        #     min_mask_region_area=512
        # )
        #masks = mask_generator.generate(resized_image)

        
        # 결과 시각화
        #visualize_sam_result(resized_image, masks)
        
        # 4. 객체 추출 및 Base64 인코딩
        cutout_objects = extract_objects_from_fastsam_masks(resized_image, masks)
        
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
