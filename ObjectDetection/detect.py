import argparse
import os

import cv2
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights, ssd300_vgg16, SSD300_VGG16_Weights, retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from PIL import Image

def detect(model, source, preprocess, class_names, threshold, device, output):
    model.eval()
    with torch.no_grad():
        if source != '0':
            img = cv2.imread(source)
            if img is None:
                print(f"画像を読み込めませんでした: {source}")
                return

            converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(converted_img)
            img_tensor = preprocess(pil_img).unsqueeze(0)
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                outputs = model(img_tensor)[0]
            
            for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
                if score >= threshold:
                    x1, y1, x2, y2 = box.int().tolist()
                    class_name = class_names[label]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{class_name}: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            result_path = os.path.join(output, 'result.jpg')

            cv2.imwrite(result_path, img)
            print(f"結果を保存しました: {result_path}")

        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("カメラが開けませんでした")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("フレームの取得に失敗しました")
                    break

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                img_tensor = preprocess(pil_img).unsqueeze(0)
                img_tensor = img_tensor.to(device)

                with torch.no_grad():
                    outputs = model(img_tensor)[0]

                for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
                    if score >= threshold:
                        
                        x1, y1, x2, y2 = box.int().tolist()
                        class_name = class_names[label]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow('Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection with Pretrained Models")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on (cpu or cuda)')
    parser.add_argument('--source', type=str, help='Path to the source images or video, or 0 for webcam')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold for displaying bounding boxes')
    parser.add_argument('--output', type=str, default='./output', help='Output folder for saving results')
    args = parser.parse_args()

    device = args.device
    source = args.source
    threshold = args.threshold
    output = args.output
    
    if not os.path.exists(output):
        os.makedirs(output)

    #weight = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    #model = ssdlite320_mobilenet_v3_large(weights=weight)
    
    #weight = SSD300_VGG16_Weights.DEFAULT
    #model = ssd300_vgg16(weights=weight)

    weight = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    model = retinanet_resnet50_fpn_v2(weights=weight)

    model.to(device)

    preprocess = weight.transforms()
    class_names = weight.meta['categories']

    detect(model, source, preprocess, class_names, threshold, device, output)