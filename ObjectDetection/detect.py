import argparse

import cv2
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision import transforms

def detect(model, source, preprocess, class_names):
    model.eval()
    with torch.no_grad():
        if source != 0:
            img = cv2.imread(source)
            if img is None:
                print(f"画像を読み込めませんでした: {source}")
                return
            img_tensor = transforms.ToTensor()(img).unsqueeze(0)
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

            # BGR→RGB変換
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor = preprocess(img).unsqueeze(0)  # バッチ次元追加

                with torch.no_grad():
                    outputs = model(img_tensor)[0]

                # スコアが0.5以上の検出結果のみ表示
                for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
                    if score >= 0.5:
                        x1, y1, x2, y2 = box.int().tolist()
                        class_name = class_names[label]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow('Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection with SSDLite320 MobileNet V3")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on (cpu or cuda)')
    parser.add_argument('--source', type=str, default='0', help='Path to the source images or video, or 0 for webcam')
    args = parser.parse_args()

    device = args.device
    source = args.source

    weight = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weight)
    model.to(device)

    preprocess = weight.transforms()
    class_names = weight.meta['categories']

    detect(model, source, preprocess, class_names)