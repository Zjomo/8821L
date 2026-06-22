from ultralytics import YOLO
import torch

def main():
    # 自动下载预训练权重 yolov8n-seg.pt
    model = YOLO("yolov8n-seg.pt")

    data_yaml = r"e:\jupyter file\2_Optics\8821L\Utils\YoloUI\Fore_BackGround_70_export\data.yaml"

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=8,
        device=device,
        project="runs/segment",
        name="fore_back_70",
        exist_ok=True,
    )

    print("Training finished!")
    print(f"Best weights: runs/segment/fore_back_70/weights/best.pt")

if __name__ == "__main__":
    main()
