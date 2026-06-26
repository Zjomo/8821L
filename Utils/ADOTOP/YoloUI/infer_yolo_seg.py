from ultralytics import YOLO
import os
import glob

def main():
    weights_path = r"e:\jupyter file\2_Optics\8821L\Utils\YoloUI\runs\segment\fore_back_70\weights\best.pt"
    
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        print("Please train the model first.")
        return

    model = YOLO(weights_path)

    # 推理示例：对 val 集的前几张图片推理
    val_img_dir = r"e:\jupyter file\2_Optics\8821L\Utils\YoloUI\Fore_BackGround_70_export\images\val"
    images = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))[:5]

    if not images:
        print("No images found in val set.")
        return

    results = model.predict(source=images, save=True, project="runs/segment/predict", name="exp", exist_ok=True)
    print(f"Inference done. Results saved to runs/segment/predict/exp")

if __name__ == "__main__":
    main()
