# ML-Final

1.安裝 Miniconda（Linux 環境）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version

2.建立 YOLOv8 訓練環境
conda create -n yolov8 python=3.11 -y
conda activate yolov8

3.安裝 PyTorch + CUDA（支援 GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

4. 安裝 YOLOv8
pip install ultralytics
yolo help

5.準備資料集與 YAML 設定
下載fisheye8k dataset與FishEye1Keval test data
將 FishEye8K 的影像依照以下結構擺放：
configs/
├── fisheye8k.yaml
dataset/
├── fisheye8k
         ├── train/images/
         ├── train/labels/
         ├── val/images/
         ├── val/labels/
         ├── test/images/
建立 configs/fisheye8k.yaml：
# configs/fisheye8k.yaml
train: /path/to/dataset/train/images
val: /path/to/dataset/val/images
test: /path/to/dataset/test/images
nc: 5
names: ['bus', 'bike', 'car', 'pedestrian', 'truck']

6.訓練 YOLOv8n 模型
yolo detect train \
  model=yolov8n.pt \
  data=configs/fisheye8k.yaml \
  imgsz=640 \
  batch=8 \
  epochs=500 \
  patience=20 \
  name=yolov8_fisheye_baseline

7.測試模型
yolo detect predict \
model=runs/detect/yolov8_fisheye_baseline_500epoch_20patience/weights/best.pt \
source='/your/pic/path/' \
  imgsz=640 \
  save \
  save_txt \
  save_conf \
  save_crop \
  name=pred_yolov12
