
# ConvNeXt Implementation for 2022 AI Cup (Optimized for Apple M-series GPUs)

## 農地作物現況調查影像辨識競賽 - 秋季賽：AI作物影像判釋

本專案基於 ConvNeXt 模型進行實作，並針對 **macOS + Apple Silicon (M1/M3/M4)** 環境進行深度優化，包含：

- 支援 AMP（自動混合精度訓練，使用 bfloat16），將每個 epoch 時間從約 2 天縮短至 5 小時
- 使用 MPS 後端時支援的 batch size 可達 24，超過 RTX 3060 Ti 上的 12
- 現在可將 num_workers 設成小於 8 的數字。以往DataLoader 的 num_workers 設定為大於 0 時，整體訓練速度反而比 num_workers=0 更慢。
- 支援 Channels Last 格式以提升卷積效率

---
### 1. 環境安裝

可參考 [ConvNeXt](https://github.com/facebookresearch/ConvNeXt/blob/main/INSTALL.md) 官網之安裝方式:

Create a new conda virtual environment
```
conda create -n convnext python=3.8 -y
conda activate convnext
```
Install [Pytorch](https://pytorch.org/)>=1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.9.0 following official instructions. For example:
```
pip3 install torch torchvision torchaudio
pip install timm==0.3.2 tensorboardX six
```


註: 如果安裝之 pytorch > 1.8.0, 由於 torch._six 將部分 module 移除會導致錯誤出現, 建議將 timm helpers 裡的

from torch._six import container_abcs

修改為

import _collections_abc as container_abcs

即可正常運作

### 2. 資料集結構

```
/path/to/ai_cup_dataset/
  train/
    class1/
      img1.jpg
    class2/
      img2.jpg
  val/
    class1/
      img3.jpg
    class2/
      img4.jpg
  test/
    0/
      img5.jpg
    1/
      img6.jpg

```

### 3. 執行

1 複製本程式至工作目錄

2. 如果工作目錄裡沒有 pretrained ConvNext_B 模型, 請至下面網址下載
https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth

3. 此專案所用之命令 (基於 MacBook air M3 with 10 core CPU, 10 core GPU, 16GB RAM, 512GB SSD)

python3 main_ai_cup_base_ema.py --model convnext_base --batch_size 12 --update_freq 21 --auto_resume False --input_size 360 --drop_path 0.5 --data_path /Volumes/VAP-512G/train --eval_data_path /Volumes/VAP-512G/val --aa rand-m9-mstd0.5-inc1 --use_amp False --data_set image_folder --nb_classes 33 --lr 0.0001 --output_dir /Volumes/VAP-512G/OUTPUT --weight_decay 1e-6 --epochs 90 --model_ema true --model_ema_eval true --num_workers 8

若需 debug，可使用此 flags 陣列：
["--model","convnext_base","--batch_size","24","--device","mps","--update_freq","21","--auto_resume","False",
 "--input_size","360","--drop_path","0.5","--data_path","/Volumes/VAP-512G/train",
 "--eval_data_path","/Volumes/VAP-512G/val","--aa","rand-m9-mstd0.5-inc1","--use_amp","True",
 "--data_set","image_folder","--nb_classes","33","--lr","0.0001",
 "--output_dir","/Volumes/VAP-512G/OUTPUT","--weight_decay","1e-6","--epochs","90",
 "--model_ema","true","--model_ema_eval","true","--num_workers","8"]


由於上述 command 是將 ema 開啟, 因此會儲存一般與 ema 兩種模型, 經過測試, ema 模型結果比較好, 建議採用該組結果
