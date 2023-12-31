# Project Structure

```python
.
├── bin
│   ├── blue_team.pkl
│   └── red_team.pkl
├── data
│   ├── bin_2_img.py # binary --> image
│   ├── bin_add_noise.py # binary + noise --> image
│   ├── blue # train/test csv
│   ├── brutual_noise.csv
│   ├── motif_reports.csv
│   ├── processed
│   │   ├── add_noise # binary + noise
│   │   ├── add_noise_image # binary + noise --> image
│   │   ├── binary2image # raw --> image
│   │   ├── noise # noise
│   ├── raw # binary files (PE32 --> binary)
│   └── red # train/test csv
├── Dockerfile
├── Makefile
├── model.py
├── noise_extract.py
├── README.md
├── requirements.txt
├── score
├── shap.png
├── test.py # test blue team's model (original/with noise)
├── train.py # train blue/red team's model
└── utils.py

```

# 加入雜訊
1. 使用 `shap` 找出有意義的 feature
1. 直接把不同的病毒檔案加在尾端

# PE32 轉 binary + 加入雜訊
## PE32 轉 binary

最左邊的 8 個 bit 是：
> **ChatGPT**  
> 可能是檔案中的偏移位置（offset）或記憶體位址的表示。PE32（Portable Executable 32-bit）是一種用於 Windows 上可執行檔（例如 .exe）和動態連結庫（例如 .dll）的標準檔案格式。

```
00000000: 01001101 01011010 10010000 00000000 00000011 00000000  MZ....
00000006: 00000000 00000000 00000100 00000000 00000000 00000000  ......
0000000c: 11111111 11111111 00000000 00000000 10111000 00000000  ......
00000012: 00000000 00000000 00000000 00000000 00000000 00000000  ......
00000018: 01000000 00000000 00000000 00000000 00000000 00000000  @.....
0000001e: 00000000 00000000 00000000 00000000 00000000 00000000  ......
00000024: 00000000 00000000 00000000 00000000 00000000 00000000  ......
0000002a: 00000000 00000000 00000000 00000000 00000000 00000000  ......
00000030: 00000000 00000000 00000000 00000000 00000000 00000000  ......
00000036: 00000000 00000000 00000000 00000000 00000000 00000000  ......
0000003c: 11111000 00000000 00000000 00000000 00001110 00011111  ......
00000042: 10111010 00001110 00000000 10110100 00001001 11001101  ......
00000048: 00100001 10111000 00000001 01001100 11001101 00100001  !..L.!
```
參考資料：
---
- https://github.com/JiamanBettyWu/CNN-SHAP/blob/main/cnn.ipynb
- [Explain an Intermediate Layer of VGG16 on ImageNet (PyTorch)](https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/Explain%20an%20Intermediate%20Layer%20of%20VGG16%20on%20ImageNet%20%28PyTorch%29.html)