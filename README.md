# PhantomLight

# Abstract

本研究旨在優化日本環球影城的魔杖軌跡辨識系統。系統由三部分組成：視覺擷取（CAM）、人工智慧（AI）、魔法實體化（MAGIC）。首先，CAM系統使用夜視攝影機捕捉魔杖尖端紅外線反光點形成的軌跡。然後，透過影像處理技術，提取魔杖動作細節。AI階段則利用訓練有素的卷積神經網絡模型，識別包括火咒、水咒等五種魔咒軌跡。最後，在魔法實體化階段，識別出的魔咒通過Arduino實現相關裝置的啟動和聲光效果，為使用者呈現真實的魔法互動體驗。

This study aims to optimize the wand trajectory recognition system at Universal Studios Japan. The system consists of three parts: visual capture (CAM), artificial intelligence (AI), and magic materialization (MAGIC). First, the CAM system uses a night vision camera to capture the trajectory of an infrared reflective dot at the tip of the wand. Then, through image processing technology, the details of the wand's movement are extracted. The AI ​​stage uses a well-trained convolutional neural network model to identify five types of magic spell trajectories, including fire spells and water spells. Finally, in the magic materialization stage, the identified magic spell is used to activate the relevant devices and achieve sound and light effects through Arduino, presenting the user with a real magic interactive experience.

# How to use?
### Clone repository

```bash
git clone https://github.com/tudohuang/PhantomLight.git
cd PhantomLight/
```

#### Run on Windows

```bash
./run.bat
```

#### Run on Linux

```bash
chmod +x run.sh
./run.sh
```

#### Directly running python

```py
python train.py --train_path your_path --model_path model_save_path
```

# Acknowledgement
本項目的成功歸功於多人的支持和幫助，
- 首先，我感謝日本環球影城，因為他們的原始魔杖軌跡辨識系統啟發了本研究的開始。
- 接著我感謝NTUST的Tengyi Huang教授以及 CGSH的Dlinda老師在這段時間的支持與教導，
- 最後，我感謝Meta公司的Pytorch，使得專案成形;Openai公司的ChatGPT，在Debug的時候幫了許多忙。

# Citation
```bibtex
@software{Huang2023PhantomLight,
  author = {Huang, Tudoh},
  title = {{PhantomLight}},
  month = {12},
  year = {2023},
  url = {https://github.com/tudohuang/PhantomLight}
}
```
