# **재활용 품목 분류를 위한 Object Detection**

## Project Overview
### 프로젝트 목표
 - 임의의 사진이 주어졌을 때, **쓰레기를 Detection** 하는 모델 제작

### 기대 효과
- 바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다. 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다. 따라서 우리는 사진에서 쓰레기를 Detection 하는 모델의 제작을 통해 이러한 문제점을 해결할 수 있습니다.

### Dataset
- 쓰레기가 포함된 사진 및 annoations *(10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing)*
- 해상도 : 1024, 1024
- 총 이미지 수 : 9,754장(Train 이미지 수 : 4,883장)

### Framework
- MMDetection, Detectron2

### 협업 tools
- Slack, Notion, Github, Wandb

### GPU
- V100(vram 32GB) 5개


### 평가기준
- mAP50


<br>
  
## Team Introduction
### Members
| 고금강 | 김동우 | 박준일 | 임재규 | 최지욱 |
|:--:|:--:|:--:|:--:|:--:|
|<img  src='https://avatars.githubusercontent.com/u/101968683?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/113488324?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/106866130?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/77265704?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/78603611?v=4'  height=80  width=80px></img>|
|[Github](https://github.com/TwinKay)|[Github](https://github.com/dwkim8155)|[Github](https://github.com/Parkjoonil)|[Github](https://github.com/Peachypie98)|[Github](https://github.com/guk98)|
|twinkay@yonsei.ac.kr|dwkim8155@gmail.com|joonil2613@gmail.com|jaekyu.1998.bliz@gmail.com|guk9898@gmail.com|

### Members' Role

| 팀원 | 역할 |
| -- | -- |
| 고금강_T5011 | - MMDetection 라이브러리 실험 <br> - Swin Transformer Large 구현 <br> - Data Augmentation Experiments <br> - Label Correction |
| 김동우_T5026 | - MMDetection 라이브러리 실험(VFNet, FocalNet, UniverseNet) <br> - BBox EDA <br> - Pseudo Labeling 구현 |
| 박준일_T5094 | - Detectron2 라이브러리 실험 <br> - TridentNet 구현 <br> - Label Correction <br> - Model 선정 |
| 임재규_T5174 | - MMDetection와 Detectron2 라이브러리 실험 <br> - CutOut, CutMix, MixUp 등 Detectron2 데이터 증강 기법 구현 <br> - 데이터셋과 모델 추론 결과 Bounding Boxes 시각화 구현 <br> - 데이터셋 라벨 조사 |
| 최지욱_T5219 | - MMDetection을 이용한 모델 실험(Deformable DETR, RetinaNet) + YoloV8 <br> - Stratified Group K-fold <br> - Weighted Boxes Fusion <br> - Confidence score calibration |

<br>

## Procedure & Techniques

  

| 분류 | 내용 |
| :--: | -- |
|Data|**Stratified Group K-fold** <br> - 하나의 이미지가 하나의 class에 할당되는 것이 아닌 여러 개의 object(class)를 포함 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> object들의 class별 분포가 최대한 유사하도록 각각 5개의 Train/Valid set(8:2로 분할)을 구성 <br> <br>  **Augmentation** <br> - 각 모델에 기본적인 데이터 증강으로 Horizontal Flip과 Vertical Flip을 적용 <br> - 그 외에도 Rotate, Sharpen, Emboss 등 다양한 augmentation 사용  <br> - 다양한 augmentation을 적용할수록 더 높은 mAP 점수를 보임 <br> <br> **Label Correction** <br> - train dataset의 Paper와 General Trash의 경계가 애매모호하다는 것을 확인 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> 라벨링 기준을 정하여 Correction을 한 결과, mAP50 점수가 상승되었다. (0.5371->0.5420)
|Model|**Cascade-RCNN** <br> - Backbone : Swin-L <br> - Neck : FPN <br> - Head : Cascade-RCNN <br> <br> **ATSS** <br> - Backbone : Swin-L <br> - Neck : FPN <br> - Head : ATSS + Dyhead <br> <br> **Deformable DETR** <br> - Backbone : Swin-L <br> - Neck : Channel Mapper <br> - Head : Deformable DETR Head
|HyperParameters|**Cascade-RCNN** <br> - Batch Size : 32 <br> - Class Loss : Cross Entropy <br> - BoundingBox Loss : Smooth-L1 <br> - Learning Rate : 0.0001 <br> - Optimizer : AdamW <br> - Epochs : 13 <br> <br> **ATSS** <br> - Batch Size : 32 <br> - Class Loss : Focal Loss <br> - BoundingBox Loss : GioU Loss <br> - Learning Rate : 0.00005 <br> - Optimizer : AdamW <br> - Epochs : 18 <br> <br> **DETR** <br> - Batch Size : 32 <br> - Class Loss : Focal Loss <br> - BoundingBox Loss : L1-Loss <br> - Learning Rate : 0.0002 <br> - Optimizer : AdamW <br> - Epochs : 21
|Other Methods|**Ensemble** <br> - Weighted Boxes Fusion <br> - Confidence score calibration 적용 <br> <br>  **Pseudo Labeling** <br> - 주어진 Train dataset 뿐만 아니라 label이 없는 Test dataset까지 학습에 이용해서 모델 성능을 최대한 향상시키기 위함 <br> - ATSS 1epoch 적용 (Public mAP : 0.7157 -> 0.7185)

<br>

## Results

### 단일모델

| Method | Backbone | mAP50 | mAP75 | mAP50(LB) |
| :--: | :--: | :--: | :--: | :--: |
|Faster RCNN| ResNet101| 0.4845| 0.313 |0.4683|
|DetectoRS| ResNext101| 0.514 |0.385 |0.4801|
|TridentNet |Trident + ResNet101| 0.5341| 0.4311| 0.5428|
|**Cascade RCNN**| Swin-L |0.633| 0.539| 0.6257|
|**Deformable DETR**| Swin-L| 0.621 |0.533| 0.6373|
|**ATSS**| Swin-L| 0.689| 0.596| 0.6741|

<br>

### 앙상블
| Emsemble | Calibration | mAP50(LB) |
| :--: | :--: | :--: |
|ATSS (5Fold), Deformable DETR (5Fold), Swin-L + Cascade (5Fold)| |0.7054|
|ATSS (5Fold), Deformable DETR (5Fold), Swin-L + Cascade (5Fold)| ✔|0.7116|
|ATSS + Pseudo (5Fold), Deformable DETR (5Fold), Swin-L + Cascade (5Fold)| ✔|0.7185|

### 최종 과정 및 결과
<img  src='https://i.esdrop.com/d/f/JnUd4EePHi/RtXESQ1EMi.png'  height=530  width=900px></img>

### 최종 순위
- 🥈 **Public LB : 2nd / 19**
<img  src='https://i.esdrop.com/d/f/JnUd4EePHi/pRxm1J5V4K.png'  height=200  width=900px></img>
- 🥈 **Private LB : 2nd / 19**
<img  src='https://i.esdrop.com/d/f/JnUd4EePHi/lH6U2wutr6.png'  height=200  width=900px></img>