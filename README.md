# Shoppingmall_Review
## 1. Project Overview
  - 목표
    - 상품 리뷰 텍스트로 상품의 평점을 예측.
    - 주어진 쇼핑몰 리뷰 데이터셋을 이용하여 상품의 평점(1점, 2점, 4점, 5점)을 분류.
  - 모델
    - [klue/bert-base](https://github.com/KLUE-benchmark/KLUE) fine-tuned model.
  - Data
    - 쇼핑몰 리뷰 데이터셋 (train: 25000개, test: 25000개).

## 2. Code Structure
``` text
├── dataset
|   ├── stopword.txt
|   ├── train.csv (not in repo)
|   └── test.csv  (not in repo)
├── argument.py
├── dataset.py
├── inference.py
├── preprocess.py
└── train.py
```

## 3. Detail 
  - Preprocess 
    - 정규표현식을 이용한 특수문자 제거.
    - [Open Korean Text Processor](https://github.com/open-korean-text/open-korean-text) 를 이용한 Stopword(출처: https://www.ranks.nl/stopwords/korean) 제거.
    - [hanspell](https://github.com/ssut/py-hanspell)을 이용한 맞춤법 교정.
  - Model
    - 전처리 된 데이터를 klue/bert-base에 fine-tunning 과정을 거침.
    - K-Fold(5 Fold)의 교차 검증과증을 통해 보다 정교한 모델 제작(Public Accuracy 기준 0.02 상승)
    - Batch size: 16 / Epoch : 5
  - Inference
    - 각 Fold model의 inference 과정을 거친후 나온 softmax 확률을 soft-voting 과정을 거쳐 최종추론 함.
  - 최종성적
    - Public Score: 0.6972
    - Private Score: TBD
