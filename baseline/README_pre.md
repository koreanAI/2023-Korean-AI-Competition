
# Dataset
> 본 문서는 2023 한국어 AI 경진대회 예선 데이터셋에 대한 설명입니다.
### Dataset Name
`TRACK1 - 소리로 더 가까이, 노인 및 소아 계층 특화 음성 인식` : `Tr1Ko`  
`TRACK2 - KB국민은행: 상담 음성 인식(금융)` : `Tr2KB`  
`TRACK2 - 더존비즈온: 비대면 진료를 위한 음성 인식(의료 분야)` : `Tr2DZ`  

`rootpath = nova.DATSET_PATH`
### Train Dataset

`DATASET_PATH/train/train_data/`  
- `train_data (wav 형식)`
  - 파일명: idx_000000 ~ 
  - 샘플링 주파수: 16000Hz
  - Mono Channel


### Train Label

`DATASET_PATH/train/train_label`
  - `train_label (csv 형식)`
    - columns - `["filename", "text"]`
    - `filename` - train_data 폴더에 존재하는 파일명 (ex. idx_000000)
    - `text` - train_data 폴더에 존재하는 파일의 음성 전사 Text 정보 (ex. 인공지능 훈민정음에 꽃 피우다)


# Baseline code
- `main.py` : 실행파일
- `setup.py`: 환경설정(Base Docker Image, Python libraries)
- `nova_package.txt`: packages(by apt or yum)
- `modules`
  - `audio` : 오디오 모듈(parsing)
  - `data.py` : 데이터 로더
  - `inference.py`: 인퍼런싱
  - `metrics.py` : 평가지표 관련(CER)
  - `model.py`: 모델 빌드 관련(DeepSpeech2)
  - `preprocess.py`: 전처리(라벨/transcripts 제작)
  - `trainer.py`: 학습 관련
  - `utils.py` : 기타 설정 및 필요 함수
  - `vocab.py` : Vocabulary Class 파일

## 실행 방법
```bash
# 명칭이 'Tr1KO'인 데이터셋을 사용해 세션 실행하기
$ nova run -d Tr1Ko
# 메인 파일명이 'main.py'가 아닌 경우('-e' 옵션으로 entry point 지정)
# 예: nova run -d Tr1Ko -e anotherfile.py
$ nova run -d Tr1Ko -e [파일명]
# 2GPU와 16CPU, 160GB 메모리를 사용하여 세션 실행하기   
$ nova run -d Tr1Ko -g 2 -c 16 --memory 160G  

# 세션 로그 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nova logs [세션명]

# 세션 종료 후 모델 목록 및 제출하고자 하는 모델의 checkpoint 번호 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nova model ls [세션명]

# 모델 제출 전 제출 코드에 문제가 없는지 점검하기('-t' 옵션)
$ nova submit -t [세션명] [모델_checkpoint_번호]

# 모델 제출하기
# 제출 후 리더보드에서 점수 확인 가능
$ nova submit [세션명] [모델_checkpoint_번호]
```

본 베이스라인 코드는 김수환 님께서 개발해 공개하신 kospeech (https://github.com/sooftware/kospeech) 를 기반으로 하였으며 
nova 플랫폼에서 사용 가능한 형태로 수정하였습니다.
