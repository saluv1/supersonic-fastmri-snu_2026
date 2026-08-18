# 2026 SNU FastMRI Challenge - Team Supersonic

2026 SNU FastMRI Challenge knee annotation track 최종 제출 코드입니다.

* 모델은 PromptMR+를 기반으로 하며, 8 GB GPU에서 학습할 수 있도록 크기를 줄였습니다.
* 제공된 train/validation 데이터만 사용하였고, 외부 데이터나 사전학습 가중치는 사용하지 않았습니다.
* 최종 채점에는 `recon_eval.sh`만 사용합니다.
* 공식 파일인 `recon_eval.py`와 `utils/common/metrics.py`는 수정하지 않았습니다.

## 1. 최종 제출 모델

| 항목 | 값 |
|---|---|
| 모델 | PromptMR+, 8 cascades |
| 파라미터 수 | 6.01M |
| 학습 epoch | 50 |
| best checkpoint epoch | 39 |
| best validation objective | 0.1075935853 |
| 체크포인트 | `../result/promptmr8_metric_aligned_50ep_v1/checkpoints/best_model.pt` |
| SHA-256 | `47530c3c029fb25674a9a582795fd05ffd272265f30c62c89b5471ec2a485e8c` |

VESSL 서버에서 측정한 공개 리더보드 결과는 다음과 같습니다.

```text
Leaderboard SSIM_full  : 0.9320
Leaderboard SSIM_bbox  : 0.9301
Leaderboard Recon Time : 794.68s (359.1 ms/slice)

SSIM_full (acc4): 0.9496   SSIM_full (acc8): 0.9145
SSIM_bbox (acc4): 0.9504   SSIM_bbox (acc8): 0.9098
Recon Time (acc4): 410.57s (360.2 ms/slice)
Recon Time (acc8): 384.10s (358.0 ms/slice)
```

원본 출력은 `experiments/recon_eval_gpu.log`에 저장되어 있습니다.

## 2. 폴더 구조

본 저장소와 데이터, 결과 폴더는 다음과 같이 형제 관계로 배치합니다.

```text
<root>/
├── FastMRI_challenge/                # 본 저장소
├── Data/
│   ├── train/
│   │   ├── image/
│   │   └── kspace/
│   ├── val/
│   │   ├── image/
│   │   └── kspace/
│   └── leaderboard/
│       ├── acc4/
│       │   ├── image/
│       │   └── kspace/
│       └── acc8/
│           ├── image/
│           └── kspace/
└── result/
    └── promptmr8_metric_aligned_50ep_v1/
        └── checkpoints/
            └── best_model.pt
```

`image`와 `kspace` 폴더에는 같은 이름의 H5 파일이 있어야 합니다.

fastMRI+ annotation은 image H5의 `attrs['annotations']`에 JSON 형식으로 저장되어 있습니다. 좌표는 384 x 384 target image를 기준으로 합니다.

## 3. 환경 설정

학습에 사용한 환경은 다음과 같습니다.

```text
Python 3.10.12
PyTorch 2.3.1+cu121
torchvision 0.18.1+cu121
NumPy 1.24.4
CUDA runtime 12.1
NVIDIA GeForce GTX 1080 8 GB
```

패키지는 아래 명령어로 설치합니다.

```bash
pip3 install -r requirements.txt
```

체크포인트는 NumPy 1.24.4 환경에서 저장했습니다. NumPy major version이 달라지면 체크포인트를 불러오지 못할 수 있으므로 NumPy 1.x 환경을 사용해야 합니다.

전체 패키지 목록은 `experiments/evidence/pip_freeze.txt`에 있습니다.

## 4. 경로 설정

### 4-1. 전제 폴더 구조

공식 `recon_eval.py`가 CWD 기준 `'../result'`를 사용하므로, 아래 세 폴더가
형제 관계여야 합니다.

```text
<root>/
├── FastMRI_challenge/    ← 본 저장소
├── Data/
│   ├── train/  {image, kspace}
│   ├── val/    {image, kspace}
│   └── leaderboard/  {acc4, acc8}/{image, kspace}
└── result/
    └── promptmr8_metric_aligned_50ep_v1/
        └── checkpoints/best_model.pt
```

이 구조라면 인자 없이 그대로 실행됩니다. 셸 스크립트가 자신의 위치로 이동하므로
**어느 디렉터리에서 호출해도 무방합니다.**

```bash
bash /any/path/FastMRI_challenge/recon_eval.sh    # 채점
bash /any/path/FastMRI_challenge/train.sh         # 학습
```

### 4-2. 데이터 위치가 다른 경우

코드를 수정하지 마시고 인자로 넘겨 주십시오.

```bash
bash recon_eval.sh /path/to/leaderboard
bash train.sh      /path/to/Data          # 하위에 train/, val/ 가 있는 디렉터리
bash train.sh      /path/to/Data  my_run  # 실험 이름도 변경
GPU_NUM=1 bash recon_eval.sh              # GPU 번호 변경
```

### 4-3. 체크포인트 위치

```text
../result/promptmr8_metric_aligned_50ep_v1/checkpoints/best_model.pt
SHA-256: 47530c3c029fb25674a9a582795fd05ffd272265f30c62c89b5471ec2a485e8c
```

파일명은 `best_model.pt` 여야 합니다 (`utils/learning/test_part.py::load_model`).
용량 문제로 저장소에 포함하지 않았으며, VESSL 워크스페이스의
`/root/result/promptmr8_metric_aligned_50ep_v1/checkpoints/best_model.pt`
및 제출 압축파일에 있습니다.

### 4-4. 절대경로 제거 내역

초기 준비 단계에서 학습 서버(`/root/…`) 기준 절대경로가 남아 있었으며 전부
제거했습니다. 현재 `backups/`를 제외한 모든 `.py` / `.sh`에 절대경로가 없습니다.

**셸 스크립트 4종** (`train.sh`, `recon_eval.sh`, `reconstruct.sh`,
`leaderboard_eval.sh`)

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"; pwd)"
cd "$SCRIPT_DIR"
DATA_ROOT="${1:-$SCRIPT_DIR/../Data}"
NET_NAME="${2:-promptmr8_metric_aligned_50ep_v1}"
GPU_NUM="${GPU_NUM:-0}"
```

* `cd /root/FastMRI_challenge` → `cd "$SCRIPT_DIR"`
* 데이터·출력 경로를 전부 `$SCRIPT_DIR` 기준 또는 인자로 전환
* `reconstruct.sh`의 죽은 기본값 `test_Varnet` 제거

**파이썬 argparse 기본값** (`train.py`, `smoke_test_bbox_train.py`,
`smoke_test_metric_aligned_train.py`)

```python
PROJECT_ROOT = Path(__file__).resolve().parent
default=PROJECT_ROOT.parent / "Data" / "train"     # was "/root/Data/train/"
default=PROJECT_ROOT.parent / "Data" / "val"       # was "/root/Data/val/"
```

`train.py`의 출력 경로도 CWD 의존성을 없앴습니다.

```python
result_root = PROJECT_ROOT.parent / "result" / args.net_name   # was Path("../result")
```

셸 스크립트가 CWD를 저장소 루트로 고정하므로, 이 값은 공식 `recon_eval.py`의
`'../result'`와 항상 동일한 경로로 해석됩니다.

**학습 하이퍼파라미터는 변경하지 않았습니다.** 수정 전후 `train.sh`의 argparse
옵션 집합이 완전히 동일함을 확인했습니다.

### 4-5. `../result`를 유지한 이유

`recon_eval.py`(119·128행), `recon_eval_cpuonly.py`, `reconstruct.py`의
`'../result'`는 **의도적으로 유지**했습니다. `recon_eval.py`는 수정 금지 대상인
공식 파일이며 이 규약을 하드코딩하고 있으므로, 나머지 코드를 여기에 맞췄습니다.

또한 `recon_eval.py`는 sys.path를 `os.getcwd() + '/utils/model/'`로 설정하므로
(22–23행) 실행 위치에 의존합니다. 공식 파일이라 수정할 수 없어, 셸 스크립트가
자신의 위치로 이동하도록 하여 해결했습니다.

### 4-6. 검증

저장소 밖(`/tmp`)에서 호출해도 동일한 점수가 재현됨을 확인했습니다.

```bash
$ cd /tmp && bash /root/FastMRI_challenge/recon_eval.sh \
    /root/Data/leaderboard promptmr8_metric_aligned_50ep_v1

Leaderboard SSIM_full  : 0.9320
Leaderboard SSIM_bbox  : 0.9301
Leaderboard Recon Time : 793.91s (358.7 ms/slice)
```

저장소 루트에서 실행한 결과(`experiments/recon_eval_gpu.log`)와 SSIM이 완전히
일치합니다.

## 5. 학습

학습은 `train.sh` 하나로 실행합니다.

```bash
cd FastMRI_challenge
bash train.sh
```

최종 학습 설정은 다음과 같습니다.

| 항목 | 값 |
|---|---|
| Epochs | 50 |
| Batch size | 1 |
| Optimizer | AdamW, weight decay 0.01 |
| Initial learning rate | 2e-4 |
| LR scheduler | MultiStepLR |
| LR milestones | 16, 27 |
| LR gamma | 0.3 |
| Seed | 430 |
| Iterations per epoch | 4651 |

학습 시간은 epoch당 약 3시간이며 전체 학습에는 약 150시간이 소요되었습니다.

학습 결과는 다음 폴더에 저장됩니다.

```text
../result/promptmr8_metric_aligned_50ep_v1/
├── checkpoints/
│   ├── model.pt
│   └── best_model.pt
├── reconstructions_val/
└── val_loss_log.npy
```

`best_model.pt`는 다음 validation objective가 가장 낮은 epoch에서 저장됩니다.

```text
val_objective = (1 - SSIM_full) + 0.3 * (1 - SSIM_bbox)
```

최종 제출 체크포인트는 epoch index 38의 검증 결과로 선택되었고, 체크포인트의 `epoch` 값은 39입니다.

## 6. 모델 설정

PromptMR+의 기본 구조를 사용하되, 8 GB GPU에서 학습할 수 있도록 feature 크기를 줄였습니다.

| 파라미터 | 사용 값 | PromptMR+ 원 설정 |
|---|---:|---:|
| `num_cascades` | 8 | 12 |
| `n_feat0` | 8 | 48 |
| `feature_dim` | [24, 32, 40] | [72, 96, 120] |
| `prompt_dim` | [8, 16, 24] | [24, 48, 72] |
| `sens_n_feat0` | 8 | 24 |
| `sens_feature_dim` | [12, 16, 20] | - |
| `sens_prompt_dim` | [4, 8, 12] | - |
| `n_history` | 3 | 11 |
| `num_adj_slices` | 1 | - |
| `compute_sens_per_coil` | True | - |

`compute_sens_per_coil=True`를 사용하여 sensitivity map을 coil 단위로 계산했습니다. GTX 1080에서 coil 병렬 계산보다 메모리 사용량이 적고 추론 시간도 조금 더 빨랐습니다.

학습 시에는 gradient checkpointing을 사용했습니다.

```text
--use-checkpoint true
```

추론에서는 gradient checkpointing을 사용하지 않습니다.

## 7. Loss

학습 loss는 foreground SSIM loss와 bbox SSIM loss를 함께 사용합니다.

```text
loss = foreground_ssim_loss + 0.3 * bbox_ssim_loss
```

* foreground loss는 공식 foreground mask 내부의 SSIM으로 계산합니다.
* bbox loss는 각 annotation box별 SSIM을 평균하여 계산합니다.
* annotation이 없는 slice에서는 bbox loss를 적용하지 않습니다.
* validation은 공식 metric과 동일하게 acc4와 acc8을 각각 계산한 뒤 동일한 비중으로 평균합니다.

## 8. Data augmentation

### K-space mask augmentation

`utils/data/mask_augment.py`에서 acc4와 acc8 sampling mask를 새로 생성합니다.

```text
--mask-aug true
--mask-aug-weight 1.0
--mask-aug-start 16
--mask-aug-schedule exp
--mask-aug-plateau-epoch 25
--mask-aug-accelerations 4 8
--mask-aug-random-ratio 0.0
--mask-aug-random-offset true
```

mask augmentation은 epoch 16부터 적용되며 epoch 25에서 최대 확률에 도달합니다.

### MRAugment

MRAugment는 image domain의 spatial transform을 k-space에 반영하는 방식으로 적용했습니다. annotation box에도 같은 좌표 변환을 적용하여 target과 box가 어긋나지 않도록 했습니다.

```text
--aug_on
--aug_schedule exp
--aug_delay 30
--aug_strength 0.5
--aug_exp_decay 5.0
--aug_weight_translation 0.1
--aug_weight_rotation 0.1
--aug_weight_shearing 0.1
--aug_weight_scaling 1.0
--aug_weight_fliph 0.4
--aug_weight_flipv 0.0
--aug_weight_rot90 0.0
```

epoch 30까지는 augmentation 확률이 0이며 epoch 31부터 적용됩니다. Knee 영상의 방향성을 유지하기 위해 vertical flip과 90도 rotation은 사용하지 않았습니다.

## 9. Reconstruction 및 평가

최종 평가는 `recon_eval.sh`로 실행합니다.

```bash
cd FastMRI_challenge
bash recon_eval.sh
```

`recon_eval.sh`는 다음 checkpoint를 불러옵니다.

```text
../result/promptmr8_metric_aligned_50ep_v1/checkpoints/best_model.pt
```

`recon_eval.py`는 reconstruction과 SSIM 계산을 한 번에 수행하며 다음 세 값을 출력합니다.

* SSIM_full
* SSIM_bbox
* slice당 reconstruction time

최종 채점에는 `recon_eval.sh`만 사용합니다. 아래 파일은 개발 및 확인 과정에서 사용한 것으로 최종 채점에는 사용하지 않습니다.

* `recon_eval_cpuonly.py`: CPU에서 SSIM을 확인하기 위한 스크립트
* `reconstruct.py`, `leaderboard_eval.py`: baseline의 분리 실행 경로
* `benchmark_sens_per_coil.py`: sensitivity map 계산 방식 비교
* `smoke_test_*.py`: 학습과 annotation 정렬 확인

## 10. 추론 규칙

`utils/learning/test_part.py`는 공식 `recon_eval.py`에서 요구하는 세 함수를 구현합니다.

### `load_model()`

`best_model.pt`에 저장된 학습 인자로 PromptMR+를 생성하고 weight를 불러옵니다.

### `prep_volume()`

H5 파일에서 원본 k-space와 sampling mask를 읽어 host memory에 보관합니다. 모델 연산이나 reconstruction은 수행하지 않으며, 중간 reconstruction이나 수정된 weight를 `ctx`에 저장하지 않습니다.

### `recon_slice()`

해당 slice의 mask 적용, tensor 변환, device 이동 및 model forward를 수행합니다. 실제 reconstruction 연산은 모두 이 함수 안에서 수행되므로 공식 per-slice 시간 측정에 포함됩니다.

추론 시에는 image H5의 GRAPPA, annotation 및 bbox 정보를 사용하지 않습니다. 모델 입력은 k-space와 sampling mask뿐입니다.

## 11. 재현성

학습 seed는 430으로 고정했습니다.

`utils/common/utils.py`의 `seed_fix()`에서 다음 항목을 설정합니다.

```python
torch.manual_seed(n)
torch.cuda.manual_seed(n)
torch.cuda.manual_seed_all(n)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(n)
random.seed(n)
```

또한 mask augmentation과 MRAugment는 각각 별도의 seeded `RandomState`를 사용합니다. DataLoader는 별도 worker를 사용하지 않으므로 worker별 seed 차이가 없습니다.

CUDA 연산 특성상 전체 학습이 bit-identical하게 재현되지는 않습니다. 초기 weight와 데이터 순서는 동일하게 재현되지만, 일부 CUDA backward 연산의 부동소수점 누적 순서로 인해 iteration별 loss는 소수점 아래에서 조금씩 달라질 수 있습니다.

4 epoch 재학습에서 iteration 0 loss는 원본 학습과 같은 `0.6352`였고, 이후 loss는 네 번째 유효숫자 부근부터 차이가 발생했습니다. 해당 로그는 `experiments/repro4_seed430.log`에 저장했습니다.

재현성 관련 파일은 다음과 같습니다.

| 파일 | 내용 |
|---|---|
| `experiments/train_stdout.log` | 50 epoch 전체 학습 로그 |
| `experiments/repro4_seed430.log` | seed 430 초기 4 epoch 재학습 로그 |
| `experiments/val_loss_log.npy` | epoch별 loss 기록 |
| `experiments/code_before_train.tar.gz` | 최종 학습 직전 코드 스냅샷 |
| `experiments/evidence/pip_freeze.txt` | 전체 Python 패키지 목록 |
| `experiments/recon_eval_gpu.log` | 최종 GPU 평가 로그 |
| `experiments/recon_eval_cpu.log` | CPU/GPU SSIM 교차 확인 로그 |

VESSL 학습 터미널도 삭제하지 않고 보존했습니다.

## 12. Validation 결과

50 epoch 학습의 마지막 epoch 결과는 다음과 같습니다.

| Acceleration | SSIM_full | SSIM_bbox |
|---|---:|---:|
| acc4 | 0.9222 | 0.9455 |
| acc8 | 0.8972 | 0.9383 |
| 평균 | 0.9097 | 0.9419 |

최종 checkpoint는 마지막 epoch가 아니라 validation objective가 가장 낮았던 epoch 39 checkpoint입니다.

## 13. 주요 파일

```text
FastMRI_challenge/
├── README.md
├── requirements.txt
├── train.py
├── train.sh
├── recon_eval.py
├── recon_eval.sh
├── experiments/
│   ├── train_stdout.log
│   ├── repro4_seed430.log
│   ├── recon_eval_gpu.log
│   ├── recon_eval_cpu.log
│   ├── val_loss_log.npy
│   └── evidence/
│       └── pip_freeze.txt
└── utils/
    ├── common/
    │   ├── loss_function.py
    │   ├── metrics.py
    │   └── utils.py
    ├── data/
    │   ├── annotation_utils.py
    │   ├── load_data.py
    │   ├── mask_augment.py
    │   ├── transforms.py
    │   └── mraugment/
    ├── learning/
    │   ├── train_part.py
    │   └── test_part.py
    └── model/
        ├── promptmr_plus.py
        ├── reentrant_wrapper.py
        └── fastmri/
```
```
```
## 14. References

1. Xin, B., Ye, M., Axel, L., and Metaxas, D. N. *Rethinking Deep Unrolled Model for Accelerated MRI Reconstruction*. ECCV 2024.
2. Fabian, Z., Heckel, R., and Soltanolkotabi, M. *Data Augmentation for Deep Learning Based Accelerated MRI Reconstruction with Limited Data*. ICML 2021.
3. Zbontar, J. et al. *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI*. The fastMRI components included in `utils/model/fastmri/` are distributed under the MIT License.
4. Zhao, R. et al. *fastMRI+: Clinical Pathology Annotations for the fastMRI Dataset*.
