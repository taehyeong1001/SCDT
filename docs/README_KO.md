# 실행 안내

참고 저장소처럼 시스템별 `train.py`와 `test.py`를 실행하는 구조로 정리했습니다.

| 폴더 | 역할 |
|---|---|
| `System_Food_Chain` | Food 훈련·테스트와 해당 ODE/레저버 구현 |
| `System_Power_System` | Power 훈련·테스트와 해당 ODE/레저버 구현 |
| `System_Kuramoto` | Kuramoto 훈련·테스트와 해당 오실레이터/네트워크 구현 |
| `common` | 공통 경로, 모델 로딩, warm-up, 판정 및 결과 저장 |
| `models`, `data` | 최종 가중치, 훈련 시퀀스, 제출 수치 및 측정 기록 |
| `tools` | 부가 기능: 그림 생성, GT 계산, 검증 |

Food의 논문 결과 재현 예시는 다음과 같습니다. 훈련을 먼저 실행할 필요는 없으며, 나머지는 폴더 이름만 바꾸면 됩니다.

```bash
python System_Food_Chain/test.py
```

훈련은 포함된 훈련 시퀀스와 고정된 레저버 행렬을 사용해 readout을 다시 학습합니다. BO, 무작위 행렬 생성, 모든 훈련 ODE의 재적분까지 수행하는 것은 아닙니다.

재학습 결과 자체를 테스트할 때는 다음처럼 명시적으로 지정합니다.

```bash
python System_Food_Chain/train.py
python System_Food_Chain/test.py --readout outputs/training/food/readout.npz --output outputs/refit_evaluation
```

테스트는 훈련 파일의 유무와 관계없이 기본으로 제출 모델을 사용합니다. `train.py`를 실행해도 기본 모델은 바뀌지 않습니다. 새 가중치를 평가하려면 `--readout outputs/training/food/readout.npz`처럼 직접 지정해야 합니다. `--frozen`은 기본 동작과 같으며 이전 명령과의 호환을 위해 유지했습니다. 사용한 모델은 화면과 평가 기록에 표시합니다.

Power 재학습 readout은 제출본과 약 3 × 10⁻⁹의 상대 오차가 있으며, 긴 자율 예측에서 crossing 근처 조건의 판정을 바꿀 수 있습니다. 따라서 논문 수치 재현과 readout 재학습 실험은 구분해야 합니다. 새 가중치 실험은 `--output outputs/refit_evaluation` 등 별도 출력 폴더를 권장합니다.

테스트 기본 작업은 전체 rate 평가입니다. `--task selected`는 inter/pre/post, `--task reconstruction`은 Food/Power의 통계 재구성을 계산합니다. `--condition 0 --members 2`처럼 줄이면 빠른 점검용이며 논문 앙상블 결과가 아닙니다. 모든 warm-up은 훈련 궤적을 사용합니다.

`python tools/render_figures.py`는 저장된 수치로 그림을 재출력합니다. 새 학습 결과를 자동으로 논문 그림에 넣는 것은 아닙니다. `python tools/verify.py`는 최종 좌표·판정·파일 해시를 확인합니다. 기존 `run.py` 명령도 호환용으로 유지했습니다.

수치·가중치·Supplementary 원본은 바꾸지 않았습니다. Food 재구성 통계의 기존 표 대비 최대 약 0.00198 차이는 그대로 검증 기록에 명시합니다. 자세한 설정과 제한은 `REPRODUCTION.md`, `VALIDATION_KO.md`, `provenance/packaging_validation.json`에 있습니다.

이전 폴더 구조는 로컬 보관 폴더에 백업되어 있으며 실행에 필요하지 않습니다. GitHub 업로드 및 라이선스 결정은 아직 진행하지 않았습니다.
