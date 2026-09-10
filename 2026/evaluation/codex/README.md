# Dataset 2026 공인인증 평가 도구

이 폴더는 다음 세 데이터셋을 같은 기준으로 평가한다.

- `dataset_v2` → train
- `dataset_v2_val` → validation
- `dataset_v2_test` → test

원본 데이터는 읽기 전용으로 사용한다. 모든 결과는 기본적으로
`results/certification_2026`에 저장된다. 실행 인자를 요구하지 않으며,
경로·샘플 수·실행 범위는 `config.py` 상단 변수로 관리한다.

## 평가 구성

### 1. 데이터 카운팅 및 완전성

`count_data.py`는 conf를 scene 기준점으로 사용한다. 두 카메라의 RGB,
depth, normals, bbox, inst_seg 이미지와 mapping, pointcloud 3종과 mapping,
scene_meta를 센다. train에서는 pre_grasp와 output_grasp도 필수로 센다.
validation/test의 grasp 파일은 의도적으로 필수가 아니다.

단순 총합 외에 다음을 platform 단위로 기록한다.

- scene 번호 범위와 중간 누락
- conf에는 있지만 modality에 없는 scene
- modality에는 있지만 conf에는 없는 orphan scene
- 모든 필수 modality가 존재하는 완전 scene 수와 비율
- 규칙에 맞지 않는 파일명과 누락 폴더

결과: `data_count.json`, `data_count_platforms.csv`,
`data_count_modalities.csv`

### 2. 구문 정확성

`validate_syntax.py`는 2026 CSV의 실제 객체 ID(`obj_274`부터)를 허용
목록으로 사용한다. 기존 2025용 `obj_001~obj_273` 설정은 사용하지 않는다.

- 엄격한 JSON 파싱(NaN/Infinity 거부)
- conf의 환경·객체 transform·카메라 행렬 구조
- bbox 키, 좌표 순서와 1920×1080 범위
- inst_seg RGBA mapping과 객체 ID
- scene_meta 필드와 타입
- pre_grasp/output_grasp의 pose, bbox, joint, gripper, target 구조
- pointcloud instance mapping 구조
- 계층화 표본에 대한 PNG 디코딩/크기/모드 및 NPY shape/dtype

파일 단위 정확도와 grasp record 단위 정확도를 분리해 기록하며, 오류별
파일·scene·JSON 경로를 남긴다.

결과: `syntax_validation.json`, `syntax_validation_summary.csv`

### 3. 자동 의미 일관성

`analyze_semantics.py`는 사람이 보지 않고 판정할 수 있는 의미 관계를
독립 지표로 분석한다.

- conf 객체 ID 및 USD 파일명 ↔ 2026 CSV 객체명
- scene_meta 객체 집합/속성/Description ↔ conf 및 CSV
- inst_seg mapping foreground class ↔ conf 객체 집합
- 실제 inst_seg 픽셀 색 ↔ semantics mapping
- bbox class ↔ conf, bbox 좌표 ↔ mask의 실제 min/max 픽셀
- pointcloud instance mapping ↔ inst_seg class 및 pixel count
- pre_grasp/output_grasp target_object ↔ 해당 scene 객체

결과: `semantic_analysis.json`, `semantic_analysis_summary.csv`

### 4. inst_seg 수기 의미 정확성

`make_inst_seg_samples.py`는 split/platform/camera별로 균형 잡힌 고정
표본을 생성한다. seed와 CSV hash가 manifest에 기록되므로 표본을 재현할
수 있다. 기존 manifest는 기본적으로 덮어쓰지 않는다.

`review_inst_seg.py`에서는 한 항목씩 다음 화면을 동시에 확인한다.

- RGB 원본
- RGB + inst_seg overlay + bbox/class
- raw inst_seg
- conf class, mapping class, 색상별 픽셀 수, mask bbox, 저장 bbox

판정 키는 `1`/`O`=정상, `2`/`X`=오류, `3`/`S`=보류이다. 판정할 때마다
원자적으로 저장되며 재실행하면 미판정 항목부터 이어진다. 오류 유형과
수기 메모, 판정자, 수정 이력을 남긴다. 정확도는 보류를 제외한 O/(O+X)이고
Wilson 95% 신뢰구간도 계산한다.

결과: `manual_inst_seg_records.json/csv`, `manual_inst_seg_summary.json`

## 권장 실행 순서

수집 프로세스를 멈춘 일관된 snapshot에서 실행하는 것이 좋다.

```bash
cd /home/uon/ochansol/isaac_code/python/sanjabu/2026/evaluation/codex

# 자동 카운팅 + 구문 + 의미 분석 + 중간 보고서
/home/uon/ochansol/isaac_code/isaac_chansol/.venv/bin/python run_automatic_evaluation.py

# 수기 표본 생성(최초 한 번)
/home/uon/ochansol/isaac_code/isaac_chansol/.venv/bin/python make_inst_seg_samples.py

# 수기 판정/이어하기
/home/uon/ochansol/isaac_code/isaac_chansol/.venv/bin/python review_inst_seg.py

# 수기 결과까지 포함해 최종 표와 PNG/HTML 생성
/home/uon/ochansol/isaac_code/isaac_chansol/.venv/bin/python build_report.py
```

전체 실행 전에 빠르게 확인하려면 `config.py`에서
`SYNTAX_MAX_SCENES_PER_SPLIT`, `BINARY_PAYLOAD_SAMPLES_PER_SPLIT`,
`SEMANTIC_SAMPLES_PER_SPLIT`만 작은 정수로 바꾼다. 공인 결과에는 결과
JSON의 `scope`가 반드시 함께 제출되어야 전체 검사인지 표본 검사인지
구분할 수 있다.

## 판정 해석

이 도구는 임의의 합격 기준을 만들지 않는다. 완전성, 구문 정확성,
자동 의미 일관성, 수기 inst_seg 정확도를 각각 원시 분자/분모와 함께
제공한다. 인증기관의 합격 기준이 정해지면 최종 보고서에서 그 값과 직접
대조한다.
