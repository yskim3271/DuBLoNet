# LaCoSENet

## 프로젝트 목적

LaCoSENet 음성 향상(speech enhancement) 모델에 대한 학회 논문 작성을 위한 학습 및 실험 프로젝트.
- **모델**: Backbone — 주파수 도메인 기반 causal speech enhancement 모델
- **핵심 연구**: 비대칭 convolution의 padding ratio를 조절하여 algorithmic latency를 제어하고, state buffer + lookahead buffer 기반 streaming 추론 구현
- **데이터셋**: VoiceBank-DEMAND 16kHz (HuggingFace: `JacobLinCool/VoiceBank-DEMAND-16k`)
- **평가 지표**: PESQ, STOI, CSIG, CBAK, COVL, segSNR

## 코드베이스 구성

```
LaCoSENet/
├── conf/
│   └── config.yaml              # Hydra 학습 설정 (모델, 데이터셋, 손실함수, 최적화)
├── src/
│   ├── train.py                 # 학습 엔트리포인트 (Hydra)
│   ├── evaluate.py              # 평가 스크립트 (PESQ/STOI 등 메트릭 산출)
│   ├── batch_evaluate.py        # 배치 평가 스크립트 (여러 실험 일괄 평가)
│   ├── enhance.py               # 음성 향상 및 wav 저장 (streaming 옵션 포함)
│   ├── measure_rtf.py           # Real-Time Factor 측정 스크립트
│   ├── solver.py                # 학습 루프 (MetricGAN discriminator 포함)
│   ├── data.py                  # VoiceBankDataset, 세그먼트 샘플링
│   ├── stft.py                  # mag-phase STFT/iSTFT 유틸리티
│   ├── compute_metrics.py       # PESQ, STOI 등 메트릭 계산
│   ├── data_dns.py              # DNS/paired wav evaluation dataset
│   ├── evaluate_dns.py          # DNS-style cross-dataset evaluation
│   ├── receptive_field.py       # 수용장(receptive field) 계산기
│   ├── utils.py                 # 모델 로드, 체크포인트, 로깅 유틸리티
│   ├── models/
│   │   ├── backbone.py          # Backbone 모델 정의 (CausalConv, DenseEncoder, TSBlock 등)
│   │   ├── discriminator.py     # MetricGAN Discriminator
│   │   ├── streaming/           # 스트리밍 추론 관련 모듈
│   │   │   ├── layers/          # StatefulConv, ReshapeFree 등 스트리밍 레이어
│   │   │   ├── converters/      # 일반 모델 → 스트리밍 모델 변환기 (conv, reshape_free)
│   │   │   ├── lacosenet.py      # LaCoSENet wrapper (STFT overlap-add + lookahead buffering)
│   │   │   ├── utils.py         # 스트리밍 유틸리티 (manual iSTFT, latency 계산 등)
│   │   │   └── cpu_optimizations.py  # CPU 추론 최적화
│   │   └── onnx_export/         # ONNX 내보내기용 모듈
│   │       ├── layers/          # Stateful functional 레이어, ConvTranspose wrapper
│   │       ├── exportable_core.py
│   │       ├── stateful_core.py
│   │       ├── stateful_core_rf.py
│   │       ├── streaming_wrapper.py
│   │       ├── state_registry.py
│   │       └── verify_utils.py
├── ablation/                    # Ablation study (복잡도 측정 등)
├── baselines/                   # 베이스라인 모델 비교 (MP-SENet, MUSE, GTCRN 등)
│   ├── repos/                   # 베이스라인 모델 레포지토리
│   └── samples/                 # 베이스라인별 향상 오디오 샘플
├── demo/                        # 데모용 오디오/스펙트로그램
├── scripts/                     # 유틸리티 스크립트
├── results/
│   ├── experiments/             # Hydra 실험 출력 (체크포인트, 로그, TensorBoard)
│   ├── evaluation/              # 평가 결과
│   └── rtf/                     # RTF 측정 결과
├── paper_works/                 # 논문 작성 관련 문서
└── requirements.txt             # 의존성 목록
```

## 주요 커맨드

### 실행 환경
```bash
conda activate fullcomplex
```

- 이 레포의 기본 Python 실행 환경은 `fullcomplex`이다.
- Rebuttal DNSMOS 평가에는 `speechmos==0.0.1.1`이 필요하며, 해당 환경에 설치되어 있어야 한다.

### 학습
```bash
python -m src.train                                    # 기본 설정으로 학습
python -m src.train model.norm_type=batch               # BatchNorm으로 학습
python -m src.train model.causal_ts_block=true model.encoder_padding_ratio=[1.0,0.0] model.decoder_padding_ratio=[1.0,0.0]
python -m src.train hydra.run.dir=./results/experiments/my_exp  # 출력 디렉토리 지정
```

### 평가
```bash
python -m src.evaluate --model_config <exp_dir>/.hydra/config.yaml --chkpt_dir <exp_dir> --chkpt_file model_<step>.th
python -m src.batch_evaluate fullseq --exp_pattern "*s2039" --device cuda
python -m src.batch_evaluate chunksweep --experiments M1_12.5ms_s2039 --chunk_sizes 1 64 --device cuda
```

### 향상 (Enhancement)
```bash
# 기본 (non-streaming)
python -m src.enhance --chkpt_dir <exp_dir> --chkpt_file model_<step>.th

# Streaming (stateful conv)
python -m src.enhance --chkpt_dir <exp_dir> --chkpt_file model_<step>.th --use_stateful_conv
```

### RTF / DNS 재현
```bash
python -m src.measure_rtf --chkpt_dir results/experiments/M1_12.5ms/s2039 --chkpt_file auto --use_onnx --chunk_size 8 --output_json results/rtf/M1_cs8.json
bash scripts/run_rtf_chunk_sweep.sh
bash scripts/run_dns_eval_all.sh
```

## 문서 구성 (paper_works/)

### 논문 원고 (주 작업 문서)

| 파일 | 내용 |
|------|------|
| `interspeech2026/paper.tex` | **Interspeech 2026 논문 원고 (LaTeX)** — 모든 논문 편집은 이 파일 기준 |
| `interspeech2026/mybib.bib` | BibTeX 참고문헌 |
| `interspeech2026/Interspeech.cls` | Interspeech 2026 LaTeX 클래스 파일 |

### 참고 자료 (markdown)

| 파일 | 내용 |
|------|------|
| `method/method_overview.md` | LaCoSENet 아키텍처 및 novelty 정리 |
| `method/streaming_mechanics.md` | Streaming wrapper, state/lookahead buffer 구현 메모 |
| `method/latency_definition.md` | Algorithmic latency 정의 및 계산식 |
| `experiments/main_results.md` | 학습 실험 결과 기록 |
| `experiments/rtf_chunk_sweep.md` | ONNX steady-state RTF chunk sweep |
| `literature/benchmark_models.md` | 벤치마크 모델 latency/성능 ledger |
| `literature/latency_conventions.md` | 외부 모델 latency convention 검증 |
| `literature/related_work_notes.md` | 관련 연구 메모 |
| `figures/` | 논문 Figure 생성 스크립트 (plot_latency_vs_pesq.py, figure_style_guide.py) |

## 설정 (conf/config.yaml)

- Hydra 기반 설정 관리, `hydra.run.dir`로 실험별 디렉토리 자동 생성
- 주요 모델 파라미터: `causal_ts_block`, `encoder_padding_ratio`, `decoder_padding_ratio`, `norm_type`, `sca_kernel_size`
- 실험 결과는 `results/experiments/<날짜시간>/`에 저장 (체크포인트, `.hydra/config.yaml`, TensorBoard 로그)

## 코딩 가이드라인 (Karpathy-derived)

### 1. Think Before Coding
- 가정을 명시적으로 밝힌다. 불확실하면 먼저 질문한다.
- 여러 해석이 가능하면 선택지를 제시한다 — 임의로 하나를 고르지 않는다.
- 더 단순한 접근이 존재하면 말한다. 반박이 필요하면 반박한다.
- 혼란스러우면 멈추고, 무엇이 불명확한지 짚고 질문한다.

### 2. Simplicity First
- 요청된 것만 구현한다. 투기적 기능, 단일 용도 추상화, 요청되지 않은 "유연성"이나 "설정 가능성"을 추가하지 않는다.
- 200줄로 작성한 코드가 50줄로 가능하면 다시 쓴다.

### 3. Surgical Changes
- 요청과 직접 관련된 코드만 수정한다. 인접 코드, 주석, 포맷팅을 "개선"하지 않는다.
- 깨지지 않은 것을 리팩토링하지 않는다. 기존 스타일을 따른다.
- 내 변경으로 인해 미사용된 import/변수/함수는 제거한다. 기존 dead code는 요청 없이 건드리지 않는다.

### 4. Goal-Driven Execution
- 성공 기준을 먼저 정의하고, 검증될 때까지 루프한다.
- 다단계 작업은 검증 체크포인트가 포함된 간단한 계획을 먼저 제시한다.
