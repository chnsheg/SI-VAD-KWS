# Bit-Accurate MFCC Frontend Accuracy-First Design

This document defines an accuracy-first design plan for a bit-accurate MFCC
frontend. The chosen method combines:

- Scheme 1: feature alignment first
- Scheme 3: progressive hardware constraint tightening

The goal is not to immediately minimize every intermediate bit width. The first
goal is to recover the accuracy upper bound under the four hard constraints that
matter most for the target deployment:

- signed INT8 PCM frontend input
- rectangular Mel filtering
- PWL log approximation
- signed INT8 MFCC output

After that upper bound is verified, intermediate widths and fixed-point
operators are tightened step by step.

## Target

Use full-precision MFCC models as the starting point, then train and evaluate a
bit-accurate MFCC frontend plus INT8 backbone path.

Primary accuracy target:

- compare against `snr_scene_arch_sweep_grid_results.csv`
- average accuracy drop should be no more than 1 percentage point on common
  evaluation cells

First-stage scope:

- architecture: `L5_C64`
- datasets:
  - `mobvoi_hi_xiaowen_binary_hardneg`
  - `mobvoi_nihao_wenwen_binary_hardneg`
- evaluation: same scene and SNR grid as the existing TAU noise-scene sweep
- Mel shape: rectangular
- log form: PWL log
- frontend input: signed INT8 PCM
- final frontend output: signed INT8 MFCC

## Design Principle

The current direct bit-accurate frontend result drops much more than the normal
INT8 MFCC frontend. This means the main risk is not rectangular Mel itself, but
the accumulated mismatch from scale, shift, rounding, saturation, log/DCT
approximation, and training/inference frontend differences.

Therefore the design separates the problem into three layers:

1. Feature upper bound:
   keep intermediate computation wide or floating-point-like, only enforce
   signed INT8 PCM input, rectangular Mel, PWL log, and final INT8 MFCC output.
2. Trainable fake-quant frontend:
   add differentiable quantization operators with straight-through estimators,
   and train from the full-precision checkpoint.
3. Integer inference frontend:
   convert the accepted fake-quant contract into a deterministic integer
   frontend and verify parity.

Only after a layer passes accuracy and feature-alignment checks should the next
layer be tightened.

## Naming and Legacy Boundary

This design uses the correct naming consistently:

```text
bit-accurate   // prose
bit_accurate   // Python files, directories, CSV/JSON/Markdown artifacts
BitAccurate    // Python classes
```

The earlier rough experiment accidentally used `bit_accuracy` / `BitAccuracy`
names. Those scripts and frontend files are historical references only. They
should not be used as the implementation base for this experiment.

The old names may appear only when reading legacy CSVs for comparison. The new
training, QAT, inference, export, and analysis flows should use newly written
`bit_accurate` scripts and classes, with no imports or inheritance from the old
`bit_accuracy` scripts.

## Frontend Variants

### 1. Upper-Bound Frontend

Purpose: answer whether signed INT8 PCM input, rectangular Mel, PWL log, and
INT8 MFCC output can preserve accuracy when intermediate precision is not the
bottleneck.

Behavior:

- waveform input follows the same input normalization contract as the baseline
- frontend input is quantized to signed INT8 PCM; this hardware boundary is
  enabled from Phase 1 onward
- Mel filter is rectangular
- FFT, power, Mel accumulation, and DCT are high precision
- log must use the PWL form; the upper-bound phase may keep PWL input/output
  widths wide, but should not fall back to exact log
- final MFCC output is quantized to signed INT8
- quantization should use a straight-through estimator during training

Recommended name:

- `BitAccurateMFCCHighPrecisionFrontend`

This frontend is the first gate. If it cannot reach the target, reducing
intermediate widths is not meaningful yet.

### 2. Fake-Quant Bit-Accurate Frontend

Purpose: train the model while exposing it to the same quantization effects that
the final bit-accurate frontend will have.

Behavior:

- each stage can optionally enable quantization, clipping, rounding, and scale
  constraints
- quantization uses STE during backpropagation
- forward statistics are recorded for saturation, zero ratio, min/max, and
  distribution drift
- output shape and value contract match the final integer frontend

Recommended name:

- `BitAccurateMFCCFakeQuantFrontend`

This is the QAT frontend. The backbone sees the frontend distortion during
training instead of only after training.

### 3. Integer Bit-Accurate Frontend

Purpose: deterministic inference reference for the hardware-facing contract.

Behavior:

- no gradient requirement
- uses integer arithmetic, explicit shifts, explicit rounding, and saturation
- consumes the validated frontend spec exported from the fake-quant experiments
- generates the final signed INT8 MFCC tensor consumed by the backbone

Recommended new target:

- `dscnn_kws/frontend/bit_accurate_mfcc_frontend.py`

Recommended class name:

- `BitAccurateMFCCFrontend`

This integer reference path should be reimplemented for the new experiment. The
old `dscnn_kws/frontend/bit_accuracy_mfcc_frontend.py` file is only a historical
rough-experiment reference, and should not be a parent class, import dependency,
or config source for the new frontend. The final implementation should be driven
by an exported spec rather than hand-tuned constants scattered in code.

## Stage Contract

Every stage must have an explicit value contract. The final validated design
should produce a table like this, but with evidence attached to every choice:

```text
PCM_W              = 8 signed
MFCC_SAMPLE_W      = 12 signed
HANN_COEFF_W       = 16 unsigned
FFT_IN_W           = 18 signed
FFT_DATA_W         = 20 signed
TWIDDLE_W          = 16 signed
POWER_W            = 41 unsigned
MEL_ACC_W          = 46 unsigned
PWL_IN_W           = 32 unsigned
LOG_W              = 24 signed
DCT_COEFF_W        = 8 signed
DCT_ACC_W          = 40 signed
MFCC_OUT_W         = 8 signed
```

The final spec should include more than bit widths:

```text
PCM_W
MFCC_SAMPLE_W
PREEMPH_COEFF_W
HANN_COEFF_W
FFT_IN_W
FFT_DATA_W
TWIDDLE_W
FFT_STAGE_SHIFT
POWER_W
MEL_COEFF_W
MEL_ACC_W
MEL_DRAIN_SHIFT
PWL_IN_W
LOG_APPROX_MODE
LOG_PWL_NUM_SEGMENTS
LOG_PWL_BREAKPOINTS
LOG_PWL_SLOPES
LOG_PWL_INTERCEPTS
LOG_W
LOG_FRAC_BITS
DCT_COEFF_W
DCT_ACC_W
DCT_SHIFT
MFCC_OUT_W
MFCC_OUT_SCALE
ROUNDING_MODE
SATURATION_MODE
```

For each field, the report should record:

- chosen width or Q format
- signedness
- scale or fractional bits
- shift amount
- rounding rule
- saturation range
- observed min/max
- saturation ratio
- zero ratio
- feature error versus reference
- accuracy delta versus full precision

## Feature Alignment

Feature alignment is the first part of the combined scheme. It should be run
before long QAT sweeps.

Reference frontend:

- existing full-precision `TorchMFCC` path used by the baseline model

Candidate frontend stages:

```text
waveform
pcm_or_scaled_sample
preemphasis_or_sample_scale
windowed_frame
fft_input
fft_complex
power
rectangular_mel
pwl_log_mel
dct_acc
mfcc_prequant
mfcc_int8
```

For each stage, compute:

- MAE
- RMSE
- max absolute error
- cosine similarity
- Pearson correlation
- mean/std/min/max
- saturation ratio
- zero ratio
- per-coefficient MFCC error
- per-time-frame MFCC error

The most important alignment target is not bit equality with float MFCC. The
target is to identify where the frontend becomes distributionally different
enough that the backbone cannot recover.

## Experiment Phases

### Phase 0: Baseline Bookkeeping

Inputs:

- full-precision grid result: `snr_scene_arch_sweep_grid_results.csv`
- INT8 MFCC QAT result:
  `dscnn_kws/quantization/qat_int8_mfcc_frontend_grid_results.csv`
- legacy rough result, read-only comparison only:
  `dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_grid_results.csv`

Outputs:

- common-cell comparison table
- per-dataset and per-SNR baseline summary
- selected full-precision checkpoint paths for `L5_C64`

This phase makes sure every later experiment compares the same dataset,
architecture, scene, SNR, and checkpoint family.

### Phase 1: Accuracy Upper Bound

Constraint profile:

```text
mel_filter_shape      = rectangular
log_approx_mode       = pwl
pcm_input             = signed INT8
mfcc_output           = signed INT8
intermediate_widths   = high precision
pwl_log_io            = high precision
dct_coeff             = high precision or at least >= 16 bits
power/mel accum       = no effective clipping
```

Training:

- initialize from the full-precision `L5_C64` checkpoint
- use the upper-bound frontend
- keep the frontend input fixed to signed INT8 PCM; intermediate operators may
  start wide
- keep the log operator fixed to PWL; only the PWL input/output widths may start
  wide
- train with frontend output INT8 fake quant enabled
- QAT the backbone as in the existing INT8 MFCC pipeline

Acceptance:

- average drop versus full-precision baseline should be no more than 1 pp
- if it fails, inspect MFCC output scale, C0 handling, log normalization, and
  per-coefficient distribution before trying narrower widths

### Phase 2: Frontend-Aware QAT

Constraint profile:

```text
mel_filter_shape    = rectangular
log_approx_mode     = pwl
pcm_input           = signed INT8
mfcc_output         = signed INT8
frontend_quant      = differentiable fake quant
backbone_quant      = INT8 QAT
```

Training should expose the backbone to the same MFCC dynamic range and clipping
behavior that inference will use. This phase is the actual accuracy-first QAT
baseline.

Default MFCC output scaling:

- start with per-coefficient MFCC output scale
- then test per-tensor scale as a hardware simplification

Reason:

- per-coefficient scale gives a more realistic accuracy upper bound
- per-tensor scale is cheaper, but may over-compress low-energy coefficients

Acceptance:

- average drop versus full precision no more than 1 pp
- integer/fake-quant frontend accuracy gap no more than 0.5 pp after Phase 4

### Phase 3: Progressive Constraint Tightening

Once Phase 1 and Phase 2 pass, reduce hardware cost in a controlled order.

Recommended order:

1. DCT coefficient width:
   `16 -> 12 -> 10 -> 8`
2. DCT accumulator and final shift:
   choose the smallest width with low saturation and stable accuracy
3. PWL log output width, fractional bits, and lookup/segment parameters:
   sweep `LOG_W`, `LOG_FRAC_BITS`, `PWL_IN_W`, and PWL segment parameters;
   do not fall back to exact log
4. Mel drain:
   sweep `MEL_DRAIN_SHIFT`
5. power and Mel accumulator widths:
   reduce only after observing real min/max and saturation ratios
6. FFT and twiddle widths:
   tighten last because errors here affect all later stages

Do not change many stages at once. Each sweep should have one primary variable,
with all other stage contracts fixed.

Acceptance for each narrowing step:

- mean accuracy drop from the previous accepted profile should be small
- no broad low-SNR collapse
- saturation ratio should remain explainable and stable
- feature drift should identify the exact stage affected

### Phase 4: Integer Inference Parity

After a fake-quant profile is accepted, export the profile as a spec and run the
integer frontend.

Checks:

- fake-quant frontend output versus integer frontend output
- integer frontend plus trained backbone versus fake-quant frontend plus trained
  backbone
- per-dataset, per-SNR, and per-scene accuracy gap

Acceptance:

- average accuracy gap no more than 0.5 pp
- any single scene/SNR gap larger than 5 pp must be reported and explained
- stage-level mismatch should be traceable to rounding, shift, saturation, or
  lookup approximation

### Phase 5: Validated Bit-Width Spec

The final output is a validated frontend contract, not just a handwritten table.

Recommended output files:

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments/
  L5_C64_accuracy_first_summary.md
  L5_C64_bit_accurate_mfcc_validated_spec.json
  L5_C64_bit_accurate_mfcc_validated_spec.md
  feature_alignment_summary.csv
  constraint_sweep_summary.csv
  integer_parity_summary.csv
```

The final Markdown spec should contain:

- bit-width table
- Q-format table
- shift table
- PWL log segment table, including breakpoints, slopes, intercepts, and
  input/output scales
- rounding and saturation rules
- stage min/max statistics
- saturation and zero ratios
- feature alignment metrics
- accuracy comparison against full precision and INT8 MFCC QAT
- final recommendation for hardware implementation

## Proposed Scripts

These are proposed deliverables for the implementation phase.

```text
dscnn_kws/frontend/bit_accurate_mfcc_high_precision.py
dscnn_kws/frontend/bit_accurate_mfcc_fakequant.py
dscnn_kws/frontend/bit_accurate_mfcc_frontend.py
dscnn_kws/quantization/analyze_bit_accurate_mfcc_stages.py
dscnn_kws/quantization/sweep_bit_accurate_mfcc_constraints.py
dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py
dscnn_kws/quantization/eval_bit_accurate_mfcc_integer_parity.py
```

### Stage Analysis Command

```bash
python dscnn_kws/quantization/analyze_bit_accurate_mfcc_stages.py \
  --arch L5_C64 \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --mel-filter-shape rectangular \
  --profile upper_bound_s8_mfcc \
  --output-dir dscnn_kws/quantization/bit_accurate_mfcc_experiments
```

### Accuracy-First QAT Command

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --arch L5_C64 \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --mel-filter-shape rectangular \
  --constraint-profile upper_bound_s8_mfcc \
  --init-from-fp \
  --output-dir dscnn_kws/quantization/bit_accurate_mfcc_experiments
```

### Constraint Sweep Command

```bash
python dscnn_kws/quantization/sweep_bit_accurate_mfcc_constraints.py \
  --arch L5_C64 \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --base-profile accepted_upper_bound_s8_mfcc \
  --sweep dct_coeff_w,log_w,pwl_in_w,mel_acc_w,fft_data_w \
  --output-dir dscnn_kws/quantization/bit_accurate_mfcc_experiments
```

### Integer Parity Command

```bash
python dscnn_kws/quantization/eval_bit_accurate_mfcc_integer_parity.py \
  --arch L5_C64 \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --spec dscnn_kws/quantization/bit_accurate_mfcc_experiments/L5_C64_bit_accurate_mfcc_validated_spec.json \
  --output-dir dscnn_kws/quantization/bit_accurate_mfcc_experiments
```

## Default Decisions

Use these defaults unless the experiment results clearly argue otherwise.

```text
First model                 = L5_C64
Datasets                    = hi_xiaowen + nihao_wenwen
Mel filter                  = rectangular
Log implementation          = PWL log
Frontend input              = signed INT8 PCM
Training start              = full-precision checkpoint
First-stage frontend        = S8 PCM input + high precision intermediate + S8 MFCC output
First-stage MFCC scale      = per-coefficient scale
Hardware simplification     = test per-tensor scale after upper bound passes
DCT coefficients            = include the same coefficients as the existing model
Primary metric              = average accuracy delta on common grid cells
Secondary metric            = low-SNR and per-scene robustness
```

## Failure Diagnosis

If Phase 1 fails, debug in this order:

1. Compare final MFCC S8 distribution against float MFCC distribution.
2. Check whether per-tensor output scale clips important coefficients.
3. Check C0 and log-energy handling.
4. Compare rectangular-Mel high-precision output against the previous
   rectangular-Mel sweep result.
5. Inspect whether PWL log breakpoints, slopes, intercepts, input clipping, or
   output scaling change the feature mean too much.
6. Train with frontend frozen first, then unfreeze or tune scale parameters only
   if needed.

If Phase 2 passes but Phase 4 fails, the issue is not model capacity. It is an
implementation mismatch between fake quant and integer inference. Prioritize
rounding, shift order, saturation point, and lookup/PWL approximation.

If progressive narrowing fails, revert the last stage only and keep the last
accepted profile as the hardware candidate.

## Final Expected Output

Yes, the combined Scheme 1 + Scheme 3 flow should eventually produce a table
like the current baseline width list. The difference is that the final table
will be experiment-backed:

```text
PCM_W              = 8 signed
MFCC_SAMPLE_W      = 12 signed
HANN_COEFF_W       = 16 unsigned
FFT_IN_W           = 18 signed
FFT_DATA_W         = 20 signed
TWIDDLE_W          = 16 signed
POWER_W            = 41 unsigned
MEL_ACC_W          = 46 unsigned
PWL_IN_W           = 32 unsigned
LOG_W              = 24 signed
DCT_COEFF_W        = 8 signed
DCT_ACC_W          = 40 signed
MFCC_OUT_W         = 8 signed
```

The accepted version may keep these values, or it may choose wider/narrower
values depending on the sweep. The key is that each value must be justified by:

- stage-level feature alignment
- saturation and dynamic-range statistics
- QAT accuracy
- integer inference parity
