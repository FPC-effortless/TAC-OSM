# BENCH-FM-002 — Real performance in the actual Google foundation model task domains

**Experiment date:** October 21, 2026. **All displayed numbers are genuine fresh evaluations.** Stage A official TimesFM3 and original trained TAC/GRU controls completed on [corrected workflow 39601162496](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39601162496). Stage B official TabFM1.0 reported separately once its raw artifact exists.

## A. Real multivariate long-horizon ETTh1 prediction

Official publisher ETTh1 SHA256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`. Train original first 8,760 hours, original known DEV to hour 11,640, then **48 predetermined chronological later test origins** from hour 11,640 to 17,323; 7 genuine input/target channels, 24 and 96 future hourly horizons. Report train-only standardized physical units; no synthetic scoring.

| Model | Forecast hours 24 MAE | 96 MAE | Macro MAE (lower better) | MSE |
|---|---:|---:|---:|---:|
| TimesFM3.0 official 512h context, zero shot | **0.355104** | **0.468594** | **0.411849** | **0.421402** |
| TimesFM3.0 official 96h context, zero shot | 0.387456 | 0.507394 | 0.447425 | 0.518914 |
| Persistence | 0.40157 | 0.57006 | 0.485815 | 0.618855 |
| Seasonal lag24 | 0.54423 | 0.59410 | 0.569164 | 0.791701 |
| New ME035-inspired two-timescale TAC, seed35002 | 0.5253 | 0.6560 | 0.590689 | 0.780677 |
| New ME035-inspired two-timescale TAC, seed35001 | 0.5566 | 0.6493 | 0.602978 | 0.786492 |
| GRU seed35001 | 0.5289 | 0.6726 | 0.600733 | 0.732850 |
| GRU seed35002 | 0.5651 | 0.6814 | 0.623226 | 0.790342 |

The 96h TimesFM is compared with exactly the same 96h history, 7 features and 24/96 targets as TAC. The 512h TimesFM has five-plus times longer context and is a **separate native-use capability demonstration**. TimesFM score succeeded on authentic publisher data with officially loaded pretrained `google/timesfm-3.0-pytorch` weights. Inference seconds excluding model startup/download 9.07s/23.37s for all 48 cases at 96h/512h contexts.

The trained TAC is a **NEW 18,280-parameter fast32/slow32 ME035-inspired model** with a declared 24/96h head, not the original exported ME035 32,154-param 1/6/24h composite. It was newly trained twice at 250 AdamW updates, batch64 using exactly real ETTh1 train, with 19,086-parameter GRU two seeds trained for precisely the same updates/data. CPU training times TAC 6.84-7.17s; GRU 6.22-6.59s. All weights retained and hashes audited by manifest. The best TAC seed reaches 0.590689 MAE and **LOSES to persistence and official TimesFM**; Google TimesFM3 wins by 24.3% at 96h context, 30.3% at 512h context. For Brier style this is not a binary task. Significant model pretraining differences preclude a fully fair *architecture* comparison.

**Evidence artifacts in run 39601162496:** `bench-fm-002-native-timeseries-original` (four checkpoint weights and real scores) and `bench-fm-002-google-timesfm3-official-or-failure` (successful official checkpoint inference and metrics). Source code is in this PR.

## B. Real mixed tabular classification

Official UCI Bank Marketing archive outer ZIP SHA256 `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4`, 45,211 genuine label-containing records, 15 mixed numerical/categorical pre-contact predictors. `duration` (known only after contact) excluded from every model. Original first70% chronological rows form source of **256 training examples**; final20% source of **96 later test examples**; no random train/test overlap. Labels shift **16/256 positive training (6.25%) to 37/96 positive test (38.54%)**, a severe honest temporal regime change (not easy i.i.d. few-shot distribution). This source is not a causal marketing evaluation.

| Model | Brier (lower) | ROC-AUC (higher) | PR-AUC |
|---|---:|---:|---:|
| Histogram gradient boost | **0.300111** | 0.5905 | 0.5066 |
| Official **TabFM1.0 zero-shot** (256 labels in context) | **0.320011** | **0.620247** | **0.477496** |
| Logistic regression | 0.3068 | **0.6280** | **0.5456** |
| New TAC-inspired tabular feature recurrence seed31001 | 0.332223 | 0.625744 | 0.494826 |
| Constant rate | 0.3411 | 0.5000 | 0.3854 |
| New TAC-inspired tabular feature recurrence seed31002 | 0.392837 | 0.519927 | 0.445206 |

Original baseline weights/features and results retained in successful real-data CI artifact `bench-fm-002-native-tabular-original`. This is explicitly **not** a trained native TAC software-action model: the gated fast/slow recurrence is adapted across a fixed 15-feature field sequence, with 12,769 params, 400 updates per seed, equal supervised 256 labeled rows. TabFM gets 256 in-context labeled examples, with original TabFM 1.0 pretrained weights held fixed. It is NOT TabFM 1.1; the official PyPI exports v1.0 only. All models receive same authentic 15 fields and labeled split, but their pretrained datasets and training compute are unmatched.

**Official TabFM v1.0 observed outcome: COMPLETED SUCCESSFULLY, verified original GitHub Actions artifact [12345664566](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39601162496/artifacts/12345664566).** Original official downloaded `google/tabfm-1.0.0-pytorch` pretrained weights loaded, kept frozen; 256 authentic training examples passed as label context, 96 authentic chronological holdout rows predicted. TabFM Brier **0.32001140986571125**, logloss **0.9907556567746676**, ROC-AUC **0.6202473660100779**, PR-AUC **0.4774959013443591**. Mean predicted positive probability **0.0921518** versus real test observed positive **0.3854167**, illustrating heavy prevalence/calibration shift. Google weights load (including official HF download) 45.63s and CPU prediction 135.59s for 96 rows, excluding framework installation. TAC Brier 0.332223/0.392837: TabFM beats both TAC seeds in Brier but **not** tuned conventional histogram boost (0.3001) or logistic (0.3068). In AUC TabFM 0.6202 beats the weaker TAC but slightly loses to TAC seed31001 (0.6257) and logistic (0.6280). This is a genuine task-scoped outcome, not a universal model ranking. The separate older BENCH-FM-001 TabFM v1.1 import and v1.0 loader failures remain archived in PR299; this new successful v1.0 model is NOT v1.1.

## Rules and limitations

No synthetic success claims. Exactly original real publisher sources; integrity checks freeze original bytes. Fresh timestamps used for ETTh1 after previous ME035 dev, but possible overlap with Google's training datasets cannot be independently ruled out. Test windows 48, no statistical confidence intervals and an observed single-site task; cannot infer generalized foundation-model performance. Google pretrained models have dramatically different training data, trainable sizes and compute; **compulsory fair-training rule forbids a claim that TAC/TIMESFM architecture is generally superior based on these experiments**. No commercial license claim for Google's noncommercial weights. No executable action planning, real decision outcome or System1/System2/ROI qualification. Keep research-only draft; no merge.
