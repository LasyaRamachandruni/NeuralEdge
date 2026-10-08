
Sleep-EDF Expanded (sleep-cassette), subjects 0-19 (20 subjects, 42307 scored 30-s test epochs), Fpz-Cz, leave-one-subject-out CV.

| Model | Macro-F1 (mean ± std over folds) | Pooled acc. | Cohen's κ | W | N1 | N2 | N3 | REM | ONNX CPU p50 (batch 1) |
|---|---|---|---|---|---|---|---|---|---|
| `resnet1d_tiny` (this repo) | 0.724 ± 0.061 | 0.781 | 0.709 | 0.880 | 0.371 | 0.835 | 0.842 | 0.735 | 1.19 ms |
| DeepSleepNet (Supratak et al., 2017) [ref] | 0.769 (reported) | 0.820 | - | - | - | - | - | - | - |

Per-class F1 is computed on all test folds pooled. Latency: onnxruntime 1.30.0 CPUExecutionProvider on Intel(R) Xeon(R) CPU @ 2.00GHz, 1 thread(s) - a cloud CPU, not an edge device.

[ref] A. Supratak, H. Dong, C. Wu, Y. Guo, "DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG", IEEE Trans. Neural Syst. Rehabil. Eng. 25(11), 2017. arXiv:1703.04046. Not a like-for-like comparison: DeepSleepNet uses the 2013 Sleep-EDF release and models sequences of epochs; this repo classifies each 30-s epoch independently with a much smaller CNN.
