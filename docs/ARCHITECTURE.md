# Architecture: real offline inference

## Functional description

1. The user enters/pastes a short English message into `EditText`. The user can also load a fictional spam or normal sample.
2. Tapping `Analyze message` invokes `MainActivity.analyze`.
3. The Java `MessageVectorizer` normalizes URLs, long numbers and currency symbols, tokenizes English words and produces a binary vector of unigram and adjacent-bigram feature indicators.
4. `SpamClassifier` loads the trained model and vocabulary from APK `assets`, checks `[1,N]` float32 input and `[1,2]` float32 output and invokes the **LiteRT `Interpreter.run`** method with a `float[][]` tensor.
5. The output corresponds to `[normal, spam]` softmax scores. The app shows both values, a cautious verdict band, and a reminder that the classification is not security proof.
6. Pressing `Clear` removes input and any prior analysis. Editing a message hides stale predictions.

## Privacy by design

- Manifest contains **no** `READ_SMS`, `RECEIVE_SMS`, `SEND_SMS`, `INTERNET` or storage permissions.
- The app never logs message contents and does not save message history.
- All analysis happens locally with the packaged model, after installation.
- User must copy or type a message; the app cannot read a device's SMS inbox.

## Training method and caveat

This is an original training pipeline for classroom demonstration. `train_export.py` creates labeled synthetic spam and normal messages from explicit English templates, trains an actual PyTorch feedforward network, and serializes learned weights to a valid TFLite model. A held-out split of those generated messages and a small author-written spot check are NOT evidence of real-world accuracy because messages share common templates and style. A spam classifier can wrongly flag a legitimate message or miss a malicious one. Don't present it as operational cyber protection.

For a more scientifically rigorous assignment, you can replace this synthetic training set with a stratified, leakage-aware train/test split of a real labeled SMS corpus, record class-specific precision/recall and false-positive rates, then export and validate a new model against the same Java preprocessing.
