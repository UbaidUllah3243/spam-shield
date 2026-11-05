# Instructor's original repository — clone/run/study worksheet

Repository: https://github.com/Uzair390-Del/NumberGuessingAppTflite.git

**This worksheet is for required *actual observations*, not evidence that we have already run the reference project.**

## Steps

1. Open Windows Terminal or Android Studio Terminal and run:

   ```powershell
   git clone https://github.com/Uzair390-Del/NumberGuessingAppTflite.git
   ```

2. Open the **root cloned folder** in Android Studio; allow Gradle Sync and install any missing SDK components.
3. Inspect README, Gradle dependency files, `.tflite` assets, Java or Kotlin source, XML layout and manifest.
4. Try to **run it on an emulator/phone**. If it cannot run, record the exact error and any fix you tried.
5. Find and record the actual **model path**, `Interpreter` initialization, input preprocessing, input/output tensor shapes and data types, inference call, and UI result display.
6. Add meaningful code comments to any studied example file, but do not misrepresent that as code reused by Spam Shield.

## Observations to fill after inspection

- Repository commit/version: _____________________
- README purpose: _____________________
- Actual app screens and output: _____________________
- `.tflite` model location: _____________________
- Inference API and dependency version: _____________________
- Model input tensor shape and data type: _____________________
- Model output tensor shape and data type: _____________________
- User input processing: _____________________
- How the prediction is displayed: _____________________
- Gradle Sync / Run outcome: _____________________
- Screenshot filenames: _____________________

## Comparison to Spam Shield

Both projects illustrate on-device ML integration via a `.tflite` model, model loading, preparing a numeric tensor, inference and UI feedback. Spam Shield differs in its **application domain** (text-based spam screening), feature engineering (binary words/bigrams), visual design, and educational synthetic training pipeline. Confirm the original implementation details from code rather than assuming they are identical.
