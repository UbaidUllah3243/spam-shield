# Spam Shield — Offline AI SMS Spam Detector

**Android Assignment No. 2 | Ubaid Ullah | 242201064 | Submitted to Sir Uzair**

**Project:** A complete Java + XML Android Studio mini-project for **offline text-message screening** using a trained **TensorFlow Lite (LiteRT) neural network**. The user manually enters or pastes English SMS text. The app shows a spam score and a cautionary assessment (Likely spam / Review carefully / Likely normal). It includes two example buttons, a Clear button and security advice. **It does not silently read, send or automatically filter SMS.**

## Status and testing honesty

- **Included:** Android Java/XML source code, real trained `.tflite` neural-network file, matching vocabulary, Gradle configuration, learning/training code, model documentation and tests.
- **Checked here:** XML parsing, Java syntax/type check against local Android API stubs, Java-versus-Python feature-vector parity, TFLite FlatBuffer structural and weight checks, desktop neural-network inference.
- - **Verified by student in Android Studio:** Gradle Sync, successful app launch on the emulator, spam and normal-message predictions, offline inference, empty-input handling, and genuine emulator screenshots.
- **Still pending verification:** Native-library ELF alignment, APK ZIP alignment for 16 KB memory pages, and testing on a compatible 16 KB page-size device.
- **Model limitation:** Training sentences are **original author-written synthetic templates**; they are not the UCI SMS Spam Collection or a representative modern scam dataset. Scores are not calibrated probabilities, and the model can make incorrect security classifications. **Do not rely on it to decide whether a link or request is safe.**

## How to run on Windows 11

1. **Extract** `SpamShield_Java_LiteRT_AndroidStudio_Project.zip` using **Extract All**. Do not open or run the ZIP itself.
2. In Android Studio select **File → Open** and choose the outer **`SpamShield_Android_Project` folder**, which contains `settings.gradle`. Do not select the nested `app` or `java` directory.
3. If Android Studio asks to trust the project, review the code and choose **Trust Project**.
4. Enable internet access for **the first Gradle sync**. In **SDK Manager**, install Android SDK Platform **35** and recent Android SDK Build Tools. Use a Gradle JDK compatible with Android Gradle Plugin 8.7.3 (JDK **17** is recommended).
5. Click **File → Sync Project with Gradle Files**. Or press **Ctrl+Shift+A**, search for that command, and run it.
6. Open **Tools → Device Manager**, create an Android emulator, or connect your real Android phone with USB debugging enabled.
7. Select the `app` run configuration and device, then click **▶ Run** (Shift+F10).
8. Tap **Spam example** → **Analyze message**, and then **Normal example** → **Analyze message**. Try typing your own messages. The app itself does not need internet for inference.
9. Test the blank-message error, Clear button, airplane mode and repeated predictions. Record real observations and screenshots in `docs/screenshots/`.

### If Android Studio says "Design editor is unavailable until next Gradle sync"

This is usually because Android Studio hasn't finished reading the Gradle project. Use **File → Sync Project with Gradle Files**, wait for sync, and inspect **Build Output** for the *first* actual error. Ensure you opened the complete project folder, Android SDK 35 is installed, and Gradle isn't in Offline mode on its first download.

### About the Gradle Wrapper

This project includes `gradle/wrapper/gradle-wrapper.properties` configured for official Gradle 8.9. Since the generation environment did not have the official wrapper JAR available, `gradle-wrapper.jar` is an **openly disclosed nonstandard bootstrap** (source: `gradle/wrapper/GradleWrapperMain.java`) carried over from the previous educational project. It downloads the **official Gradle distribution** when invoked from the terminal. **Review any wrapper binary before running it.** Android Studio's built-in Gradle handling can use the distribution URL. If you have Gradle installed, generate a standard wrapper with `gradle wrapper --gradle-version 8.9`. Details in `gradle/wrapper/README.txt`.

## Critical: Android 16 KB page alignment

The first DigitAI app triggered the native-library 16 KB alignment warning. To reduce that risk this app **does not use old** `org.tensorflow:tensorflow-lite:2.17.0` and instead uses:

- **LiteRT 1.4.2**: `com.google.ai.edge.litert:litert:1.4.2` and `litert-api:1.4.2`, retaining the `org.tensorflow.lite.Interpreter` API, with native libraries built for 16 KB pages.
- **Android Gradle Plugin 8.7.3** + **Gradle 8.9** for suitable APK ZIP alignment.
- `packaging { jniLibs { useLegacyPackaging = false } }` for uncompressed JNI packaging.

**Final verification after building an APK in Android Studio:**

1. Go to **Build → Analyze APK** and open `app/build/outputs/apk/debug/app-debug.apk`. Inspect the `lib/` folder and **Alignment** column. Each native `.so` needs 16 KB ELF LOAD-segment alignment.
2. Run the Android SDK `zipalign` tool, adjusting the build-tools path if needed:

   ```powershell
   & "$env:LOCALAPPDATA\Android\Sdk\build-tools\35.0.0\zipalign.exe" -c -P 16 -v 4 "app\build\outputs\apk\debug\app-debug.apk"
   ```

3. If possible, test an Android emulator configured with 16 KB memory pages. Verify it with `adb shell getconf PAGE_SIZE` (expected **16384**), then open the app and run predictions.
4. If Android Studio still warns, expand the warning and send the **name of the `.so` library** and full Build Output. A transitive native dependency might still need updating.

**The warning is not suppressed with a manifest switch. We do not claim complete 16 KB compatibility without an actual APK check.** Official reference: https://developer.android.com/guide/practices/page-sizes

## Files and structure

```text
SpamShield_Android_Project/
├── app/
│   ├── build.gradle                         # LiteRT 1.4.2, 16 KB build settings
│   └── src/
│       ├── main/
│       │   ├── AndroidManifest.xml          # no SMS/INTERNET permissions
│       │   ├── java/com/ubaid/spamshield/
│       │   │   ├── MainActivity.java       # UI state and error handling
│       │   │   ├── MessageVectorizer.java  # exact text preprocessing
│       │   │   └── SpamClassifier.java     # Interpreter.run() inference
│       │   ├── res/layout/activity_main.xml
│       │   ├── res/values/ and drawable/    # theme and visuals
│       │   └── assets/
│       │       ├── spamshield.tflite         # trained neural network
│       │       └── vocabulary.txt           # learned feature order
│       └── test/                            # Java input-preparation tests
├── ml/
│   ├── train_export.py                      # synthetic dataset and training
│   ├── weights.npz                          # actual trained weights
│   ├── flatbuffer_writer.py                 # native TFLite schema exporter
│   ├── tensorflow_conversion.py             # optional official TF converter
│   ├── validate_model.py                    # model structure parity checks
│   └── check_java_vectorizer.py             # Java/Python preprocessing parity
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TESTING_CHECKLIST.md
│   ├── ORIGINAL_REPOSITORY_STUDY.md
│   ├── GITHUB_UPLOAD.md
│   └── screenshots/                         # insert actual screenshots only
├── MODEL_CARD.json
├── BUILD_STATUS.md
├── settings.gradle
├── build.gradle
├── gradle.properties
└── README.md
```

## How the AI works

`message text → lowercasing & URL/number normalization → binary vocabulary vector → LiteRT Interpreter.run() → NORMAL/SPAM scores → app verdict`.

1. The Python training code generates author-written example messages and trains a **1,197-feature → 40 ReLU → 2 softmax** neural network in PyTorch. The feature count is learned from generated training data and recorded in `MODEL_CARD.json`.
2. The learned weight tensors are serialized into a version-3 TensorFlow Lite FlatBuffer using standard `FULLY_CONNECTED`, `RELU` and `SOFTMAX` operations. The `ml/tensorflow_conversion.py` script also shows a conventional Keras `TFLiteConverter.from_keras_model` export of the saved weights, if TensorFlow is installed on your machine.
3. `SpamClassifier.java` loads `spamshield.tflite` and `vocabulary.txt`, initializes the **LiteRT Interpreter**, validates tensor shapes and data types, and calls `interpreter.run(input, output)`.
4. Input shape is **`[1, 1197]` float32**; output shape is **`[1, 2]` float32** with scores ordered **Normal, Spam**.
5. `MessageVectorizer.java` normalizes URLs to `urltoken`, numbers of 5+ digits to `numbertoken` and currency symbols to `moneytoken`, then computes the same binary unigram/bigram features in the same vocabulary order used by Python.
6. UI labels are **Likely normal**, **Review carefully**, and **Likely spam** based on score bands, with risk disclaimers and basic cautionary advice. These bands are interface decisions, not calibrated thresholds proven by a clinical or security study.

### Reproduce model preparation (optional)

```powershell
python -m pip install numpy scikit-learn torch
python ml/train_export.py
python ml/validate_model.py
python ml/check_java_vectorizer.py
```

To demonstrate the formal *TensorFlow model → .tflite* procedure in the assignment:

```powershell
python -m pip install tensorflow numpy
python ml/tensorflow_conversion.py
```

It writes a **comparison** model under `ml/` and does not overwrite the deployed `.tflite` asset. The training script and vocabulary must be kept in sync if the network is retrained.

## Student assignment requirements and source attribution

**Instructor reference repository:** https://github.com/Uzair390-Del/NumberGuessingAppTflite.git

The instructor's reference project, NumberGuessingAppTflite, has been opened, explored, and executed successfully in Android Studio. Its project structure, MainActivity, TensorFlow Lite model (linear.tflite), and inference workflow have been studied. Spam Shield was developed separately using these concepts as a reference.`docs/ORIGINAL_REPOSITORY_STUDY.md`.

Once Spam Shield runs on your PC, upload this project to your own GitHub repository and capture actual app results. Then prepare the theory PDF, practical PDF, and code-explanation PDF from the verified project, referencing the original repo and **the educational synthetic-data limitation**. Details in `docs/GITHUB_UPLOAD.md`.

**Data attribution:** All packaged training messages are generated locally by our original, transparent training script; **no external SMS dataset is included**. For a production-grade follow-up, consider the UCI SMS Spam Collection (Almeida & Hidalgo, 2011, CC BY 4.0), retrain and reevaluate separately: https://archive.ics.uci.edu/dataset/228/sms+spam+collection. **This model was not trained on it.**

**Libraries:** Google LiteRT, PyTorch, NumPy, and scikit-learn. Use the licensing terms of these projects as applicable.
