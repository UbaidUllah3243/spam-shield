# Actual Android Testing Checklist — fill in after running on Windows 11

**Device:** __________  **Android version:** __________  **Date:** __________

| # | Test | Expected behavior | Actual result / screenshot | Pass? |
|---|---|---|---|---|
| 1 | Gradle sync | Sync completes without errors | | |
| 2 | Debug APK build | APK compiles and installs | | |
| 3 | Launch app | Main screen loads, AI model ready | | |
| 4 | Tap Spam example → Analyze | Spam score and result panel appear | | |
| 5 | Tap Normal example → Analyze | Score and result panel appear | | |
| 6 | Empty message → Analyze | Error is displayed, no crash | | |
| 7 | Clear after prediction | Clears text, hides old result | | |
| 8 | Edit message after prediction | Hides stale result automatically | | |
| 9 | Turn off internet / airplane mode | Model still runs after install | | |
| 10 | Repeated predictions | No crash or stale UI | | |
| 11 | APK Analyzer native libraries | All `.so` ELF LOAD segments 16 KB-aligned | | |
| 12 | zipalign verification | 16 KB APK ZIP alignment verified | | |
| 13 | 16 KB page emulator if available | `PAGE_SIZE=16384`, app runs | | |

Screenshots should be captured **from the actual running app**; never use generated screenshots as proof of executed tests. Suggested filenames: `01_gradle_sync.png`, `02_home.png`, `03_spam_prediction.png`, `04_normal_prediction.png`, `05_blank_warning.png`, `06_16kb_apk_analyzer.png`.

## Be clear about scores

Do not claim the educational model has robust real-world spam-detection accuracy. In particular, synthetic template hold-out scores may be misleadingly high; report their scope and any misclassified tests honestly.
