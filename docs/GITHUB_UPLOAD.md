# Upload tested Spam Shield code to your GitHub account

After **Android Studio Sync, build, and device tests are complete**:

1. On github.com create a new empty public repository named `SpamShield-Java-TFLite` (or use another name).
2. In Android Studio Terminal, open the project root and run:

   ```powershell
   git init
   git add .
   git commit -m "Develop Spam Shield offline AI Android app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/SpamShield-Java-TFLite.git
   git push -u origin main
   ```

3. Replace `YOUR_USERNAME`. Use your usual GitHub authentication (no passwords/tokens in repository files).
4. Check that the repository includes **app source, Gradle configuration, model asset, vocabulary, training code, README, and documentation**.
5. Ensure it excludes `local.properties`, `.gradle`, local build output, APKs, and any private user SMS content.
6. Copy the repository URL into both assignment reports. Keep the instructor's original reference repository URL as a separate citation.
7. Add only authentic screenshots from your Android Studio run and a completed test checklist.
