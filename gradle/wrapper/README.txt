IMPORTANT: This project contains an openly documented lightweight Gradle bootstrap,
not an official Gradle Wrapper JAR. Its complete Java source is the adjacent
GradleWrapperMain.java. It downloads the OFFICIAL Gradle 8.9 distribution from
https://services.gradle.org/distributions/gradle-8.9-bin.zip on first use,
then runs the downloaded Gradle executable.

WHY: This development container had no Gradle distribution or access to the
standard wrapper JAR. This transparent bootstrap keeps Windows CLI use possible.

For a standard Gradle wrapper, after Gradle is installed run:
    gradle wrapper --gradle-version 8.9
This overwrites the custom bootstrap JAR with the official wrapper JAR.

Android Studio Gradle Sync uses the distributionUrl in gradle-wrapper.properties.
Java 17+ required. Do NOT use third-party bootstrap JARs without examining source.
