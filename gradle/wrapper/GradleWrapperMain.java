package org.gradle.wrapper;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Self-contained minimal Gradle 8.9 bootstrap for the educational project.
 * NOT Gradle's official wrapper jar. This transparent source is included so
 * that the bootstrap behavior can be inspected and the wrapper replaced by
 * official `gradle wrapper --gradle-version 8.9` after installing Gradle.
 */
public final class GradleWrapperMain {
    private static final String URL =
        "https://services.gradle.org/distributions/gradle-8.9-bin.zip";
    public static void main(String[] args) throws Exception {
        Path home = Path.of(System.getProperty("user.home"), ".gradle", "wrapper", "dists", "spamshield-gradle-8.9");
        boolean windows = System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win");
        Path executable = home.resolve(windows ? "gradle-8.9/bin/gradle.bat" : "gradle-8.9/bin/gradle");
        if (!Files.isRegularFile(executable)) {
            System.out.println("Downloading official Gradle 8.9 distribution (first build only)...");
            Files.createDirectories(home);
            Path archive = Files.createTempFile(home,"gradle-8.9-", ".zip");
            try {
                HttpClient client = HttpClient.newBuilder()
                    .followRedirects(HttpClient.Redirect.ALWAYS)
                    .build();
                HttpRequest request = HttpRequest.newBuilder(URI.create(URL)).GET().build();
                HttpResponse<Path> response = client.send(request,
                    HttpResponse.BodyHandlers.ofFile(archive));
                if (response.statusCode()!=200) {
                    throw new IOException("Gradle download HTTP " + response.statusCode());
                }
                try (ZipInputStream zip = new ZipInputStream(Files.newInputStream(archive))) {
                    ZipEntry entry;
                    while ((entry=zip.getNextEntry())!=null) {
                        Path target=home.resolve(entry.getName()).normalize();
                        if (!target.startsWith(home)) throw new IOException("Unsafe ZIP entry");
                        if (entry.isDirectory()) Files.createDirectories(target);
                        else {
                            Files.createDirectories(target.getParent());
                            Files.copy(zip,target,StandardCopyOption.REPLACE_EXISTING);
                        }
                        zip.closeEntry();
                    }
                }
            } finally { Files.deleteIfExists(archive); }
        }
        if (!Files.isRegularFile(executable)) throw new IOException("Gradle distribution incomplete: " + executable);
        if (!windows) executable.toFile().setExecutable(true);
        List<String> command=new ArrayList<>();
        if (windows) {command.add("cmd.exe");command.add("/c");}
        command.add(executable.toString());
        command.addAll(Arrays.asList(args));
        Process process=new ProcessBuilder(command).inheritIO().start();
        System.exit(process.waitFor());
    }
}
