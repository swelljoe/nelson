package com.github.junrar;

import com.github.junrar.rarfile.FileHeader;
import java.nio.file.Files;
import java.nio.file.Path;

public final class CompatibilityControls {
    private CompatibilityControls() { }

    public static void main(String[] args) throws Exception {
        Path root = Files.createTempDirectory("junrar-controls-");
        Path destination = Files.createDirectory(root.resolve("destination"));
        Path forwardEscape = root.resolve("outside").resolve("escape.txt");

        try {
            LocalFolderExtractor extractor = new LocalFolderExtractor(destination.toFile());

            Path ordinary = extractor.extract(new Archive(), new FileHeader("ordinary.txt")).toPath();
            require(ordinary.equals(destination.resolve("ordinary.txt")), "ordinary path changed");
            require(Files.isRegularFile(ordinary), "ordinary file was not extracted");

            Path nested = extractor.extract(new Archive(), new FileHeader("nested\\child.txt")).toPath();
            require(nested.equals(destination.resolve("nested/child.txt")), "nested path changed");
            require(Files.isRegularFile(nested), "nested file was not extracted");

            try {
                extractor.extract(new Archive(), new FileHeader("../outside/escape.txt"));
                throw new AssertionError("forward-slash traversal was accepted");
            } catch (IllegalStateException expected) {
                require(!Files.exists(forwardEscape), "forward-slash traversal wrote a file");
            }
        } finally {
            deleteTree(root);
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void deleteTree(Path path) throws Exception {
        if (!Files.exists(path)) {
            return;
        }
        try (var paths = Files.walk(path)) {
            paths.sorted((left, right) -> right.compareTo(left)).forEach(item -> {
                try {
                    Files.deleteIfExists(item);
                } catch (Exception error) {
                    throw new RuntimeException(error);
                }
            });
        }
    }
}
