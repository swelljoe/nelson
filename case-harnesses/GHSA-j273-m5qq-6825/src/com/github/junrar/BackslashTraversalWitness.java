package com.github.junrar;

import com.github.junrar.rarfile.FileHeader;
import java.nio.file.Files;
import java.nio.file.Path;

public final class BackslashTraversalWitness {
    private BackslashTraversalWitness() { }

    public static void main(String[] args) throws Exception {
        Path root = Files.createTempDirectory("junrar-witness-");
        Path destination = Files.createDirectory(root.resolve("destination"));
        Path escaped = root.resolve("escaped.txt");

        try {
            LocalFolderExtractor extractor = new LocalFolderExtractor(destination.toFile());
            try {
                extractor.extract(new Archive(), new FileHeader("..\\escaped.txt"));
            } catch (IllegalStateException expected) {
                if (Files.exists(escaped)) {
                    System.err.println("traversal was rejected only after writing outside the destination");
                    System.exit(2);
                }
                System.exit(0);
            }

            if (Files.isRegularFile(escaped)) {
                System.err.println("archive entry wrote outside the extraction destination");
                System.exit(1);
            }
            System.err.println("entry was accepted but the expected behavior was not observed");
            System.exit(2);
        } finally {
            Files.deleteIfExists(escaped);
            Files.deleteIfExists(destination);
            Files.deleteIfExists(root);
        }
    }
}
