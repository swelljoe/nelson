package com.github.junrar.rarfile;

public class FileHeader {
    private final String fileName;

    public FileHeader(String fileName) {
        this.fileName = fileName;
    }

    public String getFileName() {
        return fileName;
    }

    public boolean isDirectory() {
        return false;
    }
}
