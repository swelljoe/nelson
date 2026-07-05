package com.github.junrar;

import com.github.junrar.exception.RarException;
import com.github.junrar.rarfile.FileHeader;
import java.io.IOException;
import java.io.OutputStream;

public class Archive {
    public void extractFile(FileHeader header, OutputStream output) throws RarException, IOException {
        output.write(1);
    }
}
