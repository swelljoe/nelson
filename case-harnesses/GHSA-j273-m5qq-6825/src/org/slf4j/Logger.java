package org.slf4j;

public interface Logger {
    void error(String message, Object argument, Throwable throwable);
}
