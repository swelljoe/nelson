package org.slf4j;

public final class LoggerFactory {
    private static final Logger NO_OP = (message, argument, throwable) -> { };

    private LoggerFactory() { }

    public static Logger getLogger(Class<?> type) {
        return NO_OP;
    }
}
