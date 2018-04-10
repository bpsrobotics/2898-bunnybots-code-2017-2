package com.team2898.engine.logging

import edu.wpi.first.wpilibj.Timer
import java.io.PrintWriter
import java.io.StringWriter

/** Generates stack trace
 * @param throwable throwable to extract stack trace
 * @return formatted stack trace
 */
fun getStackTrace(throwable: Throwable): String {
    val sw = StringWriter()
    val pw = PrintWriter(sw)
    throwable.printStackTrace(pw)
    return sw.toString()
}

/** Generates stack trace
 * @return trace from current spot
 */
fun getStackTrace(): String = getStackTrace(Throwable())

data class InfoLog(val source: String, val level: LogLevel, val message: String) {
    override fun toString(): String {
        return "$level $source at ${Timer.getFPGATimestamp()}: $message"
    }
}

data class DataLog(val source: String, val name: String, val data: Any, val time: Double = Timer.getFPGATimestamp()) {
    override fun toString(): String {
        return "$source, $time, $name, $data"
    }
}

fun reflectLocation(): String {
    val trace = Exception().stackTrace[1]
    return "$trace:${trace.lineNumber}"
}

class SelfCheckFailException(val reason: String, val level: LogLevel): Exception(reason)
