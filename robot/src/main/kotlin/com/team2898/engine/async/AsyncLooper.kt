package com.team2898.engine.async

import com.team2898.engine.logging.LogLevel
import com.team2898.engine.logging.Logger
import com.team2898.engine.logging.TimeBombAsync
import com.team2898.engine.logging.reflectLocation
import edu.wpi.first.wpilibj.Timer
import kotlinx.coroutines.experimental.*


class AsyncLooper(var hz: Double, val logOnCantKeepUp: Boolean=true, val startStopTimeout: Double=1.0, val func: () -> Unit) {
    var m_job: Job = init()
    var m_targetHz = hz

    private fun init(): Job {
        return launch(context=CommonPool, start=CoroutineStart.LAZY) {
            while (isActive) {
                val startTime = Timer.getFPGATimestamp()
                func()
                val deltaTime = Timer.getFPGATimestamp() - startTime
                if (deltaTime < 1000/hz) delay((1000/m_targetHz - deltaTime).toLong())
                else if (logOnCantKeepUp) logCantKeepUp (deltaTime)
            }
        }
    }

    fun start(): Job {
        m_job = init()
        return TimeBombAsync(startStopTimeout) {
            m_job.start()
        }.start()
    }

    fun stop(): Job {
        return TimeBombAsync(startStopTimeout) {
            m_job.cancel()
            m_job.join()
        }.start()
    }

    private fun logCantKeepUp(lastTime: Double) {
        Logger.logInfo(reflectLocation(), LogLevel.WARNING, "AsyncLooper cannot keep up! Running at $m_targetHz hz and previous loop took ${lastTime*1000} ms")
    }

    fun setTargetHz(hz: Double) {
        m_targetHz = hz
    }
}
