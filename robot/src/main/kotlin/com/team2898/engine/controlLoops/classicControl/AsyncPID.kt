package com.team2898.engine.controlLoops

import com.team2898.engine.async.AsyncLooper
import com.team2898.engine.async.IAsyncLoop
import com.team2898.engine.extensions.drivetrain.blockJoin
import edu.wpi.first.wpilibj.Timer

/* import edu.wpi.first.wpilibj.util.BoundaryException */

/**
 * Asynchronous PID control loop wrapper
 * Inherits a standard synchronous PID loop as well as Kotlin's concurrent Job system to provide "true" concurrency (none of that notifier trash)
 * @author Solomon
 */

class AsyncPID(
        Kp: Double = 0.0,
        Ki: Double = 0.0,
        Kd: Double = 0.0,
        Kf: Double = 0.0,
        integratorMax: Double = Double.POSITIVE_INFINITY,
        currentTime: Double = Timer.getFPGATimestamp(),
        execHz: Double
) : StandardPID(Kp, Ki, Kd, Kf, integratorMax, currentTime), IAsyncLoop {

    var sensorValue: Double = 0.0 // Current sensor value
    var sensorOutput: Double = 0.0

    var getSensorInput: () -> Double = {
        throw NotImplementedError("Warning: getSensorInput() -> Double not implemented")
        0.0
    }
        @Synchronized get
        @Synchronized set

    var useControllerOutput: (Double) -> Unit = { throw NotImplementedError("Warning: useControllerOutput (Double) -> Unit not implemented") }
        @Synchronized set
        @Synchronized get

    var m_asyncLooper: AsyncLooper = AsyncLooper(execHz) {
        sensorValue = getSensorInput()
        sensorOutput = update(sensorOutput, currentTime = Timer.getFPGATimestamp())
    }

    override fun getLoop(): AsyncLooper = m_asyncLooper

    @Synchronized
    override fun start() = m_asyncLooper.start().blockJoin()

    @Synchronized
    override fun stop() = m_asyncLooper.stop().blockJoin()

    @Synchronized
    override fun setTargetHz(hz: Double) = m_asyncLooper.setTargetHz(hz)

}
