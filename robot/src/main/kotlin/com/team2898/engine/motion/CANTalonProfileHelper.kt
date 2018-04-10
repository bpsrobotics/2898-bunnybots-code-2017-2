package com.team2898.engine.motion

import com.ctre.CANTalon.TalonControlMode
import com.ctre.CANTalon.TrajectoryPoint
import com.team2898.engine.async.AsyncLooper

class CANTalonProfileHelper(val talon: CANTalonWrapper, var bufferPushHz: Double = 50.0) {
    val bufferPusher: AsyncLooper = AsyncLooper(bufferPushHz) {

    }

    var lastControlMode = talon.controlMode ?: TalonControlMode.PercentVbus
    var beforeProfileMode: TalonControlMode = lastControlMode

    var profileBuffer: MutableList<TrajectoryPoint> = mutableListOf<TrajectoryPoint>()
    var profileExecuting: Boolean = false
    var pushingBuffer = false

    fun startProfile() {
        lastControlMode = talon.controlMode ?: TalonControlMode.PercentVbus
        beforeProfileMode = lastControlMode
        talon.changeControlMode(TalonControlMode.MotionProfile)
    }

    fun setBuffer(push: Boolean = true, points: Array<TalonPoint>) {
        pushingBuffer = push
    }

    fun addToBuffer(push: Boolean = true, points: Array<TalonPoint>) {
        pushingBuffer = push
    }

    fun pushBuffer() {
        pushingBuffer = true
    }

    fun clearProfile() {
        if (profileExecuting) stopProfile()
        profileBuffer = mutableListOf<TrajectoryPoint>()
    }

    fun stopProfile() {
        if (!profileExecuting) return
        talon.changeControlMode(lastControlMode)
    }

}

fun TalonPointsToNative(points: Array<TalonPoint>): Array<TrajectoryPoint> {
    return Array<TrajectoryPoint>(points.size) { i ->
        TrajectoryPoint().apply {
            this.velocity = points[i].velocity
            this.timeDurMs = points[i].duration.toInt()
            this.position = points[i].position
        }
    }
}
