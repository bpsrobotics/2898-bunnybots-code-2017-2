package com.team2898.engine.motion

import com.ctre.CANTalon

/** Stores left + right motor speed + brake value
 * @param left left power/speed
 * @param right right power/speed
 * @param brake brake mode or not
 */
data class DriveSignal(val left: Double = 0.0, val right: Double = 0.0, val brake: Boolean = false) {
    companion object {
        val NEUTRAL: DriveSignal
            get() = DriveSignal(0.0, 0.0, false)
        val BRAKE: DriveSignal
            get() = DriveSignal(0.0, 0.0, true)
    }
}

/** Generated motion profile for built in profiling, Talon profiling compatible
 * @param position position in rotations
 * @param velocity velocity in rpm
 * @param duration duration in ms
 */
data class TalonPoint(val position: Double, val velocity: Double, val duration: Double)

/** Data transient object, allows for easy passing of framerate speeds into a CANTalonWrapper
 * @param generalFrameHz general frame hz
 * @param feedbackFrameHz closed loop feedback info frame hz
 * @param quadFrameHz quadrature encoder frame hz
 * @param analogTempVbatFrameHz temperature and vbat feedback frame hz
 * @param pulseWidthFrameHz pulse width position frame hz
 */
data class FrameSpeeds(
        val generalFrameHz: Double = 10.0,
        val feedbackFrameHz: Double = 10.0,
        val quadFrameHz: Double = 10.0,
        val analogTempVbatFrameHz: Double = 10.0,
        val pulseWidthFrameHz: Double = 10.0,
        val speedsMap: MutableMap<CANTalon.StatusFrameRate, Double> = mutableMapOf(
                Pair(CANTalon.StatusFrameRate.General, generalFrameHz),
                Pair(CANTalon.StatusFrameRate.Feedback, feedbackFrameHz),
                Pair(CANTalon.StatusFrameRate.QuadEncoder, quadFrameHz),
                Pair(CANTalon.StatusFrameRate.PulseWidth, pulseWidthFrameHz),
                Pair(CANTalon.StatusFrameRate.AnalogTempVbat, analogTempVbatFrameHz)
                )
)

