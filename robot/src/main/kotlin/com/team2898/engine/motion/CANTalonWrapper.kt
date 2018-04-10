package com.team2898.engine.motion

import com.ctre.CANTalon
import com.team2898.engine.async.AsyncLooper
import com.team2898.engine.logging.DataLog
import com.team2898.engine.logging.Logger
import com.team2898.engine.logging.reflectLocation
import com.team2898.engine.signalProcessing.MovingAverageFilter
import com.team2898.engine.types.MinMax
import com.team2898.engine.util.clamp
import com.team2898.robot.Robot
import edu.wpi.first.wpilibj.Timer
import jdk.net.SocketFlow

/** Custom CANTalon wrapper enabling databinding and easy motion profiles. Also can automatically optimize status frames.
 * @param deviceID CAN ID of the device
 * @param stateMachineHz Hz to run the internal state machine at. 10 is good for low priority subsystems, 100 for high
 * @param bufferPushHz Formality, 50 is good
 * @param generalFrameHz Minimum and maximum
 * @param framerateFilterSize The number of previous unique setpoint delays to average
 */
class CANTalonWrapper(
        deviceID: Int,
        stateMachineHz: Double = 10.0,
        bufferPushHz: Double = 50.0,
        frameSpeeds: FrameSpeeds = FrameSpeeds(),
        useAdaptiveFramerate: Boolean = true
) : CANTalon(deviceID) {

    companion object {
        fun createSlave(deviceID: Int, other: CANTalon): CANTalonWrapper {
            return CANTalonWrapper(deviceID) slaveTo other
        }
    }

    protected var frameSpeeds: MutableMap<StatusFrameRate, Double> = frameSpeeds.speedsMap

    init {
        setFrameHz()
    }

    var lastSetpointTimestamp = Double.NaN
    val useAdaptiveFramerate: Boolean by lazy { useAdaptiveFramerate } // Quick hack so we can use this from set()

    var lastSetpoint: Double = Double.NaN
    var lastControlMode: TalonControlMode = controlMode ?: TalonControlMode.PercentVbus

    val profileHelper = CANTalonProfileHelper(this, bufferPushHz)

    var lastPIDProfile: Int = -1

    var stateMachineLooper: AsyncLooper = AsyncLooper(stateMachineHz) {
        // If we're using an adaptive framerate for the general status frames, set it to the average of how often our
        // setpoints change
        //if (useAdaptiveFramerate) {
        //    var framerate = clamp(framerateFilter.getAverage(), 1 / generalFrameHz.min, 1 / generalFrameHz.max)
        //    setStatusFrameRateMs(StatusFrameRate.General, (framerate * 1000).toInt())
        //}

    }

    @Synchronized
    override fun set(value: Double) {
        if (lastSetpoint != value) {
            super.set(value)
            lastSetpoint = value

            // Update moving average filter if we're doing the adaptive framerate thingy
            //if (useAdaptiveFramerate) {
            //    if (lastSetpointTimestamp == Double.NaN)
            //        lastSetpointTimestamp = Timer.getFPGATimestamp()
            //    framerateFilter.addValue(Timer.getFPGATimestamp() - lastSetpointTimestamp)
            //    lastSetpointTimestamp = Timer.getFPGATimestamp()
            //}
        }
    }

    @Synchronized
    override fun changeControlMode(controlMode: TalonControlMode) {
        if (lastControlMode.value != controlMode.value) {
            super.changeControlMode(controlMode)
            lastControlMode = controlMode
        }
    }

    @Synchronized
    override fun setProfile(profile: Int) {
        if (lastPIDProfile != profile) {
            lastPIDProfile = profile
            super.setProfile(profile)
        }
    }

    fun setOpenLoop(value: Double) {
        setOpenLoop()
        set(value)
    }

    fun setVelocityControl(value: Double) {
        setVelocityControl()
        set(value)
    }

    fun setPositionControl(value: Double) {
        setPositionControl()
        set(value)
    }

    fun setMagicControl(value: Double) {
        setMagicControl()
        set(value)
    }

    fun setOpenLoop() = changeControlMode(TalonControlMode.PercentVbus)
    fun setPositionControl() = changeControlMode(TalonControlMode.Position)
    fun setVelocityControl() = changeControlMode(TalonControlMode.Speed)
    fun setMagicControl() = changeControlMode(TalonControlMode.MotionMagic)

    @Synchronized
    fun setMagEncoder() {
        setFeedbackDevice(FeedbackDevice.CtreMagEncoder_Relative)
        configEncoderCodesPerRev(4096)
    }

    fun setSlave(master: CANTalon): CANTalonWrapper {
        changeControlMode(TalonControlMode.Follower)
        super.set(master.deviceID.toDouble())
        return this
    }

    infix fun slaveTo(master: CANTalon): CANTalonWrapper {
        return setSlave(master)
    }


    fun setFrameHz(
            generalFrameHz: Double = frameSpeeds[StatusFrameRate.General] ?: 10.0,
            feedbackFrameHz: Double = frameSpeeds[StatusFrameRate.Feedback] ?: 10.0,
            quadFrameHz: Double = frameSpeeds[StatusFrameRate.QuadEncoder] ?: 10.0,
            analogTempVbatFrameHz: Double = frameSpeeds[StatusFrameRate.AnalogTempVbat] ?: 10.0,
            pulseWidthFrameHz: Double = frameSpeeds[StatusFrameRate.PulseWidth] ?: 10.0
            ) {

        frameSpeeds = FrameSpeeds(
                generalFrameHz,
                feedbackFrameHz,
                quadFrameHz,
                analogTempVbatFrameHz,
                pulseWidthFrameHz
        ).speedsMap.apply {
            mapNotNull { entry ->
                setStatusFrameRateMs(entry.key, (1000.0/entry.value).toInt())
            }
        }
    }

}
