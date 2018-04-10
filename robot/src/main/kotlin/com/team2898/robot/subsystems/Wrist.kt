package com.team2898.robot.subsystems

import com.ctre.CANTalon
import com.team2898.engine.async.AsyncLooper
import com.team2898.engine.kinematics.Rotation2d
import com.team2898.engine.logging.*
import com.team2898.engine.logic.GamePeriods
import com.team2898.engine.logic.ILooper
import com.team2898.engine.logic.ISelfCheck
import com.team2898.engine.motion.CANTalonWrapper
import com.team2898.engine.motion.TrapezoidalRamp
import com.team2898.engine.types.MinMax
import com.team2898.robot.config.WristConf.*
import com.team2898.robot.config.RobotMap.*


object Wrist: ILooper, ISelfCheck {
    val motor = CANTalonWrapper(WRIST_SRX_CANID)

    override val enableTimes = listOf(GamePeriods.AUTO, GamePeriods.TELEOP)

    val currentPose: Rotation2d
        get() = Rotation2d.createFromRadians(
                (motor.position % 1) * 2 * Math.PI
        )
    var targetPose = Rotation2d(1.0, 0.0)
        set(pose) {
            field = pose
        }

    val currentSpeed: Double
        get() = encVelToRadPerSec(motor.encVelocity)


    //TODO: Change Hz to what it should be
    override val loop = AsyncLooper(hz=LOOP_HZ) {
        motor.setPositionControl(poseToEncoderAngle(targetPose))
    }

    init {
        motor.apply {
            enableBrakeMode(true)

            setFeedbackDevice(CANTalon.FeedbackDevice.CtreMagEncoder_Relative)
            //changeControlMode(CANTalon.TalonControlMode.MotionMagic)
            motionMagicAcceleration = motion_magic_acceleration
            motionMagicCruiseVelocity = motion_magic_cruise_velocity
            configEncoderCodesPerRev(4096)
            SetVelocityMeasurementPeriod(CANTalon.VelocityMeasurementPeriod.Period_25Ms)
            SetVelocityMeasurementWindow(100)

            setPID(Kp, Ki, Kd)
            //setPID(0.0, 0.0, 0.0, 0.0223633, 0, 0.0, 1)
            //setPID(KpVel, KiVel, KdVel, KfVel,0, 0.0, 1)
            //setProfile(0)
            //setAllowableClosedLoopErr(0)

            encPosition = (pulseWidthPosition and 0xFFF) - ABSOLUTE_OFFSET

            configNominalOutputVoltage(+0.0, -0.0)
            configPeakOutputVoltage(+12.0, -12.0)

            reverseSensor(true)

            //enableForwardSoftLimit(true)
            //enableReverseSoftLimit(true)

            setForwardSoftLimit(MAX_FORWARD_POSITION)
            setReverseSoftLimit(MAX_REVERSE_POSITION)
            enableForwardSoftLimit(false)
            enableReverseSoftLimit(false)
            setFrameHz(generalFrameHz = 25.0,
                    quadFrameHz = 25.0)
        }
    }

    fun encoderAngleToPose(angle: Double): Rotation2d {
        // TODO
        return Rotation2d(0.0, 1.0)
    }

    fun poseToEncoderAngle(pose: Rotation2d): Double {
        // TODO
        return pose.radians() / (2 * Math.PI)
    }

    // RPM -> Rad/s
    fun encVelToRadPerSec(vel: Int): Double = (vel / 60) * 2 * Math.PI

    // Rad/s -> RPM
    fun radPerSecToEncVel(radPerSec: Double): Double = (radPerSec/(2*Math.PI))*60

    fun encPosRadians() = motor.position * 2 * Math.PI


    override fun selfCheckup(): Boolean {
        try {
            if (motor.isSensorPresent(CANTalon.FeedbackDevice.CtreMagEncoder_Relative) != CANTalon.FeedbackDeviceStatus.FeedbackStatusPresent) {
                throw SelfCheckFailException("Elbow encoder not found", LogLevel.ERROR)
            }
        }
        catch (ex: SelfCheckFailException) {
            Logger.logInfo(reflectLocation(), ex.level, getStackTrace(ex))
            return false
        }
        return true
    }
    override fun selfTest(): Boolean = true

    @Synchronized fun rehomePos() {
        motor.apply {encPosition = (pulseWidthPosition and 0xFFF) - ABSOLUTE_OFFSET}
    }

}
