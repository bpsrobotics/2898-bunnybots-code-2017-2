package com.team2898.robot.subsystems

import com.ctre.CANTalon
import com.team2898.engine.async.AsyncLooper
import com.team2898.engine.kinematics.Translation2d
import com.team2898.engine.logging.LogLevel
import com.team2898.engine.logging.Logger
import com.team2898.engine.logging.reflectLocation
import com.team2898.engine.logic.*
import com.team2898.robot.config.RobotMap.*
import com.team2898.engine.motion.DriveSignal
import com.team2898.engine.motion.CANTalonWrapper
import com.team2898.engine.types.MinMax
import com.team2898.robot.config.DrivetrainConf.*
import edu.wpi.first.wpilibj.DoubleSolenoid
import org.apache.commons.math3.geometry.euclidean.twod.Vector2D
import java.util.*

object Drivetrain : Subsystem(50.0) {

    enum class ControlModes { OPEN_LOOP, BASE_LOCK, VELOCITY_DRIVE, MOTION_MAGIC }
    enum class ControlSources { DIRECT_CONTROL, PURSUIT_CONTROL, ANGLE_LOCK_CONTROL }

    enum class GearModes(val value: DoubleSolenoid.Value) {
        HIGH(DoubleSolenoid.Value.kForward),
        LOW(DoubleSolenoid.Value.kReverse)
    }

    val leftMaster = CANTalonWrapper(LEFT_MASTER_CANID)
    val rightMaster = CANTalonWrapper(RIGHT_MASTER_CANID)
    val leftSlave = CANTalonWrapper(LEFT_SLAVE_CANID)
    val rightSlave = CANTalonWrapper(RIGHT_SLAVE_CANID)
    val masterList = listOf(leftMaster, rightMaster)

    val encVel
        get() = Vector2D(leftMaster.encVelocity.toDouble(), rightMaster.encVelocity.toDouble())
    val encPos
        get() = Vector2D(leftMaster.encPosition.toDouble(), rightMaster.encPosition.toDouble())

    override val enableTimes = listOf(GamePeriods.TELEOP, GamePeriods.AUTO)

    val shifterSolenoid = DoubleSolenoid(DT_SOLENOID_FORWARD_ID, DT_SOLENOID_REVERSE_ID)

    var baseLockPos = Pair(0.0, 0.0)

    // -1 to 1, vbat
    var openLoopPower = DriveSignal.BRAKE
    var closedLoopVelTarget = DriveSignal.NEUTRAL
    var motionMagicPositions = DriveSignal.NEUTRAL

    // Manages control modes
    val driveStateMachine = StateMachine()


    var controlMode = ControlModes.OPEN_LOOP
        set(value: ControlModes) {
            if (field != value) {
                driveStateMachine changeStateTo value
                field = value
            }
        }

    var gearMode = GearModes.HIGH
        set(value) {
            field = value
            if (shifterSolenoid.get() != field.value) {
                shifterSolenoid.set(field.value)
                masters {
                    setCurrentLimit(
                            if (field == GearModes.HIGH) HIGH_GEAR_MAX_AMPS else LOW_GEAR_MAX_AMPS
                    )
                }

                leftSlave.setCurrentLimit(
                        if (field == GearModes.HIGH) HIGH_GEAR_MAX_AMPS else LOW_GEAR_MAX_AMPS
                )
                rightSlave.setCurrentLimit(
                        if (field == GearModes.HIGH) HIGH_GEAR_MAX_AMPS else LOW_GEAR_MAX_AMPS
                )
            }
        }

    override fun onStart() {}

    override fun onLoop() {

        if (shifterSolenoid.get() != gearMode.value) {
            shifterSolenoid.set(gearMode.value)
            if (shifterSolenoid.get() == GearModes.HIGH.value)
                masters { setProfile(0) }
            else
                masters { setProfile(1) }
        }

        driveStateMachine.update()
    }

    override fun onStop() {}

    init {
        // Set slaves
        leftSlave slaveTo leftMaster
        rightSlave slaveTo rightMaster

        leftSlave.enableBrakeMode(true)
        rightSlave.enableBrakeMode(true)
        leftMaster.enableBrakeMode(true)
        rightMaster.enableBrakeMode(true)

        leftMaster.reverseSensor(true)
        rightMaster.reverseSensor(true)

        masters { setVoltageRampRate(72.0) }

        // Set master characteristics
        masters {
            closeLoopRampRate = 144.0
            enableBrakeMode(true)

            setFeedbackDevice(CANTalon.FeedbackDevice.CtreMagEncoder_Relative)
            configEncoderCodesPerRev(4096)


            //setPID(KP_HIGH, KI_HIGH, KD_HIGH, KF_HIGH, 0, HIGH_GEAR_RAMPRATE, 0)
            setPID(KP_LOW, KI_LOW, KD_LOW, KF_LOW, 240, LOW_GEAR_RAMPRATE, 0)
            setPID(KP_LOW, KI_LOW, KD_LOW, KF_LOW, 240, LOW_GEAR_RAMPRATE, 1)

            // yay, blindly copying 254
            SetVelocityMeasurementPeriod(CANTalon.VelocityMeasurementPeriod.Period_10Ms)
            SetVelocityMeasurementWindow(32)

            EnableCurrentLimit(false)

            setProfile(1)
        }
        leftMaster.apply {
            motionMagicAcceleration = (1250.0 / 2.0) * LEFT_MM_STRAIGHT_ADJ_FACTOR
            motionMagicCruiseVelocity = (1250.0) * LEFT_MM_STRAIGHT_ADJ_FACTOR
        }
        rightMaster.apply {
            motionMagicAcceleration = 1250.0 / 2.0
            motionMagicCruiseVelocity = 1250.0
        }

        driveStateMachine.apply {
            registerWhile(ControlModes.OPEN_LOOP) {
                masters {
                    //enableBrakeMode(openLoopPower.brake)
                }
                leftMaster.setOpenLoop(openLoopPower.left)
                rightMaster.setOpenLoop(-openLoopPower.right)
            }
            registerTo(ControlModes.OPEN_LOOP) { masters { setOpenLoop() } }
            registerFrom(ControlModes.OPEN_LOOP) { openLoopPower = DriveSignal.BRAKE }

            registerWhile(ControlModes.VELOCITY_DRIVE) {
            }

            registerTo(ControlModes.VELOCITY_DRIVE) {
                masters { setVelocityControl() }
            }

            registerFrom(ControlModes.VELOCITY_DRIVE) { closedLoopVelTarget = DriveSignal.NEUTRAL }

            registerTo(ControlModes.MOTION_MAGIC) {
                masters { setMagicControl() }
            }
            registerWhile(ControlModes.MOTION_MAGIC) {
                masters { setMagicControl() }
                leftMaster.setMagicControl(motionMagicPositions.left)
                rightMaster.setMagicControl(-motionMagicPositions.right)

                println("Setting motion magic to ${motionMagicPositions.left}, ${motionMagicPositions.right}")
                println("Speeds: ${encVel.x}, ${encVel.y}")
                println("Positions: ${encPos.x}, ${encPos.y}")
                println("Errors: ${leftMaster.closedLoopError}, ${rightMaster.closedLoopError}")
            }

        }

        driveStateMachine.changeStateTo(controlMode)
    }

    fun masters(block: CANTalonWrapper.() -> Unit) {
        masterList.forEach { srx ->
            srx.block()
        }
    }

    override fun selfCheckup(): Boolean {
        return false
    }

    override fun selfTest(): Boolean {
        return false
    }

    fun getInchesToRotation(inches: Double): Double {
        val wheel = 6 * Math.PI
        return (inches / wheel) * 1 // previously * 4
    }
}


