package com.team2898.robot.commands.testcommands

import com.ctre.CANTalon
import com.team2898.engine.async.AsyncLooper
import com.team2898.engine.async.util.go
import com.team2898.engine.extensions.drivetrain.blockJoin
import com.team2898.engine.logging.Logger
import com.team2898.engine.motion.DriveSignal
import com.team2898.robot.subsystems.Drivetrain
import com.team2898.robot.subsystems.Navx
import edu.wpi.first.wpilibj.PowerDistributionPanel
import edu.wpi.first.wpilibj.command.Command
import edu.wpi.first.wpilibj.command.TimedCommand
import kotlinx.coroutines.experimental.delay

object DrivetrainStep : TimedCommand(5.0) {

    val loop = AsyncLooper(500.0) {
        Logger.logData("drivetrainstep", "left speed", Drivetrain.encVel.x)
        Logger.logData("drivetrainstep", "right speed", Drivetrain.encVel.y)
        Logger.logData("drivetrainstep", "vbat", Drivetrain.leftMaster.busVoltage)
        Logger.logData("drivetrainstep", "heading", Navx.yaw)
    }

    init {
    }

    override fun initialize() {
        loop.start()
        Drivetrain.controlMode = Drivetrain.ControlModes.OPEN_LOOP
        Drivetrain.masters { setStatusFrameRateMs(CANTalon.StatusFrameRate.QuadEncoder, 5) }
        Drivetrain.masters { setStatusFrameRateMs(CANTalon.StatusFrameRate.Feedback, 5) }
        Drivetrain.masters { setStatusFrameRateMs(CANTalon.StatusFrameRate.AnalogTempVbat, 5) }
        Drivetrain.masters { setStatusFrameRateMs(CANTalon.StatusFrameRate.General, 5) }

        Drivetrain.gearMode = Drivetrain.GearModes.HIGH
        Drivetrain.openLoopPower = DriveSignal(1.0, 1.0)
    }

    override fun execute() {
        Drivetrain.leftMaster.set(1.0)
        Drivetrain.rightMaster.set(-1.0)
    }

    override fun end() {
        Drivetrain.openLoopPower = DriveSignal(0.0, 0.0)
        go {
            Drivetrain.openLoopPower = DriveSignal(0.0, 0.0)
            delay(2000L)
            loop.stop()
        }.blockJoin()
    }
}