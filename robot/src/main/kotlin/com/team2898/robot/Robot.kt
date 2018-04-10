package com.team2898.robot

import com.team2898.engine.async.AsyncLooper
import com.team2898.engine.async.util.go
import com.team2898.engine.kinematics.Rotation2d
import com.team2898.engine.logging.LogLevel
import com.team2898.engine.logging.Logger
import com.team2898.engine.logging.reflectLocation
import com.team2898.engine.logic.LoopManager
import com.team2898.robot.commands.CloseClaw
import com.team2898.robot.commands.MidlineDriveVision
import com.team2898.robot.commands.MotionMagicTest
import edu.wpi.first.wpilibj.IterativeRobot
import edu.wpi.first.wpilibj.command.Scheduler
import com.team2898.robot.commands.Teleop
import com.team2898.robot.commands.armcommands.KinectPose
import com.team2898.robot.commands.commandgroups.*
import com.team2898.robot.subsystems.*
import com.team2898.robot.util.PDP
import edu.wpi.first.wpilibj.DigitalOutput
import edu.wpi.first.wpilibj.command.Command
import edu.wpi.first.wpilibj.command.CommandGroup
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard
import kotlinx.coroutines.experimental.delay

class Robot : IterativeRobot() {
    companion object {
        val debug = true
    }

    val teleopCommand = Teleop()
    //val autoCommand = CenterBucketAutoCommandGroup()
    //val autoCommand = DeployArmCommandGroup()
    //val autoCommand = SafeDeployClawCommandGroup()
    //val autoCommand = MidlineDriveVision()
//    val autoCommand = object: CommandGroup() {
//        init {
//            addSequential(KinectPose())
//            addSequential(CloseClaw())
//            addSequential(MidlineDriveVision())
//        }
//    }
    //val autoCommand = BucketCycleCommandGroup()
    var autoCommand: Command = BucketsAuto(2)
    //var autoCommand: Command = CenterOnlyAuto()
    //var autoCommand = MotionMagicTest()

    val llCamLight = DigitalOutput(9)

    override fun robotInit() {
        autoCommand = BucketsAuto(2)

        // Sets parallelism (# of threads) running in the kotlin CommonPool to two, up from cores-1 = 1 by default
        // s/o to tyler from 2471 for this one
        //System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", "2")

        Logger.start()
        Logger.logInfo(reflectLocation(), LogLevel.INFO, "Robot Init")

        //NetworkTable.setUpdateRate(100.0)

        AsyncLooper(10.0) {
            SmartDashboard.putNumber("Wrist position", Wrist.motor.position)
//            SmartDashboard.putNumber("Wrist angle", Wrist.currentPose.degrees())
            SmartDashboard.putNumber("Wrist closed loop error", Wrist.motor.closedLoopError.toDouble())
//            SmartDashboard.putNumber("Wrist output voltage", Wrist.motor.outputVoltage)
            SmartDashboard.putNumber("Wrist pulse width masked", (Wrist.motor.pulseWidthPosition and 0xFFF).toDouble())
            SmartDashboard.putNumber("Wrist x", Wrist.currentPose.cos())
            SmartDashboard.putNumber("Wrist y", Wrist.currentPose.sin())
//            SmartDashboard.putString("Wrist motor mode", Wrist.motor.controlMode.toString())
//            SmartDashboard.putNumber("Wrist motor setpoint", Wrist.motor.setpoint)
//
            SmartDashboard.putNumber("Elbow position", Elbow.motor.position)
//            SmartDashboard.putNumber("Elbow angle", Elbow.currentPose.degrees())
            SmartDashboard.putNumber("Elbow closed loop error", Elbow.motor.closedLoopError.toDouble())
//            SmartDashboard.putNumber("Elbow output voltage", Elbow.motor.outputVoltage)
            SmartDashboard.putNumber("Elbow pulse width masked", (Elbow.motor.pulseWidthPosition and 0xFFF).toDouble())
            SmartDashboard.putNumber("Elbow x", Elbow.currentPose.cos())
            SmartDashboard.putNumber("Elbow y", Elbow.currentPose.sin())

            SmartDashboard.putNumber("Left dt vel", Drivetrain.leftMaster.encVelocity.toDouble())
            SmartDashboard.putNumber("Right dt vel", Drivetrain.rightMaster.encVelocity.toDouble())

            SmartDashboard.putNumber("Limelight angle", VisionComms.limelightYawDegrees)
            SmartDashboard.putNumber("Limelight distance inches", VisionComms.limelightDistanceInches)
            SmartDashboard.putNumber("Limelight area", VisionComms.limelightArea)
            SmartDashboard.putNumber("Adjusted target yaw", VisionComms.bucketAngle)
            SmartDashboard.putNumber("Adjusted target distance", VisionComms.bucketDistance)

            SmartDashboard.putNumber("NavX yaw", Navx.yaw)
            SmartDashboard.putNumber("NavX yaw rate", Navx.yawRate)

            SmartDashboard.putNumber("left dt pos", Drivetrain.leftMaster.position)
            SmartDashboard.putNumber("right dt pos", Drivetrain.rightMaster.position)
        }.start()

        LoopManager.register(Elbow)
        LoopManager.register(Wrist)

        LoopManager.onDisable()
        llCamLight.set(true)

        Drivetrain.leftMaster.position = 0.0
        Drivetrain.rightMaster.position = 0.0
        Drivetrain.leftMaster.encPosition = 0
        Drivetrain.rightMaster.encPosition = 0

        Elbow.rehomePos()
        Wrist.rehomePos()

    }

    override fun autonomousInit() {
        LoopManager.onAutonomous()
        Logger.logInfo(reflectLocation(), LogLevel.INFO, "Autonomous Init")
        //autoCommand.start()
        VisionComms.setLimelightVisionMode()
        llCamLight.set(false)
        Navx.reset()
        Elbow.rehomePos()
        Wrist.rehomePos()

        Drivetrain.gearMode = Drivetrain.GearModes.LOW
        Drivetrain.controlMode = Drivetrain.ControlModes.OPEN_LOOP
        Claw.clawState = Claw.ClawState.CLOSED
        Drivetrain.loop.start()

        Elbow.targetPose = Elbow.currentPose
        Wrist.targetPose = Wrist.currentPose

        if (teleopCommand.isRunning) teleopCommand.cancel()
        autoCommand.start()
    }


    override fun autonomousPeriodic() {
        //Scheduler.getInstance().run()
        //Wrist.motor.setPos(-0.25)
        //Wrist.motor.setPID(4.0, 0.0, 0.5)
        //Wrist.motor.changeControlMode(CANTalon.TalonControlMode.Position)
        //Wrist.motor.set(0.0)
        //Wrist.targetPose = Rotation2d(1.0, 0.0)
        //Elbow.targetPose = Rotation2d(1.0, 0.0)

        Scheduler.getInstance().run()
    }

    override fun teleopInit() {
        Logger.logInfo(reflectLocation(), LogLevel.INFO, "Teleop Init")

        Claw.clawState = Claw.ClawState.CLOSED
        Drivetrain.gearMode = Drivetrain.GearModes.LOW
        Drivetrain.controlMode = Drivetrain.ControlModes.OPEN_LOOP

        Elbow.rehomePos()
        Wrist.rehomePos()

        VisionComms.setLimelightDriverMode()
        llCamLight.set(true)

        LoopManager.onTeleop()
        // TODO: Don't be like this
        Drivetrain.loop.start()

        Elbow.motor.enable()
        Wrist.motor.enable()

        if (autoCommand.isRunning) autoCommand.cancel()
        teleopCommand.start()
    }

    override fun teleopPeriodic() {
        Scheduler.getInstance().run()
    }

    override fun disabledInit() {
        llCamLight.set(true)
        val teleopCommand = Teleop()
        //autoCommand = MotionMagicTest()
        autoCommand = BucketsAuto(2)
        LoopManager.onDisable()
        Drivetrain.loop.stop()
    }
}
