package com.team2898.robot.commands.commandgroups

import com.team2898.engine.kinematics.Rotation2d
import com.team2898.robot.commands.*
import com.team2898.robot.commands.armcommands.KinectPose
import com.team2898.robot.config.AutoConf.*
import com.team2898.robot.subsystems.Drivetrain
import edu.wpi.first.wpilibj.command.*

class CenterOnlyAutoJank() : CommandGroup() {
    init {
        addSequential(
                object : InstantCommand() {
                    override fun initialize() {
                        Drivetrain.leftMaster.position = 0.0
                        Drivetrain.rightMaster.position = 0.0
                        Drivetrain.leftMaster.encPosition = 0
                        Drivetrain.rightMaster.encPosition = 0
                        Drivetrain.controlMode = Drivetrain.ControlModes.MOTION_MAGIC
                    }
                }
        )

        addSequential(CloseClaw())

        addSequential(ArmPoseDumb(
                elbowPose = Rotation2d(0.0, -1.0),
                wristPose = Rotation2d(1.0, 0.0),
                time = 1.0)
        )

        addSequential(StraightAuto(45.0))

        addSequential(ArmPoseDumb(
                elbowPose = Rotation2d(1.0, 0.0),
                wristPose = Rotation2d(1.0, 0.0),
                time = 1.0)
        )
        addSequential(OpenClaw())
        addSequential(StraightAuto(60.0))
        addSequential(ArmPoseDumb(GRAB_POS.elbowPos, GRAB_POS.wristPos, 0.05))
        addSequential(CloseClaw())
        addSequential(ArmPoseDumb(RAISE_POS.elbowPos, RAISE_POS.wristPos, 0.75))
        addSequential(ArmPoseDumb(DUMP_POS.elbowPos, DUMP_POS.wristPos, 0.75))
    }
}
