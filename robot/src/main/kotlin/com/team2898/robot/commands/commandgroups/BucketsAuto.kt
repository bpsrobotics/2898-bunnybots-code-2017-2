package com.team2898.robot.commands.commandgroups

import com.team2898.engine.kinematics.Rotation2d
import com.team2898.robot.commands.*
import com.team2898.robot.commands.armcommands.KinectPose
import com.team2898.robot.config.AutoConf.*
import com.team2898.robot.subsystems.Drivetrain
import edu.wpi.first.wpilibj.command.CommandGroup
import edu.wpi.first.wpilibj.command.InstantCommand
import edu.wpi.first.wpilibj.command.WaitCommand
import edu.wpi.first.wpilibj.command.WaitForChildren

class BucketsAuto(val num: Int) : CommandGroup() {
    init {
        addSequential(
                object : InstantCommand() {
                    override fun initialize() {
                        Drivetrain.leftMaster.position = 0.0
                        Drivetrain.rightMaster.position = 0.0
                        Drivetrain.leftMaster.encPosition = 0
                        Drivetrain.rightMaster.encPosition = 0
                        Drivetrain.controlMode = Drivetrain.ControlModes.MOTION_MAGIC
                        Drivetrain.gearMode = Drivetrain.GearModes.LOW
                    }
                }
        )
        addParallel(CloseClaw())

        addParallel(ArmPoseDumb(
                elbowPose = Rotation2d(0.0, -1.0),
                wristPose = Rotation2d(1.0, 0.0),
                time = 0.5)
        )
        addSequential(DriveStraightDistance(mid_point_d))
        addSequential(object: WaitCommand(0.25){})
        addSequential(WaitForChildren())
    }
}