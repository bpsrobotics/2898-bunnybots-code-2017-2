package com.team2898.robot.commands.commandgroups

import com.team2898.engine.logging.Logger
import com.team2898.robot.commands.CloseClaw
import com.team2898.robot.commands.DtLowGear
import com.team2898.robot.commands.MidlineDriveFinalApproach
import com.team2898.robot.commands.MidlineDriveVision
import com.team2898.robot.commands.armcommands.KinectPose
import edu.wpi.first.wpilibj.command.CommandGroup
import edu.wpi.first.wpilibj.command.WaitForChildren

class CenterBucketAutoCommandGroup(): CommandGroup() {
    init {
        addSequential(KinectPose())
        addSequential(DtLowGear())
        addSequential(MidlineDriveVision())
        //addParallel(DeployArmCommandGroup())
        addSequential(WaitForChildren())
        addSequential(CloseClaw())
    }
}