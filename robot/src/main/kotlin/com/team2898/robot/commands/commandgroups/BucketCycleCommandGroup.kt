package com.team2898.robot.commands.commandgroups

import com.team2898.robot.commands.ArmPoseDumb
import com.team2898.robot.commands.CloseClaw
import com.team2898.robot.commands.OpenClaw
import com.team2898.robot.config.AutoConf.*
import com.team2898.robot.subsystems.Claw
import edu.wpi.first.wpilibj.command.CommandGroup
import edu.wpi.first.wpilibj.command.WaitCommand

class BucketCycleCommandGroup: CommandGroup() {
    init {
        //addSequential(CloseClaw()) // TODO: REMOVE BEFORE COMPETITION

        addSequential(ArmPoseDumb(GRAB_POS.elbowPos, GRAB_POS.wristPos, 0.75))
        addSequential(CloseClaw())
        addSequential(ArmPoseDumb(RAISE_POS.elbowPos, RAISE_POS.wristPos, 0.75))
        addSequential(ArmPoseDumb(DUMP_POS.elbowPos, DUMP_POS.wristPos, 0.75))
        addSequential(ArmPoseDumb(TOSS_POS.elbowPos, TOSS_POS.wristPos, 0.75))
        addSequential(OpenClaw())
        addSequential(object: WaitCommand(0.15){})
        addSequential(ArmPoseDumb(CLEAR_POS.elbowPos, CLEAR_POS.wristPos, 0.5))
        addSequential(object: WaitCommand(0.1){})
        addSequential(CloseClaw())
        addSequential(object: WaitCommand(0.1){})
        addSequential(ArmPoseDumb(GRAB_POS.elbowPos, GRAB_POS.wristPos, 1.0))
        addSequential(OpenClaw())
    }
}