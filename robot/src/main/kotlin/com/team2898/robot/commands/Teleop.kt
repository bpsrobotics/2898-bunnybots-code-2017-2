package com.team2898.robot.commands

import com.ctre.CANTalon
import com.team2898.engine.kinematics.Rotation2d
import com.team2898.engine.motion.CheesyDrive
import edu.wpi.first.wpilibj.command.Command
import com.team2898.robot.OI
import com.team2898.robot.VisionComms
import com.team2898.robot.config.OIConf.*
import com.team2898.robot.config.TeleopConfig.useArmController
import com.team2898.robot.subsystems.Claw
import com.team2898.robot.subsystems.Drivetrain
import com.team2898.robot.subsystems.Elbow
import com.team2898.robot.subsystems.Wrist
import edu.wpi.first.wpilibj.Timer
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard as sd

class Teleop : Command() {

    override fun initialize() {
        Wrist.targetPose = Rotation2d(1.0, 0.0)
        Elbow.targetPose = Rotation2d(1.0, 0.0)

    }

    override fun execute() {
        sd.putNumber("elbow output setpoint", Elbow.targetPose.degrees())

        CheesyDrive.updateQuickTurn(OI.quickTurn)

        Drivetrain.openLoopPower =
                CheesyDrive.updateCheesy(
                        (if (!OI.quickTurn) OI.turn else -OI.leftTrigger + OI.rightTrigger),
                        -OI.throttle,
                        OI.quickTurn,
                        true
                )

        if (OI.lowGear) {
            Drivetrain.gearMode = Drivetrain.GearModes.LOW
        } else if (OI.highGear) {
            Drivetrain.gearMode = Drivetrain.GearModes.HIGH
        }

        if (!useArmController) {
            if (Math.abs(Math.max(
                    Math.max(OI.operatorLeftX, OI.operatorLeftY),
                    Math.abs(Math.min(OI.operatorLeftX, OI.operatorLeftY)))) > ARM_CONTROL_DEADZONE_THRESHOLD) {
                Elbow.targetPose = Rotation2d(OI.operatorLeftX, OI.operatorLeftY)
            }

            if (Math.abs(Math.max(
                    Math.max(OI.operatorRightX, OI.operatorRightY),
                    Math.abs(Math.min(OI.operatorRightX, OI.operatorRightY)))) > ARM_CONTROL_DEADZONE_THRESHOLD) {
                Wrist.targetPose = Rotation2d(OI.operatorRightX, OI.operatorRightY)
            }
        } else {
            Elbow.targetPose = Rotation2d.createFromDegrees(OI.armControllerElbowEncoder)
            Wrist.targetPose = Rotation2d.createFromDegrees(OI.armControllerWristEncoder)

            sd.putNumber("arm controller elbow encoder", OI.armControllerElbowEncoder)
            sd.putNumber("arm controller wrist encoder", OI.armControllerWristEncoder)
        }

        sd.putNumber("Elbow setpoint", Elbow.targetPose.radians())
        sd.putNumber("Elbow ramping setpoint", Elbow.speedRamp.getRampedSpeed())


        Claw.clawState = if (OI.claw) Claw.ClawState.OPEN else Claw.ClawState.CLOSED
    }

    override fun isFinished(): Boolean = false
}