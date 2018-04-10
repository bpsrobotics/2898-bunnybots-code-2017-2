package com.team2898.robot.config.AutoConf

import com.team2898.engine.kinematics.Rotation2d


val VISION_DISTANCE = 2.0 // In ft

val ARM_MARGIN_ERROR_ROT = 0.05
val CLAW_MARGIN_ERROR_ROT = 100

val CLAW_DEPLOY_SAFETY_THRESHOLD = 90.0

val MOTIONMAGIC_ERROR_THRESHOLD = 500.0

val KINECT_POSE_ELBOW_DEG = -0.310

val WRIST_MAX_POSE_ERROR = 0.1
val ELBOW_MAX_POSE_ERROR = 0.1

val GRAB_POS = NamePair(
        Rotation2d(0.98680940181418553, 0.16188639378011183),
        Rotation2d(0.99988234745421256, -0.0153392062849881)
)

val RAISE_POS = NamePair(
        Rotation2d(0.94758559101774131, -0.31950203081601575),
        Rotation2d(1.0, 0.0)
)

val DUMP_POS = NamePair(
        Rotation2d(0.99170975366909953, -0.12849811079379317),
        Rotation2d(-0.52458968267846884, -0.85135519310526542)
)

val TOSS_POS = NamePair(
        Rotation2d(-0.5504579729366047, -0.83486287498638012),
        Rotation2d(0.63912444486377584, -0.76910333764557981)
)

val CLEAR_POS = NamePair(
        Rotation2d(0.37558617848921733, -0.92678747430458175),
        Rotation2d(-0.22956536582051876, -0.97329324605469825)
)

data class NamePair(val elbowPos: Rotation2d, val wristPos: Rotation2d)

// all from starting point
val bucket_one_d = 6.0 // in inches
val bucket_two_reld = 20.0 // in inches
val bucket_two_d = bucket_one_d + bucket_two_reld
val bucket_three_reld = 24.0 // in inches
val bucket_three_d = bucket_three_reld + bucket_two_d // in inches
//val mid_point_d = 240.0 // distance to mid point in inches
val mid_point_d = 268.0 // distance to mid point in inches
