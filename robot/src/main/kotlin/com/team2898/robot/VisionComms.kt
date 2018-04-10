package com.team2898.robot

import com.team2898.engine.extensions.Vector2D.atan2
import com.team2898.engine.extensions.Vector2D.l2
import com.team2898.robot.config.RobotPhysicalConf.limelightToClawTransform
import org.apache.commons.math3.geometry.euclidean.twod.Vector2D
import edu.wpi.first.wpilibj.networktables.NetworkTable as nt

object VisionComms {

    private val LimelightTable = nt.getTable("limelight")
    private val ClassicTable = nt.getTable("vision")

    val limelightYawDegrees: Double
        get() = LimelightTable.getNumber("tx", 0.0)

    val limelightDistanceInches: Double
        get() = limelightDistanceFromArea(limelightArea)

    val limelightArea: Double
        get() = LimelightTable.getNumber("ta", 0.0)

    val bucketAngle: Double
        get() = limelightToClaw(angle = limelightYawDegrees, distance = limelightDistanceInches).theta

    val bucketDistance: Double
        get() = limelightToClaw(angle = limelightYawDegrees, distance = limelightDistanceInches).r


    fun setLimelightVisionMode() {
        LimelightTable.putNumber("ledMode", 0.0)
        LimelightTable.putNumber("camMode", 0.0)
    }

    fun setLimelightDriverMode() {
        LimelightTable.putNumber("ledMode", 1.0)
        LimelightTable.putNumber("camMode", 1.0)
    }


    private fun limelightDistanceFromArea(area: Double) = 0.3878934 + (21.76969 - 0.3878934) / (1 + Math.pow(area / 0.1025658, .7963151)) * 12

    private fun limelightToClaw(angle: Double, distance: Double): Polar2D {
        val theta = Math.toRadians(angle)
        val d1 = distance
        val xy1 = Vector2D(
                d1 * Math.cos(theta),
                d1 * Math.sin(theta)
        )
        val xy2 = Vector2D(xy1.x - limelightToClawTransform.x, xy1.y - limelightToClawTransform.y)
        val phi = xy2.atan2
        val d2 = xy2.l2
        val phiDegrees = Math.toDegrees(phi)
        return Polar2D(theta = phiDegrees, r = d2)
    }

    private data class Polar2D(val theta: Double, val r: Double)
}
