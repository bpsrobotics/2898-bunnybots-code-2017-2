package com.team2898.engine.kinematics

import com.team2898.engine.extensions.Vector2D.atan2
import com.team2898.engine.extensions.Vector2D.l2
import com.team2898.engine.math.linear.rotateVector2D
import com.team2898.engine.types.Interpolable

import org.apache.commons.math3.geometry.euclidean.twod.Vector2D
import java.text.DecimalFormat

// Represents 2D rotation on the x, y plane

class Rotation2d : Interpolable<Rotation2d> {
    companion object { // Allows us to have static traits
        fun createFromRadians(angleRadians: Double): Rotation2d =
                Rotation2d(Math.cos(angleRadians), Math.sin(angleRadians))

        fun createFromDegrees(angleDegrees: Double): Rotation2d =
                createFromRadians(Math.toRadians(angleDegrees))
    }


    // represents as [cos(θ), sin(θ)]. default is 0° (pointing right)
    // for instance, 45° -> [cos(45°), sin(45°)] -> [(√2)/2, (√2)/2]
    var rotation: Vector2D = Vector2D(1.0, 0.0)

    val kEpsilon = 1.0E-8 // "Accuracy," so to speak. We assume values smaller than this are negligible/zero

    constructor()

    constructor(cos: Double, sin: Double) {
        rotation = Vector2D(cos, sin).normalize()
    }

    constructor(toSet: Rotation2d) {
        rotation = toSet.rotation.normalize()
    }

    constructor(toSetVector: Vector2D) {
        rotation = toSetVector.normalize()
    }

    val cos: Double
        get() = rotation.x
    val sin: Double
        get() = rotation.y

    val tan: Double
        get() =
            if (rotation.x > kEpsilon) rotation.y / rotation.x
            else if (rotation.y >= 0.0) Double.POSITIVE_INFINITY
            else Double.NEGATIVE_INFINITY

    val radians: Double
        get() = rotation.atan2

    val degrees: Double
        get() = Math.toDegrees(radians)

    /*
    * Rotation matrix operation to add two Rotation2ds
    */
    infix fun rotateBy(toRotateBy: Rotation2d): Rotation2d {
        val rotated = rotateVector2D(rotation, toRotateBy.rotation)
        return Rotation2d(rotated)
    }

    // Returns inverse of rotation, can undo the effects of the above matrix rotation
    val inverse: Rotation2d
        get() = Rotation2d(cos, -sin)

    override fun interpolate(upperVal: Rotation2d, interpolatePoint: Double): Rotation2d {
        when {
            interpolatePoint <= 0 -> return Rotation2d(this)
            interpolatePoint >= 1 -> return Rotation2d(upperVal)
            else -> return this.rotateBy(
                    Rotation2d.createFromRadians(inverse.rotateBy(upperVal).radians * interpolatePoint)
            )
        }
    }

    override fun toString(): String {
        return "${"%.3f".format(degrees)} deg"
    }
}
