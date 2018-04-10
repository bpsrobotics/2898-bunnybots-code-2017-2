package com.team2898.engine.math.linear

import org.apache.commons.math3.geometry.euclidean.twod.Vector2D
import org.apache.commons.math3.linear.Array2DRowRealMatrix

fun rotateVector2D(source: Vector2D, rotation: Vector2D): Vector2D {
    // We need to turn the first vector into a column vector (Array2DRowRealMatrix)
    // We need to turn the second vector into a rotation matrix in the form of
    //| cos(θ) -sin(θ) |
    //| sin(θ)  cos(θ) |
    // We assume that the vectors are in the form <cos(θ), sin(θ)>
    val sourceMatrix = Array2DRowRealMatrix(source.normalize().toArray())
    val normRot = rotation.normalize()
    val rotated = Array2DRowRealMatrix(
            arrayOf(
                    doubleArrayOf(normRot.x, -normRot.y),
                    doubleArrayOf(normRot.y, normRot.x)
            )
    ).multiply(sourceMatrix)
    return Vector2D(rotated.data[0][0], rotated.data[1][0]).normalize()
}