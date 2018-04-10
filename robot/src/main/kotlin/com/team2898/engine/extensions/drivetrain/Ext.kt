package com.team2898.engine.extensions.drivetrain

import kotlinx.coroutines.experimental.Job
import kotlinx.coroutines.experimental.runBlocking

fun Job.blockJoin() = runBlocking<Unit> {join()}

// Used for nullifying function returns
// (eg. we return a Job from AsyncLooper start() and stop() that we don't necessarily want to pass on if we're
// aliasing the function in a class)

fun Any.Unit() = Unit
