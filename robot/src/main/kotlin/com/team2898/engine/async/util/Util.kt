package com.team2898.engine.async.util

import kotlinx.coroutines.experimental.CommonPool
import kotlinx.coroutines.experimental.Job
import kotlinx.coroutines.experimental.launch
import kotlinx.coroutines.experimental.runBlocking
import kotlin.system.measureTimeMillis

fun go(block: suspend () -> Unit): Job = launch(CommonPool) { block() }

class WaitGroup {
    val jobs = mutableListOf<Job>()
    fun add(job: Job): WaitGroup = this.apply { jobs.add(job) }
    fun add(block: suspend () -> Unit): WaitGroup = this.apply {jobs.add(go{block()})}
    fun wait(): Long = measureTimeMillis { runBlocking<Unit> { jobs.forEach { it.join() } } }
}
