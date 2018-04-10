package com.team2898.engine.async.util

import kotlinx.coroutines.experimental.runBlocking
import kotlinx.coroutines.experimental.sync.Mutex
import kotlinx.coroutines.experimental.sync.withLock

class SafeVar<T>(var value: T) {
    val lock: Mutex = Mutex()

    @Synchronized
    fun get(): T = runBlocking<T> {
        return@runBlocking lock.withLock<T>() {
            return@withLock value
        }
    }

    @Synchronized
    fun set(new: T) {
        runBlocking {
            lock.withLock {
                value = new
            }
        }
    }

    @Synchronized
    fun runSafe(block: () -> Unit) {
        runBlocking<Unit> {
            lock.withLock {
                block()
            }
        }
    }
}