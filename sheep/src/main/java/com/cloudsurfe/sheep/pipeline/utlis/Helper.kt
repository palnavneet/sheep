package com.cloudsurfe.sheep.pipeline.utlis

fun generateAttentionMask(inputIds: LongArray): LongArray {
    return inputIds.map { inputId ->
        if (inputId != 0L) 1L else 0L
    }.toLongArray()
}