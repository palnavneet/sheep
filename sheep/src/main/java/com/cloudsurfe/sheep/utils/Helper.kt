package com.cloudsurfe.sheep.utils

fun generateAttentionMask(inputIds: LongArray): LongArray {
    return inputIds.map { inputId ->
        if (inputId != 0L) 1L else 0L
    }.toLongArray()
}

fun softmax(input: FloatArray): DoubleArray {
    val max = input.maxOrNull()?.toDouble() ?: 0.0
    val sum = input.sumOf { Math.exp(it.toDouble() - max) }
    return input.map { Math.exp(it.toDouble() - max) / sum }.toDoubleArray()
}

fun argmax(input : DoubleArray) : Int{
    return input.indices.maxByOrNull { input[it] } ?: -1
}

fun computeOffsetMapping(tokens: List<String>, originalText: String): List<Pair<Int, Int>> {
    val offsetMapping = mutableListOf<Pair<Int, Int>>()
    var currentPos = 0

    for (token in tokens) {
        when (token) {
            "[CLS]", "[SEP]" -> {
                offsetMapping.add(0 to 0)
            }

            "[UNK]" -> {
                val remainingText = originalText.substring(currentPos)
                val unknownWord = remainingText.split(" ").firstOrNull() ?: ""
                val start = currentPos
                val end = start + unknownWord.length
                offsetMapping.add(start to end)
                currentPos = end
            }

            else -> {
                val start = originalText.indexOf(token, currentPos)
                if (start == -1) {
                    offsetMapping.add(0 to 0)
                } else {
                    val end = start + token.length
                    offsetMapping.add(start to end)
                    currentPos = end
                }
            }
        }
    }
    return offsetMapping
}





















