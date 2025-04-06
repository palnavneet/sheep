package com.cloudsurfe.sheep.core


sealed class Pipeline{
    data class TextSimilarity(val input1 : String, val input2: String) : Pipeline()
}