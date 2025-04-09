package com.cloudsurfe.sheep.pipeline

sealed class PipelineType {
    data class TextSimilarity(val input1: String, val input2: String) : PipelineType()
    class CustomPipeline(val inputs: Array<out String>) : PipelineType()
}