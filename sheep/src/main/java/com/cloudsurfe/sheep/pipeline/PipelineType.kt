package com.cloudsurfe.sheep.pipeline

sealed class PipelineType {
    data class TextSimilarity(val input1: String, val input2: String) : PipelineType()
    data class CustomPipeline(val pipeline: Pipeline) : PipelineType()
}