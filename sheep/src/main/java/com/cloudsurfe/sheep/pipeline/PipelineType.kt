package com.cloudsurfe.sheep.pipeline

sealed class PipelineType {
    data class TextClassification(val input: String) : PipelineType()
    data class TextClassificationFineTuned(val input: String) : PipelineType()
    class CustomPipeline(val inputs: Array<out String>) : PipelineType()
}