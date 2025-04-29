package com.cloudsurfe.sheep.core

import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import com.cloudsurfe.sheep.pipeline.PipelineType

data class ModelMeta(
    val type: PipelineType,
    val inputNames: List<String>,
    val outputNames: String
)

fun detectModelPipeline(session: OrtSession): ModelMeta {
    val inputs = session.inputInfo.keys.toList()
    val outputs = session.outputInfo


    val outputName = outputs.keys.firstOrNull() ?: return ModelMeta(PipelineType.UNKNOWN, inputs, "")
    val outputInfo = outputs[outputName]!!.info as TensorInfo
    val outputShape = outputInfo.shape

    val modelType = when {
        outputShape.size == 2 && outputName == "logits" -> PipelineType.TEXT_CLASSIFICATION
        outputShape.size == 3 && outputName == "last_hidden_state" -> PipelineType.FEATURE_EXTRACTOR
        inputs.contains("input_ids") && inputs.contains("attention_mask") && outputs.containsKey("start_logits") && outputs.containsKey("end_logits") -> PipelineType.QUESTION_ANSWERING
        else -> PipelineType.UNKNOWN
    }

    return ModelMeta(modelType, inputs, outputName)
}
