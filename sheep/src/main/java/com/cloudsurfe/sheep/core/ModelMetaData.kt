package com.cloudsurfe.sheep.core

import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo

data class ModelMeta(
    val type: ModelType,
    val inputNames: List<String>,
    val outputNames: String
)

enum class ModelType {
    TEXT_CLASSIFICATION,
    FEATURE_EXTRACTOR,
    UNKNOWN
}

fun detectModelPipeline(session: OrtSession): ModelMeta {
    val inputs = session.inputInfo.keys.toList()
    val outputs = session.outputInfo


    val outputName = outputs.keys.firstOrNull() ?: return ModelMeta(ModelType.UNKNOWN, inputs, "")
    val outputInfo = outputs[outputName]!!.info as TensorInfo
    val outputShape = outputInfo.shape

    val modelType = when {
        outputShape.size == 2 && outputName == "logits" -> ModelType.TEXT_CLASSIFICATION
        outputShape.size == 3 && outputName == "last_hidden_state" -> ModelType.FEATURE_EXTRACTOR
        else -> ModelType.UNKNOWN
    }

    return ModelMeta(modelType, inputs, outputName)
}
