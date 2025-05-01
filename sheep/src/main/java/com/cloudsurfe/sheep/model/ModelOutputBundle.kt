package com.cloudsurfe.sheep.model

import ai.onnxruntime.OnnxTensor

data class ModelOutputBundle(
    val tensors : List<Map<String, OnnxTensor>>,
    val tokenizationResultSet : TokenizationResultSet
)