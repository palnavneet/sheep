package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor

class TextSimilarity(
) : Pipeline{

    override val numberOfInputs: Int = 2
    override fun pipeline(onnxTensors: List<Array<Array<FloatArray>>>): Map<Int, String> {
        return emptyMap()
    }

}