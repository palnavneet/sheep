package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor

interface Pipeline{
    val numberOfInputs : Int

    fun pipeline(onnxTensors : List<OnnxTensor>) : Map<Int, String>

}