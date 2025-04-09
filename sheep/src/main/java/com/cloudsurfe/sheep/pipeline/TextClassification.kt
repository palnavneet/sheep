package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor
import android.util.Log
import kotlin.math.exp

class TextClassification(
) : Pipeline{

    override val numberOfInputs: Int = 1
    override fun pipeline(onnxTensors: List<OnnxTensor>): Map<Int, String> {
        return onnxTensors.withIndex().associate {(position,outputTensor) ->
            val float3dArray = outputTensor.value as Array<Array<FloatArray>>
            val clsEmbedding : FloatArray = float3dArray[0][0]
            val weights = FloatArray(768){0.01f}
            val bias = 0.0f
            val score = classify(clsEmbedding,weights,bias)
            val label = if (score > 0.5f) "positive" else "negative"
            Log.d("Sheep", "$score")
            position to label
        }
    }

    fun sigmoid(x : Float) : Float{
        return 1f / (1f + exp(-x))
    }

    fun classify(clsEmbedding : FloatArray, weights : FloatArray, bias : Float) : Float{
        var sum = 0f
        for (i in clsEmbedding.indices){
            sum += clsEmbedding[i] * weights[i]
        }
        return sigmoid(sum + bias)
    }



}