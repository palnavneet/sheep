package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import com.cloudsurfe.sheep.core.Sheep.Companion.TAG
import com.cloudsurfe.sheep.tokenizer.Tokenizer
import java.nio.LongBuffer
import kotlin.math.exp
//
//class TextClassification(
//) : Pipeline{
//
//    override val numberOfInputs: Int = 1
//    override fun pipeline(onnxTensors: List<OnnxTensor>): Map<Int, String> {
//        return onnxTensors.withIndex().associate {(position,outputTensor) ->
//            val float3dArray = outputTensor.value as Array<Array<FloatArray>>
//            val clsEmbedding : FloatArray = float3dArray[0][0]
//            val weights = FloatArray(768){0.01f}
//            val bias = 0.0f
//            val score = classify(clsEmbedding,weights,bias)
//            val label = if (score > 0.5f) "positive" else "negative"
//            Log.d("Sheep", "$score")
//            position to label
//        }
//    }
//
//    override fun getOutputTensor(
//        session : OrtSession,
//        env : OrtEnvironment,
//        tokenizer: Tokenizer,
//        vararg inputText: String
//    ): List<OnnxTensor> {
//        val outputs = mutableListOf<OnnxTensor>()
//        getInputTensor(
//            env,
//            tokenizer,
//            *inputText
//        ).forEach { (index, inputTensor) ->
//            val inputMap: Map<String, OnnxTensor> = mapOf("input_ids" to inputTensor)
//            val result = session.run(inputMap)
//            val optionalOutput = result.get("last_hidden_state")
//            if (optionalOutput.isPresent && optionalOutput.get() is OnnxTensor) {
//                val outputTensor = optionalOutput.get() as OnnxTensor
//                outputs.add(outputTensor)
//            } else {
//                Log.d(TAG, "Warning: No valid output for index $index")
//            }
//        }
//        return outputs
//    }
//
//    override fun getInputTensor(
//        env : OrtEnvironment,
//        tokenizer: Tokenizer,
//        vararg inputText: String
//    ): Map<Int, OnnxTensor> {
//        return tokenizer(
//            tokenizer,
//            *inputText
//        ).withIndex().associate { (index, tokenizedInput) ->
//            index to shape(env,tokenizedInput)
//        }
//    }
//
//    override fun tokenizer(
//        tokenizer: Tokenizer,
//        vararg inputText: String
//    ): List<LongArray> {
//        tokenizer.apply {
//            loadVocab()
//        }
//        return inputText.map { text -> tokenizer.tokenize(text) }
//    }
//
//    override fun shape(env: OrtEnvironment, inputId: LongArray): OnnxTensor {
//        val shape: LongArray = longArrayOf(1, inputId.size.toLong())
//        return OnnxTensor.createTensor(env, LongBuffer.wrap(inputId), shape)
//    }
//
//    fun sigmoid(x : Float) : Float{
//        return 1f / (1f + exp(-x))
//    }
//
//    fun classify(clsEmbedding : FloatArray, weights : FloatArray, bias : Float) : Float{
//        var sum = 0f
//        for (i in clsEmbedding.indices){
//            sum += clsEmbedding[i] * weights[i]
//        }
//        return sigmoid(sum + bias)
//    }
//
//
//
//}