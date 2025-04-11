package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import com.cloudsurfe.sheep.tokenizer.Tokenizer
import java.nio.LongBuffer
import kotlin.math.exp


class TextClassificationFineTuned() : Pipeline {

    override val numberOfInputs: Int = 1
    override fun pipeline(onnxTensors: List<Map<String, OnnxTensor>>): List<Map<Int, String>> {
        val float2dArray = onnxTensors.withIndex().forEachIndexed { index, onnxTensor ->
            val outputTensor = onnxTensors[index]
            val float3DArray = outputTensor["logits"]?.value as Array<FloatArray>
            Log.d("Sheep", "$float3DArray")
            Log.d("Sheep", "Batch size: ${float3DArray.size}")
            Log.d("Sheep", "Sequence length: ${float3DArray[0].size}")
            Log.d("Sheep", "inside pipeline")
            val clsEmbedding : FloatArray = float3DArray[0]
            val weights = FloatArray(768){0.01f}
            val bias = 0.0f
            val score = classify(clsEmbedding,weights,bias)
            val label = if (score > 0.5f) "positive" else "negative"
            Log.d("Sheep", "$score")
        }
        return emptyList()
    }

    override fun getOutputTensor(
        session: OrtSession,
        env: OrtEnvironment,
        tokenizer: Tokenizer,
        vararg inputText: String
    ): List<Map<String, OnnxTensor>> {

        val outputs = mutableListOf<Map<String, OnnxTensor>>()
        getInputTensor(
            env,
            tokenizer,
            *inputText
        ).forEach { inputTensorWithMask ->
            val inputIdsInputTensor = inputTensorWithMask["input_Ids"]
            val attentionMaskInputTensor = inputTensorWithMask["attention_mask"]

            if (inputIdsInputTensor != null && attentionMaskInputTensor != null) {

                val inputMap: Map<String, OnnxTensor> = mapOf(
                    "input_ids" to inputIdsInputTensor,
                    "attention_mask" to attentionMaskInputTensor
                )

                val result = session.run(inputMap)
                val optionalOutput = result.get("logits")
                if (optionalOutput.isPresent && optionalOutput.get() is OnnxTensor) {
                    val outputTensor = optionalOutput.get() as OnnxTensor
                    outputs.add(mapOf("logits" to outputTensor))
                    Log.d("Sheep", "Output tensor")
                }

            }

        }
        return outputs
    }

    override fun getInputTensor(
        env: OrtEnvironment,
        tokenizer: Tokenizer,
        vararg inputText: String
    ): List<Map<String, OnnxTensor>> {
        Log.d("Sheep", "Input tensor")
        return tokenizer(
            tokenizer,
            *inputText
        ).map { input_Id ->
            val attention_mask = getStandardInputComponents(input_Id)
            val inputMap = mutableMapOf<String, OnnxTensor>()
            inputMap["input_Ids"] = shape(env, input_Id)
            inputMap["attention_mask"] = shape(env, attention_mask)

            inputMap
        }
    }

    override fun tokenizer(
        tokenizer: Tokenizer,
        vararg inputText: String
    ): List<LongArray> {
        Log.d("Sheep", "Tokenizer")
        tokenizer.apply {
            loadVocab()
        }
        return inputText.map { text -> tokenizer.tokenize(text) }
    }

    override fun shape(
        env: OrtEnvironment,
        inputId: LongArray
    ): OnnxTensor {
        Log.d("Sheep", "Inside shape")
        val shape: LongArray = longArrayOf(1, inputId.size.toLong()) // Re-evaluation
        return OnnxTensor.createTensor(env, LongBuffer.wrap(inputId), shape)
    }

    private fun getStandardInputComponents(inputIds: LongArray): LongArray {
        return inputIds.map { inputId ->
            if (inputId != 0L) 1L else 0L
        }.toLongArray()
    }
    fun sigmoid(x: Float): Float {
        return 1f / (1f + exp(-x))
    }

    fun classify(clsEmbedding: FloatArray, weights: FloatArray, bias: Float): Float {
        var sum = 0f
        for (i in clsEmbedding.indices) {
            sum += clsEmbedding[i] * weights[i]
        }
        return sigmoid(sum + bias)
    }
}

