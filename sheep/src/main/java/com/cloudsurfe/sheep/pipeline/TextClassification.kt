package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.cloudsurfe.sheep.tokenizer.Tokenizer
import java.nio.LongBuffer
import kotlin.collections.toFloatArray

class TextClassification(
) : Pipeline {

    override fun pipeline(onnxTensors: List<Map<String, OnnxTensor>>): List<Map<String, String>> {
        val output = mutableListOf<Map<String, String>>()
        onnxTensors.withIndex().forEachIndexed { index, outputTensor ->
            val outputTensor = onnxTensors[index]
            val float3dArray = outputTensor["last_hidden_state"]?.value as Array<Array<FloatArray>>
            val labels = listOf("Negative", "Positive")
            val clsEmbedding = float3dArray[0][0]
            val weights = arrayOf(
                FloatArray(clsEmbedding.size) { i -> if (i % 2 == 0) 0.02f else -0.01f },
                FloatArray(clsEmbedding.size) { i -> if (i % 2 != 0) 0.02f else -0.01f }
            )
            val logits = weights.map { weightVector ->
                weightVector.zip(clsEmbedding).sumOf { (w, e) -> (w * e).toDouble() }
            }.map { it.toFloat() }.toFloatArray()
            val softmaxLogits = softmax(logits)
            val predictedIndex = softmaxLogits.indices.maxByOrNull { softmaxLogits[it] } ?: -1
            val predictedLabel = labels[predictedIndex]
            val confidence = softmaxLogits[predictedIndex] * 100

            val result = mapOf(
                "predicted_label" to predictedLabel,
                "confidence" to "%.2f".format(confidence)
            )

            output.add(
                mapOf(
                    "predicted_label" to predictedLabel,
                    "confidence" to "%.2f".format(confidence)
                )
            )
        }

        return output
    }

    override fun <T> getOutputTensor(
        session: OrtSession,
        env: OrtEnvironment,
        tokenizer: Tokenizer,
        vararg input: Pair<String, T>
    ): List<Map<String, OnnxTensor>> {
        val inputMap = mapOf(*input)
        val requiredInputs = inputMap["inputs"] as List<String>
        val outputs = mutableListOf<Map<String, OnnxTensor>>()
        getInputTensor(
            env,
            tokenizer,
            requiredInputs
        ).forEach { inputTensorWithMask ->
            val inputIdsInputTensor = inputTensorWithMask["input_Ids"]
            if (inputIdsInputTensor != null) {

                val inputMap: Map<String, OnnxTensor> = mapOf(
                    "input_ids" to inputIdsInputTensor
                )

                val result = session.run(inputMap)
                val optionalOutput = result.get("last_hidden_state")
                if (optionalOutput.isPresent && optionalOutput.get() is OnnxTensor) {
                    val outputTensor = optionalOutput.get() as OnnxTensor
                    outputs.add(mapOf("last_hidden_state" to outputTensor))

                }
            }
        }
        return outputs
    }

    override fun getInputTensor(
        env: OrtEnvironment,
        tokenizer: Tokenizer,
        input: List<String>
    ): List<Map<String, OnnxTensor>> {
        return tokenizer(
            tokenizer,
            input
        ).map { input_Id ->
            val inputMap = mutableMapOf<String, OnnxTensor>()
            inputMap["input_Ids"] = shape(env, input_Id)
            inputMap
        }
    }

    override fun tokenizer(
        tokenizer: Tokenizer,
        input : List<String>
    ): List<LongArray> {
        tokenizer.apply {
            loadVocab()
        }
        return input.map { text -> tokenizer.tokenize(text) }
    }

    override fun shape(env: OrtEnvironment, inputId: LongArray): OnnxTensor {
        val shape: LongArray = longArrayOf(1, inputId.size.toLong())
        return OnnxTensor.createTensor(env, LongBuffer.wrap(inputId), shape)
    }

    fun softmax(logits: FloatArray, temperature: Float = 1.0f): FloatArray {
        val exp = logits.map { Math.exp((it / temperature).toDouble()) }
        val sumExp = exp.sum()
        return exp.map { (it / sumExp).toFloat() }.toFloatArray()
    }


}