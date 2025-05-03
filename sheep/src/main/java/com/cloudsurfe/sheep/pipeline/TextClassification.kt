package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.cloudsurfe.sheep.model.ModelOutputBundle
import com.cloudsurfe.sheep.model.TokenizationResultSet
import com.cloudsurfe.sheep.tokenizer.Tokenizer
import com.cloudsurfe.sheep.utils.generateAttentionMask
import java.nio.LongBuffer
import kotlin.collections.toFloatArray

internal class TextClassification(
) : Pipeline {

    override fun pipeline(modelOutputBundle: ModelOutputBundle): List<Map<String, String>> {
        val output = mutableListOf<Map<String, String>>()
        modelOutputBundle.tensors.forEach{ outputTensor ->
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
    ): ModelOutputBundle {
        val inputMap = mapOf(*input)
        val requiredInputs = inputMap["inputs"] as List<String>
        val inputTensorBundle = getInputTensor(env,tokenizer,requiredInputs)
        val outputs = mutableListOf<Map<String, OnnxTensor>>()
        inputTensorBundle.tensors.forEach { inputTensorWithMask ->
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
        return ModelOutputBundle(
            tensors = outputs,
            tokenizationResultSet = inputTensorBundle.tokenizationResultSet
        )
    }

    override fun getInputTensor(
        env: OrtEnvironment,
        tokenizer: Tokenizer,
        input: List<String>
    ): ModelOutputBundle {
        val tokenizationResultSet : TokenizationResultSet = tokenizer(tokenizer, input)
        val tensors: List<Map<String, OnnxTensor>> = tokenizationResultSet.tokenizedInput.map { inputId ->
            mapOf(
                "input_Ids" to shape(env,inputId)
            )
        }
        return ModelOutputBundle(
            tensors = tensors,
            tokenizationResultSet = tokenizationResultSet
        )
    }

    override fun tokenizer(
        tokenizer: Tokenizer,
        input: List<String>
    ): TokenizationResultSet {
        tokenizer.apply {
            loadVocab()
        }
        val tokenizedInput = input.map { text -> tokenizer.tokenize(text) }
        val detokenizedInput = tokenizedInput.map {tokenizedInput ->
            tokenizedInput.joinToString(","){
                tokenizer.deTokenize(it.toInt())
            }
        }
        return TokenizationResultSet(
            input = input,
            tokenizedInput = tokenizedInput,
            detokenizedInput = detokenizedInput
        )
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