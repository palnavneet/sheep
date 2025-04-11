package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import com.cloudsurfe.sheep.tokenizer.Tokenizer
import java.nio.LongBuffer


class TextClassificationFineTuned() : Pipeline {

    override val numberOfInputs: Int = 1
    override fun pipeline(onnxTensors: List<Map<String, OnnxTensor>>): List<Map<String, String>> {
        val output = mutableListOf<Map<String, String>>()
        onnxTensors.withIndex().forEachIndexed { index, onnxTensor ->
            val outputTensor = onnxTensors[index]
            val float2dArray = outputTensor["logits"]?.value as Array<FloatArray>
            val logits = float2dArray[0]
            val softmaxlogits = softmax(logits)
            val labels = listOf("Negative","Positive")
            val predictedIndex = softmaxlogits.indices.maxByOrNull { softmaxlogits[it] } ?: -1
            val predictedLabel = labels[predictedIndex]
            val confidence = softmaxlogits[predictedIndex] * 100
            output.add(mapOf(
                "predicted_label" to predictedLabel,
                "confidence" to "%.2f".format(confidence)
            ))
        }

        return output
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

    fun softmax(logits: FloatArray, temperature: Float = 2.0f): FloatArray {
        val exp = logits.map { Math.exp((it / temperature).toDouble()) }
        val sumExp = exp.sum()
        return exp.map { (it / sumExp).toFloat() }.toFloatArray()
    }
}

