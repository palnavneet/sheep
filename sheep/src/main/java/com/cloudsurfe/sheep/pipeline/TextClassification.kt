package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import com.cloudsurfe.sheep.tokenizer.Tokenizer
import java.nio.LongBuffer
import kotlin.math.exp

class TextClassification(
) : Pipeline{

    override val numberOfInputs: Int = 1
    override fun pipeline(onnxTensors: List<Map<String,OnnxTensor>>): List<Map<String, String>> {
        val output = mutableListOf<Map<String, String>>()
        onnxTensors.withIndex().forEachIndexed {index,outputTensor ->
            val outputTensor = onnxTensors[index]
            val float3dArray = outputTensor["last_hidden_state"]?.value as Array<Array<FloatArray>>
            val clsEmbedding : FloatArray = float3dArray[0][0]
            val weights = FloatArray(768){0.01f}
            val bias = 0.0f
            val confidence = classify(clsEmbedding,weights,bias)
            val predictedLabel = if (confidence > 0.5f) "positive" else "negative"
            output.add(mapOf(
                "predicted_label" to predictedLabel,
                "confidence" to "%.2f".format(confidence)
            ))
        }

        return output
    }

    override fun getOutputTensor(
        session : OrtSession,
        env : OrtEnvironment,
        tokenizer: Tokenizer,
        vararg inputText: String
    ): List<Map<String,OnnxTensor>> {
        val outputs = mutableListOf<Map<String,OnnxTensor>>()
        getInputTensor(
            env,
            tokenizer,
            *inputText
        ).forEach { inputTensorWithMask ->
            val inputIdsInputTensor = inputTensorWithMask["input_Ids"]
            if (inputIdsInputTensor != null ) {

                val inputMap: Map<String, OnnxTensor> = mapOf(
                    "input_ids" to inputIdsInputTensor
                )

                val result = session.run(inputMap)
                val optionalOutput = result.get("last_hidden_state")
                if (optionalOutput.isPresent && optionalOutput.get() is OnnxTensor){
                    val outputTensor = optionalOutput.get() as OnnxTensor
                    outputs.add(mapOf("last_hidden_state" to outputTensor))

                }
            }
        }
        return outputs
    }

    override fun getInputTensor(
        env : OrtEnvironment,
        tokenizer: Tokenizer,
        vararg inputText: String
    ): List<Map<String, OnnxTensor>> {
        return tokenizer(
            tokenizer,
            *inputText
        ).map { input_Id ->
            val inputMap = mutableMapOf<String, OnnxTensor>()
            inputMap["input_Ids"] = shape(env, input_Id)
            inputMap
        }
    }

    override fun tokenizer(
        tokenizer: Tokenizer,
        vararg inputText: String
    ): List<LongArray> {
        tokenizer.apply {
            loadVocab()
        }
        return inputText.map { text -> tokenizer.tokenize(text) }
    }

    override fun shape(env: OrtEnvironment, inputId: LongArray): OnnxTensor {
        val shape: LongArray = longArrayOf(1, inputId.size.toLong())
        return OnnxTensor.createTensor(env, LongBuffer.wrap(inputId), shape)
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