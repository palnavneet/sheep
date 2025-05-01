package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.cloudsurfe.sheep.model.ModelOutputBundle
import com.cloudsurfe.sheep.model.TokenizationResultSet
import com.cloudsurfe.sheep.tokenizer.Tokenizer
import com.cloudsurfe.sheep.utils.generateAttentionMask
import com.cloudsurfe.sheep.utils.softmax
import java.nio.LongBuffer


class TextClassificationFineTuned() : Pipeline {

    override fun pipeline(modelOutputBundle: ModelOutputBundle): List<Map<String, String>> {
        val output = mutableListOf<Map<String, String>>()
        modelOutputBundle.tensors.forEach { tensorMap ->
            val logits = (tensorMap["logits"]?.value as Array<FloatArray>)[0]
            val softmaxProbabilities = softmax(logits)
            val labels = listOf("Negative", "Positive")
            val predictedIndex =
                softmaxProbabilities.indices.maxByOrNull { softmaxProbabilities[it] } ?: -1
            val predictedLabel = labels[predictedIndex]
            val confidence = softmaxProbabilities[predictedIndex] * 100
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
        val rawInputs = inputMap["input"]
        val requiredInputs: List<String> = when (rawInputs) {
            is String -> listOf(rawInputs)
            is List<*> -> rawInputs.filterIsInstance<String>()
            else -> throw IllegalArgumentException("Unsupported input type for 'inputs'")
        }
        val inputTensorBundle = getInputTensor(env,tokenizer,requiredInputs)
        val outputs = mutableListOf<Map<String, OnnxTensor>>()
        inputTensorBundle.tensors.forEach { inputTensorWithMask ->
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
        val tokenizationResultSet: TokenizationResultSet = tokenizer(tokenizer, input)
        val tensors: List<Map<String, OnnxTensor>> = tokenizationResultSet.tokenizedInput.map { inputId ->
            val attention_mask = generateAttentionMask(inputId)
            mapOf(
                "input_Ids" to shape(env,inputId),
                "attention_mask" to shape(env,attention_mask)
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

    override fun shape(
        env: OrtEnvironment,
        inputId: LongArray
    ): OnnxTensor {
        val shape: LongArray = longArrayOf(1, inputId.size.toLong())
        return OnnxTensor.createTensor(env, LongBuffer.wrap(inputId), shape)
    }

}

