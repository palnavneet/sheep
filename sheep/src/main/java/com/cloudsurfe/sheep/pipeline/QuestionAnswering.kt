package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.cloudsurfe.sheep.pipeline.utlis.generateAttentionMask
import com.cloudsurfe.sheep.tokenizer.Tokenizer
import java.nio.LongBuffer

class QuestionAnswering() : Pipeline{

    override fun pipeline(onnxTensors: List<Map<String, OnnxTensor>>): List<Map<String, String>> {
        val output = mutableListOf<Map<String, String>>()
        onnxTensors.withIndex().forEachIndexed{ index, tensorMap ->

        }
        TODO("Not yet implemented")
    }

    override fun <T> getOutputTensor(
        session: OrtSession,
        env: OrtEnvironment,
        tokenizer: Tokenizer,
        vararg input: Pair<String, T>
    ): List<Map<String, OnnxTensor>> {
        val inputMap = mapOf(*input)
        val rawInputs = inputMap["input"]
        val requiredInputs : List<String> = when(rawInputs){
            is String -> listOf(rawInputs)
            is List<*> -> rawInputs.filterIsInstance<String>()
            else -> throw IllegalArgumentException("Unsupported input type for 'inputs'")
        }
        val outputs = mutableListOf<Map<String, OnnxTensor>>()
        getInputTensor(
            env,
            tokenizer,
            requiredInputs
        ).forEach {inputTensorWithMask ->
            val inputIdsInputTensor = inputTensorWithMask["input_Ids"]
            val attentionMaskInputTensor = inputTensorWithMask["attention_mask"]

            if (inputIdsInputTensor != null && attentionMaskInputTensor != null){

                val inputMap : Map<String, OnnxTensor> = mapOf(
                    "input_ids" to inputIdsInputTensor,
                    "attention_mask" to attentionMaskInputTensor
                )

                val result = session.run(inputMap)
                val startLogitsTensor = result.get("start_logits")
                val endLogitsTensor = result.get("end_logits")

                if (startLogitsTensor.isPresent && startLogitsTensor.get() is OnnxTensor && endLogitsTensor.isPresent && endLogitsTensor.get() is OnnxTensor){
                    val startTensor = startLogitsTensor.get() as OnnxTensor
                    val endTensor = endLogitsTensor.get() as OnnxTensor

                    outputs.add(
                        mapOf(
                            "start_logits" to startTensor,
                            "end_logits" to endTensor
                        )
                    )
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
        ).map { input_id ->
            val attention_mask = generateAttentionMask(input_id)
            val inputMap = mutableMapOf<String, OnnxTensor>()
            inputMap["input_Ids"] = shape(env,input_id)
            inputMap["attention_mask"] = shape(env,attention_mask)
            inputMap
        }
    }

    override fun tokenizer(
        tokenizer: Tokenizer,
        input: List<String>
    ): List<LongArray> {
        tokenizer.apply {
            loadVocab()
        }
        return input.map { text -> tokenizer.tokenize(text) }
    }

    override fun shape(
        env: OrtEnvironment,
        inputId: LongArray
    ): OnnxTensor {
        val shape : LongArray = longArrayOf(1,inputId.size.toLong())
        return OnnxTensor.createTensor(env, LongBuffer.wrap(inputId),shape)
    }


}