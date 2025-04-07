package com.cloudsurfe.sheep.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.cloudsurfe.sheep.pipeline.Pipeline
import com.cloudsurfe.sheep.pipeline.PipelineType
import com.cloudsurfe.sheep.pipeline.TextSimilarity
import com.cloudsurfe.sheep.tokenizer.Tokenizer
import com.cloudsurfe.sheep.tokenizer.TokenizerType
import com.cloudsurfe.sheep.tokenizer.WordPiece
import com.cloudsurfe.sheep.util.copyAssetInInternalStorage
import java.nio.LongBuffer


class Sheep(
    private val context: Context,
    private val pipeline: PipelineType,
    private val tokenizer: TokenizerType
) {

    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession

    fun run(assetModelFileName: String, assetVocabFileName: String) {
        try {
            env = OrtEnvironment.getEnvironment()
            Log.d(TAG, "Initializing ONNX model...")
            val modelPath: String? = copyAssetInInternalStorage(context, assetModelFileName)
            session = env.createSession(modelPath)
            val resolvedPipeline = when (pipeline) {
                is PipelineType.CustomPipeline -> pipeline.pipeline
                is PipelineType.TextSimilarity -> TextSimilarity()
            }
            val resolvedTokenizer = when (tokenizer) {
                is TokenizerType.CustomTokenizer -> tokenizer.tokenizer
                TokenizerType.WordPiece -> WordPiece(context, assetModelFileName)
            }

            val adjustablePipeline = when(pipeline){
                is PipelineType.CustomPipeline -> TODO()
                is PipelineType.TextSimilarity -> {
                    runModelInference(
                        resolvedTokenizer,
                        pipeline.input1,
                        pipeline.input2
                    ).forEach { outputTensor ->
                        val float3DArray = outputTensor.value as Array<Array<FloatArray>>
                        Log.d("Sheep", "$float3DArray")
                        Log.d("Sheep", "Batch size: ${float3DArray.size}")
                        Log.d("Sheep", "Sequence length: ${float3DArray[0].size}")
                        Log.d("Sheep", "Hidden size: ${float3DArray[0][0].size}")
                    }
                }
            }

        } catch (e: OrtException) {
            Log.d(TAG, "Error Initializing ONNX model: ${e.message}")
        }
    }

    private fun runModelInference(
        tokenizer: Tokenizer,
        vararg inputText: String
    ): List<OnnxTensor> {
        val outputs = mutableListOf<OnnxTensor>()
        val inputTensor = inputTensor(
            tokenizer,
            *inputText
        ).forEach {(index , inputTensor) ->
            val inputMap : Map<String, OnnxTensor> = mapOf("input_ids" to inputTensor)
            val result = session.run(inputMap)
            val optionalOutput = result.get("last_hidden_state")
            if (optionalOutput.isPresent){
                val outputTensor = optionalOutput.get() as OnnxTensor
                outputs.add(outputTensor)
            }else{
                Log.d(TAG, "Warning: No valid output for index $index")
            }
        }
        return outputs
    }

    private fun inputTensor(
        tokenizer: Tokenizer,
        vararg inputText: String
    ): Map<Int, OnnxTensor> {
        return tokenizer(
            tokenizer,
            *inputText
        ).withIndex().associate { (index, tokenizedInput) ->
            index to shape(tokenizedInput)
        }
    }

    private fun tokenizer(
        tokenizer: Tokenizer,
        vararg inputText: String,
    ): List<LongArray> {
        tokenizer.apply {
            loadVocab()
        }
        return inputText.map { text -> tokenizer.tokenize(text) }
    }

    private fun shape(inputId: LongArray): OnnxTensor {
        val shape: LongArray = longArrayOf(1, inputId.size.toLong())
        return OnnxTensor.createTensor(env, LongBuffer.wrap(inputId), shape)
    }

    companion object {
        const val TAG: String = "Sheep"
    }

    class Builder() {

        private lateinit var context: Context
        private lateinit var pipeline: PipelineType
        private lateinit var tokenizer: TokenizerType

        fun addTokenizer(tokenizer: Tokenizer) = apply {
            this.tokenizer = TokenizerType.CustomTokenizer(tokenizer)
        }

        fun addPipeline(pipeline: Pipeline) = apply {
            this.pipeline = PipelineType.CustomPipeline(pipeline)
        }

        fun build() = Sheep(
            context,
            pipeline,
            tokenizer
        )
    }

}















