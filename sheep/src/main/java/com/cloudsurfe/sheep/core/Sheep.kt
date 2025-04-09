package com.cloudsurfe.sheep.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.cloudsurfe.sheep.pipeline.Pipeline
import com.cloudsurfe.sheep.pipeline.PipelineType
import com.cloudsurfe.sheep.pipeline.TextClassification
import com.cloudsurfe.sheep.tokenizer.Tokenizer
import com.cloudsurfe.sheep.tokenizer.TokenizerType
import com.cloudsurfe.sheep.tokenizer.WordPiece
import com.cloudsurfe.sheep.util.copyAssetInInternalStorage
import java.nio.LongBuffer


class Sheep(
    private val context: Context,
    private val pipeline: Pipeline,
    private val tokenizer: TokenizerType,
    private val assetModelFilename: String,
    private val assetModelVocabFile: String
) {

    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession
    private lateinit var name: String

    init {
        try {
            env = OrtEnvironment.getEnvironment()
            Log.d(TAG, "Initializing ONNX model...")
            val modelPath: String? = copyAssetInInternalStorage(context, assetModelFilename)
            session = env.createSession(modelPath)
        } catch (e: OrtException) {
            Log.d(TAG, "Error Initializing ONNX model: ${e.message}")
        }

    }

    fun run(pipelineType: PipelineType): Map<Int, String> {

        if (!::session.isInitialized) {
            Log.d(TAG, "Onnx session is not initialized")
            return emptyMap()
        }
        // How can I check if session is initialised here
        Log.d(TAG, "Inside run")
        val resolvedPipeline = when (pipelineType) {
            is PipelineType.CustomPipeline -> pipeline
            is PipelineType.TextSimilarity -> TextClassification()
        }
        val resolvedTokenizer = when (tokenizer) {
            is TokenizerType.CustomTokenizer -> tokenizer.tokenizer
            TokenizerType.WordPiece -> WordPiece(context, assetModelVocabFile)
        }

        when (pipelineType) {
            is PipelineType.CustomPipeline -> {
                return resolvedPipeline.pipeline(
                    getOutputTensor(
                        resolvedTokenizer,
                        *pipelineType.inputs
                    )
                )
            }

            is PipelineType.TextSimilarity -> {
                getOutputTensor(
                    resolvedTokenizer,
                    pipelineType.input1,
                ).forEach { outputTensor ->
                    val float3DArray = outputTensor.value as Array<Array<FloatArray>>
                    Log.d("Sheep", "$float3DArray")
                    Log.d("Sheep", "Batch size: ${float3DArray.size}")
                    Log.d("Sheep", "Sequence length: ${float3DArray[0].size}")
                    Log.d("Sheep", "Hidden size: ${float3DArray[0][0].size}")
                }
                return resolvedPipeline.pipeline(
                    getOutputTensor(
                        resolvedTokenizer,
                        pipelineType.input1,
                    )
                )
            }
        }

    }

    private fun getOutputTensor(
        tokenizer: Tokenizer,
        vararg inputText: String
    ): List<OnnxTensor> {
        val outputs = mutableListOf<OnnxTensor>()
        getInputTensor(
            tokenizer,
            *inputText
        ).forEach { (index, inputTensor) ->
            val inputMap: Map<String, OnnxTensor> = mapOf("input_ids" to inputTensor)
            val result = session.run(inputMap)
            val optionalOutput = result.get("last_hidden_state")
            if (optionalOutput.isPresent) {
                val outputTensor = optionalOutput.get() as OnnxTensor
                outputs.add(outputTensor)
            } else {
                Log.d(TAG, "Warning: No valid output for index $index")
            }
        }
        return outputs
    }

    private fun getInputTensor(
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

    class Builder(
        private val assetModelFileName: String,
        private val assetModelVocabFile: String
    ) {

        private lateinit var context: Context
        private lateinit var pipeline: Pipeline
        private lateinit var tokenizer: TokenizerType

        fun addTokenizer(tokenizer: Tokenizer) = apply {
            this.tokenizer = TokenizerType.CustomTokenizer(tokenizer)
        }

        fun addPipeline(pipeline: Pipeline) = apply {
            this.pipeline = pipeline
        }

        fun build() = Sheep(
            context,
            pipeline,
            tokenizer,
            assetModelFileName,
            assetModelVocabFile
        )
    }

}















