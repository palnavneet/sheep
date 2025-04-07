package com.cloudsurfe.sheep.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.cloudsurfe.sheep.pipeline.PipelineType
import com.cloudsurfe.sheep.pipeline.TextSimilarity
import com.cloudsurfe.sheep.tokenizer.TokenizerType
import com.cloudsurfe.sheep.tokenizer.WordPiece
import com.cloudsurfe.sheep.util.copyAssetInInternalStorage


class Sheep(
    val context: Context,
    val pipeline: PipelineType,
    val tokenizer: TokenizerType
) {

    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession

    fun <T> run(assetModelFileName: String, assetVocabFileName: String) : T{
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

        } catch (e: OrtException) {
            Log.d(TAG, "Error Initializing ONNX model: ${e.message}")
        }
    }

    companion object {
        const val TAG: String = "Sheep"
        fun load() {

        }
    }

    class Builder(
        val context: Context,
        val pipeline: PipelineType,
        val tokenizer: TokenizerType
    ) {

        fun build() = Sheep(
            context,
            pipeline,
            tokenizer
        )
    }

    private fun inputTensor(
        context: Context,
        tokenizer: TokenizerType,
        assetVocabFileName: String,
        vararg input: String
    ): OnnxTensor {
        val tokenizer = when (tokenizer) {
            is TokenizerType.CustomTokenizer -> {
                tokenizer.tokenizer
            }

            is TokenizerType.WordPiece -> {
                WordPiece(context, assetVocabFileName)
            }
        }
        val inputIds: LongArray = tokenizer.tokenize(input)
    }


}