package com.cloudsurfe.sheep.core

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.cloudsurfe.sheep.pipeline.Pipeline
import com.cloudsurfe.sheep.pipeline.PipelineType
import com.cloudsurfe.sheep.pipeline.TextClassification
import com.cloudsurfe.sheep.pipeline.TextClassificationFineTuned
import com.cloudsurfe.sheep.tokenizer.Tokenizer
import com.cloudsurfe.sheep.tokenizer.TokenizerType
import com.cloudsurfe.sheep.tokenizer.WordPiece
import com.cloudsurfe.sheep.util.copyAssetInInternalStorage


class Sheep(
    private val context: Context,
    private val tokenizer: TokenizerType,
    private val assetModelFilename: String,
    private val assetModelVocabFile: String
) {

    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession
    private lateinit var pipeline : Pipeline
    var isInitialized : Boolean = false

    init {
        try {
            env = OrtEnvironment.getEnvironment()
            Log.d(TAG, "Initializing ONNX model...")
            val modelPath: String? = copyAssetInInternalStorage(context, assetModelFilename)
            val sessionOptions = OrtSession.SessionOptions()
            session = env.createSession(modelPath, sessionOptions)
            isInitialized = true
        } catch (e: OrtException) {
            Log.d(TAG, "Error Initializing ONNX model: ${e.message}")
        }

    }


    fun <T>run(vararg input : Pair<String, T>): List<Map<String, String>> {

        if (!::session.isInitialized) {
            Log.d(TAG, "Onnx session is not initialized")
            return emptyList()
        }
        val modelMetaData = detectModelPipeline(session)
        // How can I check if session is initialised here
        val resolvedPipeline = when (modelMetaData.type) {
            PipelineType.TEXT_CLASSIFICATION -> {
                TextClassificationFineTuned()
            }
            PipelineType.FEATURE_EXTRACTOR -> {
                TextClassification()
            }
            PipelineType.UNKNOWN -> pipeline
        }
        val resolvedTokenizer = when (tokenizer) {
            is TokenizerType.CustomTokenizer -> tokenizer.tokenizer
            TokenizerType.WordPiece -> WordPiece(context, assetModelVocabFile)
        }
        when(modelMetaData.type){
            PipelineType.TEXT_CLASSIFICATION -> {
                return resolvedPipeline.pipeline(
                    resolvedPipeline.getOutputTensor(
                        session,
                        env,
                        resolvedTokenizer,
                        *input
                    )
                )
            }
            PipelineType.FEATURE_EXTRACTOR -> {
                return resolvedPipeline.pipeline(
                    resolvedPipeline.getOutputTensor(
                        session,
                        env,
                        resolvedTokenizer,
                        *input,
                    )
                )
            }
            PipelineType.UNKNOWN -> {
                return resolvedPipeline.pipeline(
                    resolvedPipeline.getOutputTensor(
                        session,
                        env,
                        resolvedTokenizer,
                        *input,
                    )
                )
            }
        }
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
            tokenizer,
            assetModelFileName,
            assetModelVocabFile
        ).apply {
            this.pipeline = this@Builder.pipeline
        }
    }

}















