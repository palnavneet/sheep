package com.cloudsurfe.sheep.core

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.cloudsurfe.sheep.pipeline.Pipeline
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

    init {
        try {
            env = OrtEnvironment.getEnvironment()
            Log.d(TAG, "Initializing ONNX model...")
            val modelPath: String? = copyAssetInInternalStorage(context, assetModelFilename)
            val sessionOptions = OrtSession.SessionOptions()
            session = env.createSession(modelPath, sessionOptions)
        } catch (e: OrtException) {
            Log.d(TAG, "Error Initializing ONNX model: ${e.message}")
        }

    }

    fun run(a : String) : List<Map<String, String>>{
        return runInference(a)
    }
    fun run(a : String, b : String) : List<Map<String, String>>{
        return runInference(a,b)
    }

    fun run(a : String,b : String,c : String) : List<Map<String, String>>{
        return runInference(a,b,c)
    }
    fun run(vararg inputs : String) : List<Map<String, String>>{
        return runInference(*inputs)
    }

    private fun runInference(vararg input : String): List<Map<String, String>> {

        if (!::session.isInitialized) {
            Log.d(TAG, "Onnx session is not initialized")
            return emptyList()
        }
        val modelMetaData = detectModelPipeline(session)
        // How can I check if session is initialised here
        val resolvedPipeline = when (modelMetaData.type) {
            ModelType.TEXT_CLASSIFICATION -> {
                TextClassificationFineTuned()
            }
            ModelType.FEATURE_EXTRACTOR -> {
                TextClassification()
            }
            ModelType.UNKNOWN -> pipeline
        }
        val resolvedTokenizer = when (tokenizer) {
            is TokenizerType.CustomTokenizer -> tokenizer.tokenizer
            TokenizerType.WordPiece -> WordPiece(context, assetModelVocabFile)
        }
        when(modelMetaData.type){
            ModelType.TEXT_CLASSIFICATION -> {
                return resolvedPipeline.pipeline(
                    resolvedPipeline.getOutputTensor(
                        session,
                        env,
                        resolvedTokenizer,
                        *input
                    )
                )
            }
            ModelType.FEATURE_EXTRACTOR -> {
                return resolvedPipeline.pipeline(
                    resolvedPipeline.getOutputTensor(
                        session,
                        env,
                        resolvedTokenizer,
                        *input,
                    )
                )
            }
            ModelType.UNKNOWN -> {
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
//        when (pipelineType) {
//            is PipelineType.CustomPipeline -> {
//                return resolvedPipeline.pipeline(
//                    resolvedPipeline.getOutputTensor(
//                        session,
//                        env,
//                        resolvedTokenizer,
//                        *pipelineType.inputs
//                    )
//                )
//            }
//
//            is PipelineType.TextClassification -> {
//                return resolvedPipeline.pipeline(
//                    resolvedPipeline.getOutputTensor(
//                        session,
//                        env,
//                        resolvedTokenizer,
//                        pipelineType.input,
//                    )
//                )
//            }
//
//            is PipelineType.TextClassificationFineTuned -> {
//                return resolvedPipeline.pipeline(
//                    resolvedPipeline.getOutputTensor(
//                        session,
//                        env,
//                        resolvedTokenizer,
//                        pipelineType.input,
//                    )
//                )
//            }
//        }

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















