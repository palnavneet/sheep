package com.cloudsurfe.sheep.core

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.cloudsurfe.sheep.util.copyAssetInInternalStorage


class Sheep(
    val context: Context,
) : NLP {

    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession

    override fun run(assetFileName: String, pipeline: Pipeline, tokenizerType: TokenizerType) {
        try {
            env = OrtEnvironment.getEnvironment()
            Log.d(TAG, "Initializing ONNX model...")
            val modelPath: String? = copyAssetInInternalStorage(context, assetFileName)
            session = env.createSession(modelPath)
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
        val context: Context
    ) {

        fun build(): NLP = Sheep(context)


    }

}