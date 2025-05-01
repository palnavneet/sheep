package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.cloudsurfe.sheep.model.ModelOutputBundle
import com.cloudsurfe.sheep.model.TokenizationResultSet
import com.cloudsurfe.sheep.tokenizer.Tokenizer

interface Pipeline {

    fun pipeline(modelOutputBundle: ModelOutputBundle): List<Map<String, String>>
    fun <T> getOutputTensor(session: OrtSession, env: OrtEnvironment, tokenizer: Tokenizer, vararg input: Pair<String, T>): ModelOutputBundle

    fun getInputTensor(env: OrtEnvironment, tokenizer: Tokenizer, input: List<String>): ModelOutputBundle

    fun tokenizer(tokenizer: Tokenizer, input: List<String>): TokenizationResultSet
    fun shape(env: OrtEnvironment, inputId: LongArray): OnnxTensor

}