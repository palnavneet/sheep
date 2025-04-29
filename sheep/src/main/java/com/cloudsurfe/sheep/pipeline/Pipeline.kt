package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.cloudsurfe.sheep.tokenizer.Tokenizer

interface Pipeline {

    fun pipeline(onnxTensors: List<Map<String, OnnxTensor>>): List<Map<String, String>>
    fun <T> getOutputTensor(session: OrtSession, env: OrtEnvironment, tokenizer: Tokenizer, vararg input: Pair<String, T>): List<Map<String, OnnxTensor>>

    fun getInputTensor(env: OrtEnvironment, tokenizer: Tokenizer, input: List<String>): List<Map<String, OnnxTensor>>

    fun tokenizer(tokenizer: Tokenizer, input: List<String>): List<LongArray>
    fun shape(env: OrtEnvironment, inputId: LongArray): OnnxTensor

}