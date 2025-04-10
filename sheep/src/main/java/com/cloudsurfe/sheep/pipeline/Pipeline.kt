package com.cloudsurfe.sheep.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.cloudsurfe.sheep.tokenizer.Tokenizer

interface Pipeline{
    val numberOfInputs : Int

    fun pipeline(onnxTensors : List<OnnxTensor>) : Map<Int, String>
    fun getOutputTensor(session : OrtSession, env : OrtEnvironment, tokenizer : Tokenizer, vararg inputText : String) : List<OnnxTensor>
    fun getInputTensor(env : OrtEnvironment, tokenizer : Tokenizer, vararg inputText : String) : Map<Int, OnnxTensor>
    fun tokenizer(tokenizer : Tokenizer, vararg inputText : String) : List<LongArray>
    fun shape(env : OrtEnvironment, inputId : LongArray) : OnnxTensor

}