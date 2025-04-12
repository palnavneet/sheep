package com.cloudsurfe.sheep

import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import com.cloudsurfe.sheep.core.Sheep
import com.cloudsurfe.sheep.pipeline.PipelineType
import com.cloudsurfe.sheep.pipeline.TextClassification
import com.cloudsurfe.sheep.pipeline.TextClassificationFineTuned
import com.cloudsurfe.sheep.tokenizer.TokenizerType
import com.cloudsurfe.sheep.tokenizer.WordPiece

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val sheep = Sheep(
            this,
            TokenizerType.WordPiece,
            "distilbert_model_quantized.onnx",
            "vocab.txt"
        )
        val label = sheep.run("We are best friends")
        val predictedLabel = label[0]["predicted_label"]
        val confidence = label[0]["confidence"]
        Log.d("Sheep", "$predictedLabel")
        Log.d("Sheep", "$confidence")
    }
}


fun checkTokenizer(context: Context) {
    val tokenizer = WordPiece(context, "vocab.txt")
    tokenizer.loadVocab()
    val input = "i am using summoning Jutsu"
    val tokenIds = tokenizer.tokenize(input)

    Log.d("TokenizerTest", "Input: $input")
    Log.d("TokenizerTest", "Token IDs: ${tokenIds.joinToString()}")
    val decoded = tokenIds.joinToString(" ") {
        tokenizer.deTokenize(it.toInt())
    }
    Log.d("TokenizerTest", "Decoded: $decoded")
}