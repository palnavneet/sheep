package com.cloudsurfe.sheep

import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import com.cloudsurfe.sheep.core.Sheep
import com.cloudsurfe.sheep.tokenizer.TokenizerType
import com.cloudsurfe.sheep.tokenizer.WordPiece

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val sheep = Sheep(
            this,
            TokenizerType.WordPiece,
            "distilbert_classification_quantized.onnx",
            "vocab.txt"
        )
        if (sheep.isInitialized){
            val label = sheep.run("input" to listOf("Hey How was your day?", "I think we are enemies","I will kill you"))
            val predictedLabel = label[0]["predicted_label"]
            val confidence = label[0]["confidence"]
            Log.d("Sheep", "$predictedLabel")
            Log.d("Sheep", "$confidence")
            val predictedLabel1 = label[1]["predicted_label"]
            val confidence1 = label[1]["confidence"]
            Log.d("Sheep", "$predictedLabel1")
            Log.d("Sheep", "$confidence1")
            val predictedLabel2 = label[2]["predicted_label"]
            val confidence2 = label[2]["confidence"]
            Log.d("Sheep", "$predictedLabel2")
            Log.d("Sheep", "$confidence2")
        }
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