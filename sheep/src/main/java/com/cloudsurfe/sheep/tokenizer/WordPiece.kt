package com.cloudsurfe.sheep.tokenizer

import android.content.Context
import android.util.Log
import com.cloudsurfe.sheep.core.Sheep.Companion.TAG
import java.io.BufferedReader
import java.io.InputStreamReader

class WordPiece(
    private val context: Context,
    private val assetVocabFileName: String
) : Tokenizer {

    private val idToToken: MutableMap<Int, String> = HashMap()
    private val tokenToId: MutableMap<String, Int> = HashMap()
    private val whitespaceRegex = "\\s+".toRegex()
    private var isVocabLoaded = false

    override fun loadVocab() {
        if (!isVocabLoaded) {
            try {
                context.assets.open(assetVocabFileName).use { inputStream ->
                    BufferedReader(InputStreamReader(inputStream)).use { reader ->
                        reader.lineSequence().forEachIndexed { index, line ->
                            idToToken[index] = line
                            tokenToId[line] = index
                        }
                    }
                }
                isVocabLoaded = true
            } catch (e: Exception) {
                Log.d(TAG, "Failed to read $assetVocabFileName : ${e.message}")
            }
        }
    }

    override fun tokenize(inputText: String): LongArray {
        if (!isVocabLoaded) loadVocab()
        val unkId = tokenToId.getOrDefault("[UNK]", 100)
        val clsId = tokenToId.getOrDefault("[CLS]", 101)
        val sepId = tokenToId.getOrDefault("[SEP]", 102)

        val words = inputText.trim().lowercase().split(whitespaceRegex)
        val inputIds = LongArray(words.size + 2)
        inputIds[0] = clsId.toLong()
        words.forEachIndexed { index, word ->
            inputIds[index + 1] = tokenToId.getOrDefault(word, unkId).toLong()
        }
        inputIds[inputIds.size - 1] = sepId.toLong()
        return inputIds
    }

    override fun deTokenize(tokenId: Int) = idToToken.getOrDefault(tokenId, "[UNK]")

}