package com.cloudsurfe.sheep.util

import android.content.Context
import android.util.Log
import com.cloudsurfe.sheep.core.Sheep.Companion.TAG
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

fun copyAssetInInternalStorage(
    context: Context,
    assetFileName: String,
): String? {
    val file = File(context.filesDir, assetFileName)
    try {
        if (!file.exists()) {
            context.assets.open(assetFileName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        return file.absolutePath
    } catch (e: IOException) {
        Log.d(TAG, "Failed to copy $assetFileName : ${e.message}")
        return null
    }
}