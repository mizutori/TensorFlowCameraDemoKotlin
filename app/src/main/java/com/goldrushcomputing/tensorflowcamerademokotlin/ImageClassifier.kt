package com.goldrushcomputing.tensorflowcamerademokotlin

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.collections.Map.Entry


class ImageClassifier (context: Context) {

    /** Name of the model file stored in Assets.  */
    private val MODEL_PATH = "graph.lite"

    /** Name of the label file stored in Assets.  */
    private val LABEL_PATH = "labels.txt"

    /** Number of results to show in the UI.  */
    private val RESULTS_TO_SHOW = 3

    /** Dimensions of inputs.  */
    private val DIM_BATCH_SIZE = 1

    private val DIM_PIXEL_SIZE = 3

    private val IMAGE_MEAN = 128
    private val IMAGE_STD = 128.0f


    /* Preallocated buffers for storing image data in. */
    private val intValues = IntArray(DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y)

    /** An instance of the driver class to run model inference with Tensorflow Lite.  */
    private var tflite: Interpreter? = null

    /** Labels corresponding to the output of the vision model.  */
    private var labelList: List<String>

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.  */
    private var imgData: ByteBuffer

    /** An array to hold inference results, to be feed into Tensorflow Lite as outputs.  */
    private var labelProbArray: Array<FloatArray>
    /** multi-stage low pass filter  */
    private var filterLabelProbArray: Array<FloatArray>
    private val FILTER_STAGES = 3
    private val FILTER_FACTOR = 0.4f

    private val sortedLabels = PriorityQueue<Entry<String, Float>>(
        RESULTS_TO_SHOW,
        Comparator<Entry<String, Float>> { o1, o2 -> o1.value.compareTo(o2.value) })


    init {
        loadModelFile(context)?.let{mappedByteBuffer ->
            val byteBuffer = mappedByteBuffer as ByteBuffer
            tflite = Interpreter(byteBuffer, null)
        }

        labelList = loadLabelList(context)
        imgData = ByteBuffer.allocateDirect(
            4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE
        )
        imgData.order(ByteOrder.nativeOrder())
        labelProbArray = Array(1) { FloatArray(labelList.size) }
        filterLabelProbArray = Array(FILTER_STAGES) { FloatArray(labelList.size) }
        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.")
    }


    /** Classifies a frame from the preview stream.  */
    fun classifyFrame(bitmap: Bitmap): String {
        tflite?.let{
            convertBitmapToByteBuffer(bitmap)
            // Here's where the magic happens!!!
            val startTime = SystemClock.uptimeMillis()
            it.run(imgData, labelProbArray)
            val endTime = SystemClock.uptimeMillis()
            Log.d(TAG, "Timecost to run model inference: " + (endTime - startTime).toString())
            // smooth the results
            applyFilter()
            // print the results
            var textToShow = printTopKLabels()
            textToShow = (endTime - startTime).toString() + "ms" + textToShow
            return textToShow
        } ?: run{
            Log.e(TAG, "Image classifier has not been initialized; Skipped.")
            return "Uninitialized Classifier."
        }

    }

    private fun applyFilter() {
        val num_labels = labelList.size

        // Low pass filter `labelProbArray` into the first stage of the filter.
        for (j in 0 until num_labels) {
            filterLabelProbArray[0][j] += FILTER_FACTOR * (labelProbArray[0][j] - filterLabelProbArray[0][j])
        }
        // Low pass filter each stage into the next.
        for (i in 1 until FILTER_STAGES) {
            for (j in 0 until num_labels) {
                filterLabelProbArray[i][j] += FILTER_FACTOR * (filterLabelProbArray[i - 1][j] - filterLabelProbArray[i][j])
            }
        }

        // Copy the last stage filter output back to `labelProbArray`.
        for (j in 0 until num_labels) {
            labelProbArray[0][j] = filterLabelProbArray[FILTER_STAGES - 1][j]
        }
    }

    /** Closes tflite to release resources.  */
    fun close() {
        tflite?.close()
        tflite = null
    }

    /** Reads label list from Assets.  */
    //@Throws(IOException::class)
    private fun loadLabelList(context: Context): List<String> {
        val labelList = ArrayList<String>()
        val reader = BufferedReader(InputStreamReader(context.assets.open(LABEL_PATH)))
        BufferedReader(reader).use { r ->
            r.lineSequence().forEach {line->
                labelList.add(line)
            }
        }
        reader.close()
        return labelList
    }

    /** Memory-map the model file in Assets.  */
    private fun loadModelFile(context: Context): MappedByteBuffer? {
        return try {
            val fileDescriptor = context.assets.openFd(MODEL_PATH)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        } catch (ioe: IOException) {
            ioe.printStackTrace()
            null
        }
    }

    /*
    private fun loadModelFileToByteBuffer(context: Context): ByteBuffer? {
        return try {
            val inputStream = context.assets.open(MODEL_PATH)
            val fileBytes = ByteArray(inputStream.available())
            inputStream.read(fileBytes)
            inputStream.close()
            ByteBuffer.wrap(fileBytes)

        } catch (ioe: IOException) {
            ioe.printStackTrace()
            null
        }
    }
    */

    /** Writes Image data into a `ByteBuffer`.  */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        imgData.rewind()
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        // Convert the image to floating point.
        var pixel = 0
        val startTime = SystemClock.uptimeMillis()
        for (i in 0 until DIM_IMG_SIZE_X) {
            for (j in 0 until DIM_IMG_SIZE_Y) {
                val `val` = intValues[pixel++]
                imgData.putFloat(((`val` shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                imgData.putFloat(((`val` shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                imgData.putFloat(((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }
        val endTime = SystemClock.uptimeMillis()
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + (endTime - startTime).toString())
    }

    /** Prints top-K labels, to be shown in UI as the results.  */
    private fun printTopKLabels(): String {
        for (i in labelList.indices) {
            sortedLabels.add(
                AbstractMap.SimpleEntry(labelList[i], labelProbArray[0][i])
            )
            if (sortedLabels.size > RESULTS_TO_SHOW) {
                sortedLabels.poll()
            }
        }
        var textToShow = ""
        val size = sortedLabels.size
        for (i in 0 until size) {
            val label = sortedLabels.poll()
            textToShow = String.format("\n%s: %4.2f", label.key, label.value) + textToShow
        }
        return textToShow
    }



    companion object {
        const val TAG = "ImageClassifier"
        internal const val DIM_IMG_SIZE_X = 224
        internal const val DIM_IMG_SIZE_Y = 224

    }

}