package com.example.reconhecimento_lgp

import android.content.Context
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.tensorflow.lite.Interpreter
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.Locale
import kotlin.math.sqrt

class GestureRecognizerHelper(
    private val context: Context,
    private val gestureRecognizerListener: GestureRecognizerListener
) {

    private lateinit var handLandmarker: HandLandmarker
    private lateinit var tflite: Interpreter
    private lateinit var labels: List<String>

    private val executor = Executors.newSingleThreadScheduledExecutor()
    private val landmarkBuffer = mutableListOf<List<Float>>()
    private val sequenceLength = 20 // Same as in training

    init {
        try {
            setupHandLandmarker()
            setupTFLite()
        } catch (e: Exception) {
            gestureRecognizerListener.onError("Initialization failed: ${e.message}")
            throw e
        }
    }

    private fun setupHandLandmarker() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setNumHands(2) // Allow detection of both hands
                .setMinHandDetectionConfidence(0.3f) // Lower threshold for easier detection
                .setMinHandPresenceConfidence(0.3f) // Lower threshold for hand presence
                .setMinTrackingConfidence(0.3f) // Lower threshold for tracking
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener(this::onHandLandmarkerResult)
                .setErrorListener { error ->
                    gestureRecognizerListener.onError(error.message ?: "Unknown error")
                }
                .build()
            handLandmarker = HandLandmarker.createFromOptions(context, options)
            android.util.Log.d("GestureRecognizer", "HandLandmarker initialized successfully with relaxed confidence thresholds")
        } catch (e: Exception) {
            gestureRecognizerListener.onError("Failed to initialize hand landmarker: ${e.message}")
        }
    }

    private fun setupTFLite() {
        try {
            val model = loadModelFile()
            val options = Interpreter.Options()
            // Use CPU-only execution
            android.util.Log.d("GestureRecognizer", "Using CPU-only TensorFlow Lite execution")
            tflite = Interpreter(model, options)
            labels = loadLabels()
        } catch (e: Exception) {
            gestureRecognizerListener.onError("TensorFlow Lite initialization failed: ${e.message}")
            throw e
        }
    }

    private fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd("modelo_gestos_lgp.tflite")
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabels(): List<String> {
        val inputStream = context.assets.open("classes.json")
        val jsonString = inputStream.bufferedReader().use { it.readText() }
        // Parse JSON using Gson for better performance
        val gson = Gson()
        val type = object : TypeToken<Map<String, Int>>() {}.type
        val map: Map<String, Int> = gson.fromJson(jsonString, type)
        val sortedLabels = Array(map.size) { "" }
        map.forEach { (key, index) ->
            sortedLabels[index] = key
        }
        return sortedLabels.toList()
    }

    fun recognizeLiveStream(image: MPImage) {
        handLandmarker.detectAsync(image, System.currentTimeMillis())
    }

    private fun onHandLandmarkerResult(result: HandLandmarkerResult, @Suppress("UNUSED_PARAMETER") image: MPImage) {
        gestureRecognizerListener.onResults(result)

        if (result.landmarks().isNotEmpty()) {
            // Hand detected - add to buffer for dynamic gesture
            result.landmarks().firstOrNull()?.let { landmarks ->
                val landmarkData = landmarks.flatMap { landmark ->
                    listOf(landmark.x(), landmark.y(), landmark.z())
                }
                landmarkBuffer.add(landmarkData)
                android.util.Log.d("GestureRecognizer", "Buffer size: ${landmarkBuffer.size}/$sequenceLength (collecting movement)")

                if (landmarkBuffer.size == sequenceLength) {
                    // Validate that the sequence has enough movement (not just static hand)
                    if (hasSignificantMovement()) {
                        android.util.Log.d("GestureRecognizer", "Running inference on dynamic gesture sequence...")
                        runInference()
                    } else {
                        android.util.Log.d("GestureRecognizer", "Sequence too static - skipping inference")
                    }
                    // Keep sliding window - remove oldest, keep newest for continuous recognition
                    landmarkBuffer.removeAt(0)
                }
            }
        } else {
            // No hand detected - keep buffer for short gaps, clear if too long
            if (landmarkBuffer.size > sequenceLength / 2) {
                android.util.Log.d("GestureRecognizer", "No hand detected, keeping buffer for continuity")
            } else {
                landmarkBuffer.clear()
                android.util.Log.d("GestureRecognizer", "No hand detected, buffer cleared")
            }
        }
    }

    private fun hasSignificantMovement(): Boolean {
        if (landmarkBuffer.size < 2) return false
        
        // Calculate movement variance across the sequence
        var totalMovement = 0.0
        val numPoints = 21 // 21 hand landmarks
        
        for (i in 1 until landmarkBuffer.size) {
            val prev = landmarkBuffer[i - 1]
            val curr = landmarkBuffer[i]
            
            // Calculate movement for each landmark point (x, y coordinates)
            for (j in 0 until numPoints) {
                val x1 = prev[j * 3]     // x coordinate
                val y1 = prev[j * 3 + 1] // y coordinate
                val x2 = curr[j * 3]     // x coordinate
                val y2 = curr[j * 3 + 1] // y coordinate
                
                // Euclidean distance moved
                val movement = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
                totalMovement += movement
            }
        }
        
        val averageMovement = totalMovement / (landmarkBuffer.size - 1) / numPoints
        val hasMovement = averageMovement > 0.01 // Threshold for significant movement
        
        android.util.Log.d("GestureRecognizer", "Average movement: $averageMovement, Has significant movement: $hasMovement")
        return hasMovement
    }

    private fun runInference() {
        val input = ByteBuffer.allocateDirect(1 * sequenceLength * 63 * 4)
        input.order(ByteOrder.nativeOrder())
        for (sequence in landmarkBuffer) {
            for (value in sequence) {
                input.putFloat(value)
            }
        }
        input.rewind()

        val output = Array(1) { FloatArray(labels.size) }
        tflite.run(input, output)

        val maxIndex = output[0].indices.maxByOrNull { output[0][it] } ?: -1
        if (maxIndex != -1) {
            val confidence = output[0][maxIndex]
            val gesture = labels[maxIndex]
            
            // Log all gesture confidences for debugging
            val allConfidences = labels.mapIndexed { index, label -> 
                "$label: ${String.format(Locale.getDefault(), "%.3f", output[0][index])}"
            }.joinToString(", ")
            android.util.Log.d("GestureRecognizer", "All confidences: [$allConfidences]")
            
            // Calculate confidence margin (difference between top 2 predictions)
            val sortedConfidences = output[0].sortedDescending()
            val confidenceMargin = if (sortedConfidences.size >= 2) {
                sortedConfidences[0] - sortedConfidences[1]
            } else {
                confidence
            }
            
            android.util.Log.d("GestureRecognizer", "Top gesture: $gesture, Confidence: $confidence, Margin: $confidenceMargin")
            
            // Stricter validation for gesture recognition
            if (confidence > 0.8f && confidenceMargin > 0.3f) { // Much higher thresholds
                gestureRecognizerListener.onGestureRecognized(gesture)
                android.util.Log.d("GestureRecognizer", "✅ ACCEPTED: $gesture (conf: $confidence, margin: $confidenceMargin)")
            } else {
                android.util.Log.d("GestureRecognizer", "❌ REJECTED: $gesture (conf: $confidence, margin: $confidenceMargin) - Too low confidence or margin")
            }
        }
    }

    fun close() {
        try {
            executor.shutdown()
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS)
            if (this::handLandmarker.isInitialized) {
                handLandmarker.close()
            }
            if (this::tflite.isInitialized) {
                tflite.close()
            }
        } catch (e: Exception) {
            gestureRecognizerListener.onError("Error closing resources: ${e.message}")
        }
    }

    interface GestureRecognizerListener {
        fun onError(error: String)
        fun onResults(result: HandLandmarkerResult)
        fun onGestureRecognized(gesture: String)
    }
}
