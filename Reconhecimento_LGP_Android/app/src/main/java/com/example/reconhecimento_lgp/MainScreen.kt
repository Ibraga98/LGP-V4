package com.example.reconhecimento_lgp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import java.io.ByteArrayOutputStream
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

@Composable
fun MainScreen(viewModel: MainViewModel, gestureRecognizerHelper: GestureRecognizerHelper) {
    val recognizedGesture by viewModel.recognizedGesture.collectAsStateWithLifecycle()
    val handLandmarkerResult by viewModel.handLandmarkerResult.collectAsStateWithLifecycle()
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = {
                val previewView = PreviewView(it)
                val cameraProviderFuture = ProcessCameraProvider.getInstance(it)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                    val imageAnalyzer = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build()
                        .also {
                            it.setAnalyzer(cameraExecutor) { imageProxy ->
                                try {
                                    val mpImage = imageProxy.toMPImage()
                                    if (mpImage != null) {
                                        gestureRecognizerHelper.recognizeLiveStream(mpImage)
                                    }
                                } catch (e: Exception) {
                                    android.util.Log.e("MainScreen", "Error processing image", e)
                                } finally {
                                    imageProxy.close()
                                }
                            }
                        }
                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_FRONT_CAMERA,
                            preview,
                            imageAnalyzer
                        )
                    } catch (exc: Exception) {
                        android.util.Log.e("MainScreen", "Camera binding failed", exc)
                    }
                }, ContextCompat.getMainExecutor(it))
                previewView
            },
            modifier = Modifier.fillMaxSize()
        )
        Overlay(handLandmarkerResult)
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp)
        ) {
            Text(
                text = recognizedGesture ?: "...",
                fontSize = 24.sp,
                color = Color.White
            )
        }
    }
}

@Composable
fun Overlay(result: HandLandmarkerResult?) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val nativeCanvas = drawContext.canvas.nativeCanvas
        val textPaint = android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 50f
            textAlign = android.graphics.Paint.Align.CENTER
        }

        if (result == null || result.landmarks().isEmpty()) {
            nativeCanvas.drawText("No hand detected", size.width / 2, size.height / 2, textPaint)
        } else {
            result.landmarks().forEach { landmarks ->
                HandLandmarker.HAND_CONNECTIONS.forEach {
                    val start = landmarks[it.start()]
                    val end = landmarks[it.end()]
                    drawLine(
                        color = Color.Red,
                        start = androidx.compose.ui.geometry.Offset(start.x() * size.width, start.y() * size.height),
                        end = androidx.compose.ui.geometry.Offset(end.x() * size.width, end.y() * size.height),
                        strokeWidth = 3f
                    )
                }
                val minX = landmarks.minOfOrNull { it.x() } ?: 0f
                val maxX = landmarks.maxOfOrNull { it.x() } ?: 1f
                val minY = landmarks.minOfOrNull { it.y() } ?: 0f
                val maxY = landmarks.maxOfOrNull { it.y() } ?: 1f

                val padding = 0.05f
                val left = max(0f, minX - padding) * size.width
                val top = max(0f, minY - padding) * size.height
                val right = min(1f, maxX + padding) * size.width
                val bottom = min(1f, maxY + padding) * size.height

                drawRect(
                    color = Color.Green,
                    topLeft = androidx.compose.ui.geometry.Offset(left, top),
                    size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                    style = Stroke(width = 8f)
                )
            }
        }
    }
}


private fun ImageProxy.toMPImage(): MPImage? {
    val image = this.image ?: return null
    var bitmap = image.toBitmap()

    // Rotate the bitmap if needed
    val rotation = this.imageInfo.rotationDegrees
    if (rotation != 0) {
        val matrix = Matrix()
        matrix.postRotate(rotation.toFloat())
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    return BitmapImageBuilder(bitmap).build()
}

private fun android.media.Image.toBitmap(): Bitmap {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    //U and V are swapped
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}


