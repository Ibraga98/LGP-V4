package com.example.reconhecimento_lgp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.example.reconhecimento_lgp.ui.theme.Reconhecimento_LGPTheme
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

class MainActivity : ComponentActivity(), GestureRecognizerHelper.GestureRecognizerListener {

    private val hasCameraPermission = mutableStateOf(false)
    private lateinit var gestureRecognizerHelper: GestureRecognizerHelper
    private val viewModel: MainViewModel by viewModels()

    private val requestPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted: Boolean ->
            hasCameraPermission.value = isGranted
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        checkCameraPermission()
        gestureRecognizerHelper = GestureRecognizerHelper(this, this)
        
        setContent {
            Reconhecimento_LGPTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    if (hasCameraPermission.value) {
                        MainScreen(viewModel, gestureRecognizerHelper)
                    } else {
                        Text("Camera permission not granted")
                    }
                }
            }
        }
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                hasCameraPermission.value = true
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    override fun onError(error: String) {
        Log.e("MainActivity", error)
    }

    override fun onResults(result: HandLandmarkerResult) {
        viewModel.onHandLandmarkerResult(result)
    }

    override fun onGestureRecognized(gesture: String) {
        viewModel.onGestureRecognized(gesture)
    }

    override fun onDestroy() {
        super.onDestroy()
        gestureRecognizerHelper.close()
    }
}
