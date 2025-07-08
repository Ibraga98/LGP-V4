package com.example.reconhecimento_lgp

import androidx.lifecycle.ViewModel
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

class MainViewModel : ViewModel() {
    private val _recognizedGesture = MutableStateFlow<String?>(null)
    val recognizedGesture = _recognizedGesture.asStateFlow()

    private val _handLandmarkerResult = MutableStateFlow<HandLandmarkerResult?>(null)
    val handLandmarkerResult = _handLandmarkerResult.asStateFlow()

    fun onGestureRecognized(gesture: String) {
        _recognizedGesture.value = gesture
    }

    fun onHandLandmarkerResult(result: HandLandmarkerResult) {
        _handLandmarkerResult.value = result
    }
}
