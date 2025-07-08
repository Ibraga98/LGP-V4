# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /home/kenny/.gradle/caches/transforms-3/91742a6dc5a0c71fb1c6b13bbf1df27b/transformed/appcompat-1.6.1/proguard.txt
# You can edit the include path and order by changing the proguardFiles
# directive in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If you are using Java 8 language features, such as lambdas,
# you may need to add the following lines.
#-dontwarn java.util.function.**

# Retain annotation used by the TensorFlow Lite Java code generation process.
-keep @org.tensorflow.lite.annotations.UsedByReflection

# General rules for MediaPipe
-keep class com.google.mediapipe.** { *; }
-keep interface com.google.mediapipe.** { *; }
-keep class com.google.protobuf.** { *; }
-keep interface com.google.protobuf.** { *; }
-dontwarn com.google.protobuf.**
