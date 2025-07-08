

import tensorflow as tf
import os

# --- Parâmetros ---
KERAS_MODEL_FILE = "modelo_gestos_lgp.h5"
TFLITE_MODEL_FILE = "modelo_gestos_lgp.tflite"

def convert_to_tflite():
    """Converte o modelo Keras (.h5) para o formato TensorFlow Lite (.tflite)."""
    
    # Verificar se o arquivo do modelo Keras existe
    if not os.path.exists(KERAS_MODEL_FILE):
        print(f"❌ Erro: Arquivo do modelo '{KERAS_MODEL_FILE}' não encontrado.")
        print("➡️ Por favor, treine o modelo executando 'treinar_modelo.py' primeiro.")
        return

    try:
        # Carregar o modelo Keras treinado
        print(f"🔄 Carregando o modelo Keras de '{KERAS_MODEL_FILE}'...")
        model = tf.keras.models.load_model(KERAS_MODEL_FILE)

        # Inicializar o conversor TFLite a partir do modelo Keras
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # --- INÍCIO DA CORREÇÃO ---
        # Adicionar estas duas linhas para lidar com operadores LSTM que não são totalmente suportados nativamente
        # Isso inclui operadores do TensorFlow no modelo TFLite para garantir a compatibilidade.
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Habilita operadores TFLite nativos
            tf.lite.OpsSet.SELECT_TF_OPS    # Habilita operadores do TensorFlow (para compatibilidade com LSTM)
        ]
        converter._experimental_lower_tensor_list_ops = False
        # --- FIM DA CORREÇÃO ---

        # Aplicar otimizações padrão (inclui quantização dinâmica)
        # Isso reduz o tamanho do modelo e pode acelerar a inferência
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Realizar a conversão
        print("⚙️ Convertendo o modelo para TensorFlow Lite...")
        tflite_model = converter.convert()

        # Salvar o modelo TFLite no disco
        with open(TFLITE_MODEL_FILE, 'wb') as f:
            f.write(tflite_model)

        # Obter e comparar os tamanhos dos arquivos
        keras_model_size = os.path.getsize(KERAS_MODEL_FILE) / 1024  # em KB
        tflite_model_size = len(tflite_model) / 1024  # em KB

        print("\n✅ Conversão concluída com sucesso!")
        print(f"💾 Modelo TFLite salvo como '{TFLITE_MODEL_FILE}'")
        print(f"📊 Tamanho do modelo original (.h5): {keras_model_size:.2f} KB")
        print(f"📊 Tamanho do modelo convertido (.tflite): {tflite_model_size:.2f} KB")
        print(f"📉 Redução de tamanho: {(1 - tflite_model_size / keras_model_size) * 100:.2f}%")

    except Exception as e:
        print(f"❌ Ocorreu um erro durante a conversão: {e}")

if __name__ == "__main__":
    convert_to_tflite()

