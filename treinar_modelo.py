

import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf
# import seaborn as sns
# import matplotlib.pyplot as plt

# --- Parâmetros ---
DATA_FILE = "dados_processados.npz"
MODEL_FILE = "modelo_gestos_lgp.h5"
EPOCHS = 150
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss implementation for handling class imbalance.
    
    Args:
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter to reduce the relative loss for well-classified examples (default: 2.0)
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent NaN values
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy for each class
        cross_entropy = -y_true * K.log(y_pred)
        
        # Calculate p_t for each class
        p_t = K.sum(y_true * y_pred, axis=-1, keepdims=True)
        
        # Calculate alpha_t (simplified version)
        alpha_t = alpha
        
        # Calculate the modulating factor
        modulating_factor = K.pow(1.0 - p_t, gamma)
        
        # Apply focal loss formula 
        focal_loss_value = alpha_t * modulating_factor * K.sum(cross_entropy, axis=-1, keepdims=True)
        
        return K.mean(focal_loss_value)
    
    return focal_loss_fixed

def load_data():
    """Carrega os dados processados do arquivo .npz."""
    try:
        data = np.load(DATA_FILE)
        X = data['X']
        y = data['y']
        return X, y
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo de dados '{DATA_FILE}' não encontrado.")
        print("➡️ Por favor, execute o script 'processar_dados.py' primeiro.")
        return None, None

def build_model(input_shape, num_classes, use_focal_loss=True):
    """Constrói o modelo LSTM com arquitetura melhorada."""
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Choose loss function based on parameter
    if use_focal_loss:
        loss_function = focal_loss(alpha=0.25, gamma=2.0)
        print("🎯 Usando Focal Loss para lidar com classes desbalanceadas")
    else:
        loss_function = 'categorical_crossentropy'
        print("📊 Usando Categorical Crossentropy padrão")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_function, 
        metrics=['accuracy']
    )
    return model

def main():
    """Função principal para carregar dados, treinar e salvar o modelo."""
    X, y = load_data()
    if X is None:
        return

    # Carregar o mapeamento de classes
    try:
        with open("classes.json", "r") as f:
            label_map = json.load(f)
        num_classes = len(label_map)
    except FileNotFoundError:
        print("❌ Erro: Arquivo 'classes.json' não encontrado.")
        return

    print(f"Número de sequências: {X.shape[0]}")
    print(f"Frames por sequência: {X.shape[1]}")
    print(f"Número de features por frame: {X.shape[2]}")
    print(f"Número de classes: {num_classes}")

    # Analyze class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"\n📊 DISTRIBUIÇÃO DE CLASSES:")
    total_samples = len(y)
    max_count = max(counts)
    for label, count in zip(unique_labels, counts):
        class_name = [k for k, v in label_map.items() if v == label][0]
        percentage = (count / total_samples) * 100
        imbalance_ratio = max_count / count if count > 0 else float('inf')
        print(f"   {class_name}: {count} ({percentage:.1f}%) - Ratio: 1:{imbalance_ratio:.1f}")

    # Compute class weights to handle imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\n⚖️ PESOS DE CLASSE CALCULADOS (para balanceamento):")
    for i, weight in class_weight_dict.items():
        class_name = [k for k, v in label_map.items() if v == i][0]
        print(f"   {class_name}: {weight:.3f}")

    # Converter labels para o formato one-hot encoding
    y_categorical = to_categorical(y, num_classes=num_classes)

    # Dividir os dados em conjuntos de treino e validação com estratificação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )

    print(f"\nDados de treino: {X_train.shape[0]} amostras")
    print(f"Dados de validação: {X_val.shape[0]} amostras")

    # Construir o modelo com categorical crossentropy e class weights
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, num_classes, use_focal_loss=False)
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss')

    # Treinar o modelo com class weights para balanceamento
    print("\nIniciando o treino do modelo com balanceamento de classes...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,  # Apply class weights for balancing
        callbacks=[early_stop, checkpoint]
    )

    print(f"\n✅ Treino concluído. Melhor modelo salvo como '{MODEL_FILE}'.")
    
    # Evaluate the model and provide detailed metrics
    print("\n📈 AVALIAÇÃO DO MODELO:")
    
    # Load the best model
    from tensorflow.keras.models import load_model
    best_model = load_model(MODEL_FILE)
    
    # Make predictions on validation set
    y_pred = best_model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    
    # Print classification report
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    print("\n📊 RELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_val_classes, y_pred_classes, target_names=class_names))
    
    # Save detailed evaluation results
    evaluation_results = {
        'class_weights': class_weight_dict,
        'classification_report': classification_report(y_val_classes, y_pred_classes, target_names=class_names, output_dict=True),
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
    }
    
    with open("training_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print("📊 Resultados detalhados do treino salvos em 'training_results.json'")

if __name__ == "__main__":
    main()

