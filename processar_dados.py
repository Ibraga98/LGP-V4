import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import re
import json
from scipy.spatial.transform import Rotation

# --- Parâmetros ---
DATASET_PATH = "dataset_limpo"
OUTPUT_FILE = "dados_processados.npz"
SEQUENCE_LENGTH = 20  # Número de frames por amostra de gesto

# --- Inicialização do MediaPipe ---
mp_hands = mp.solutions.hands
# Much more aggressive parameters for landmark detection
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.1,  # Very low threshold
    min_tracking_confidence=0.1,   # Very low threshold
    model_complexity=1             # Higher complexity for better accuracy
)

def extract_landmarks(image_path):
    """Processa uma única imagem e extrai os marcos da mão com melhor pré-processamento."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Aviso: Não foi possível carregar a imagem: {image_path}")
            return None

        # Aggressive image preprocessing for better detection
        # Always resize to a larger standard size
        height, width = image.shape[:2]
        target_size = 640  # Larger size for better detection
        if width != target_size or height != target_size:
            scale = target_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Pad to square if needed
            if new_width != target_size or new_height != target_size:
                top = (target_size - new_height) // 2
                bottom = target_size - new_height - top
                left = (target_size - new_width) // 2
                right = target_size - new_width - left
                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Multiple preprocessing attempts
        processed_images = []
        
        # Original enhanced
        enhanced = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
        processed_images.append(enhanced)
        
        # High contrast version
        high_contrast = cv2.convertScaleAbs(image, alpha=2.0, beta=10)
        processed_images.append(high_contrast)
        
        # Histogram equalization
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
        hist_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        processed_images.append(hist_eq)
        
        # Try detection on multiple versions
        for proc_image in processed_images:
            image_rgb = cv2.cvtColor(proc_image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                return landmarks
        
        # If no detection on any version, return None
        return None
        
    except Exception as e:
        print(f"❌ Erro ao processar imagem {image_path}: {e}")
        return None

def augment_sequence(sequence, augmentation_factor=2):
    """Aplica aumentação de dados a uma sequência de landmarks."""
    augmented_sequences = []
    
    for i in range(augmentation_factor):
        # Create augmented copy
        aug_sequence = []
        
        # Random rotation around y-axis (depth)
        rotation_angle = np.random.uniform(-15, 15)  # degrees
        
        # Random scaling
        scale_factor = np.random.uniform(0.9, 1.1)
        
        # Random translation
        translation_x = np.random.uniform(-0.05, 0.05)
        translation_y = np.random.uniform(-0.05, 0.05)
        
        for frame_landmarks in sequence:
            if frame_landmarks is not None and len(frame_landmarks) == 63:  # 21 landmarks * 3 coords
                # Reshape to (21, 3)
                landmarks_3d = frame_landmarks.reshape(21, 3)
                
                # Apply rotation around z-axis (simulating slight hand rotation)
                center_x = np.mean(landmarks_3d[:, 0])
                center_y = np.mean(landmarks_3d[:, 1])
                
                # Translate to origin
                landmarks_3d[:, 0] -= center_x
                landmarks_3d[:, 1] -= center_y
                
                # Apply 2D rotation
                cos_angle = np.cos(np.radians(rotation_angle))
                sin_angle = np.sin(np.radians(rotation_angle))
                rotation_matrix = np.array([
                    [cos_angle, -sin_angle],
                    [sin_angle, cos_angle]
                ])
                
                landmarks_2d = landmarks_3d[:, :2] @ rotation_matrix.T
                landmarks_3d[:, :2] = landmarks_2d
                
                # Apply scaling
                landmarks_3d *= scale_factor
                
                # Translate back and add random translation
                landmarks_3d[:, 0] += center_x + translation_x
                landmarks_3d[:, 1] += center_y + translation_y
                
                # Clip to valid range [0, 1]
                landmarks_3d = np.clip(landmarks_3d, 0, 1)
                
                aug_sequence.append(landmarks_3d.flatten())
            else:
                # Keep original frame if it's None or invalid
                aug_sequence.append(frame_landmarks)
        
        augmented_sequences.append(aug_sequence)
    
    return augmented_sequences

def create_sequences():
    """Cria sequências de marcos a partir do dataset de imagens usando uma janela deslizante."""
    sequences = []
    labels = []
    
    try:
        class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    except FileNotFoundError:
        print(f"❌ Erro: O diretório do dataset '{DATASET_PATH}' não foi encontrado.")
        return

    if not class_names:
        print(f"❌ Erro: Nenhuma pasta de classe encontrada em '{DATASET_PATH}'.")
        return

    print(f"Encontradas {len(class_names)} classes: {class_names}")

    label_map = {name: i for i, name in enumerate(class_names)}
    
    with open("classes.json", "w") as f:
        json.dump(label_map, f, indent=4)
    print("✅ Mapeamento de classes salvo em classes.json")
    
    # Statistics tracking
    class_stats = {name: {'total_images': 0, 'successful_extractions': 0, 'sequences_generated': 0} for name in class_names}

    for class_name in class_names:
        class_path = os.path.join(DATASET_PATH, class_name)
        
        def sort_key(filename):
            # Prioritize the number after the last underscore for sorting
            match = re.search(r'_(\d+)\.(jpg|jpeg)$', filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
            # Fallback for filenames that don't match the pattern (e.g., IMG_6015.jpg)
            # We'll extract any numbers from the name to create a sort value
            fallback_match = re.findall(r'\d+', filename)
            if fallback_match:
                return int(''.join(fallback_match))
            # If no numbers are found at all, return 0
            return 0

        try:
            image_files = sorted(
                [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))], 
                key=sort_key
            )
        except FileNotFoundError:
            print(f"⚠️ Aviso: A pasta da classe '{class_name}' não foi encontrada ou está vazia.")
            continue

        print(f"\nProcessando classe: '{class_name}' ({len(image_files)} imagens)")
        class_stats[class_name]['total_images'] = len(image_files)
        
        all_landmarks = []
        successful_extractions = 0
        for image_file in tqdm(image_files, desc=f"  Extraindo marcos de '{class_name}'"):
            image_path = os.path.join(class_path, image_file)
            landmarks = extract_landmarks(image_path)
            all_landmarks.append(landmarks)
            if landmarks is not None:
                successful_extractions += 1
        
        class_stats[class_name]['successful_extractions'] = successful_extractions
        success_rate = (successful_extractions / len(image_files)) * 100 if image_files else 0
        print(f"  Taxa de sucesso na extração: {success_rate:.1f}% ({successful_extractions}/{len(image_files)})")

        sequences_for_class = []
        for i in tqdm(range(len(all_landmarks) - SEQUENCE_LENGTH + 1), desc=f"  Criando sequências para '{class_name}'"):
            window = all_landmarks[i : i + SEQUENCE_LENGTH]
            
            if not any(lm is not None for lm in window):
                continue

            sequence = []
            num_features = 63 # 21 landmarks * 3 coordenadas (x, y, z)
            for landmarks in window:
                if landmarks is not None:
                    sequence.append(landmarks)
                else:
                    sequence.append(np.zeros(num_features))

            sequences_for_class.append(sequence)
        
        # Store original sequences
        for sequence in sequences_for_class:
            sequences.append(sequence)
            labels.append(label_map[class_name])
            class_stats[class_name]['sequences_generated'] += 1
        
        # Apply aggressive data balancing
        if len(sequences_for_class) > 0:
            if class_name == 'ola':
                # Limit ola class to reduce dominance - only use first 20 sequences
                sequences_to_keep = min(20, len(sequences_for_class))
                sequences_for_class = sequences_for_class[:sequences_to_keep]
                print(f"  ⚠️ Limitando classe 'ola' para {sequences_to_keep} sequências para reduzir dominância")
                
                # Remove the extra sequences that were already added
                sequences = sequences[:-class_stats[class_name]['sequences_generated']]
                labels = labels[:-class_stats[class_name]['sequences_generated']]
                class_stats[class_name]['sequences_generated'] = 0
                
                # Re-add the limited sequences
                for sequence in sequences_for_class:
                    sequences.append(sequence)
                    labels.append(label_map[class_name])
                    class_stats[class_name]['sequences_generated'] += 1
            
            elif len(sequences_for_class) < 50:  # Boost all other classes significantly
                target_sequences = 50  # Target number of sequences per class
                augmentation_factor = max(1, target_sequences // len(sequences_for_class))
                augmentation_factor = min(8, augmentation_factor)  # Cap at 8x augmentation
                
                print(f"  🔄 Aplicando aumentação agressiva (fator: {augmentation_factor}) para classe '{class_name}'")
                
                for original_sequence in tqdm(sequences_for_class, desc=f"  Aumentando dados para '{class_name}'"):
                    augmented_sequences = augment_sequence(original_sequence, augmentation_factor)
                    for aug_sequence in augmented_sequences:
                        sequences.append(aug_sequence)
                        labels.append(label_map[class_name])
                        class_stats[class_name]['sequences_generated'] += 1

    if not sequences:
        print("\n❌ Erro: Nenhuma sequência foi criada. Verifique o dataset e as configurações.")
        return

    # Print detailed statistics
    print("\n" + "="*80)
    print("📊 ESTATÍSTICAS DETALHADAS DO PROCESSAMENTO")
    print("="*80)
    total_sequences = len(sequences)
    
    for class_name in class_names:
        stats = class_stats[class_name]
        pct_sequences = (stats['sequences_generated'] / total_sequences) * 100 if total_sequences > 0 else 0
        success_rate = (stats['successful_extractions'] / stats['total_images']) * 100 if stats['total_images'] > 0 else 0
        
        print(f"\n📁 Classe: {class_name}")
        print(f"   📷 Imagens totais: {stats['total_images']}")
        print(f"   ✅ Extrações bem-sucedidas: {stats['successful_extractions']} ({success_rate:.1f}%)")
        print(f"   🔗 Sequências geradas: {stats['sequences_generated']} ({pct_sequences:.1f}% do total)")
    
    print(f"\n🎯 RESUMO GERAL:")
    print(f"   Total de sequências: {total_sequences}")
    
    # Calculate class imbalance
    labels_array = np.array(labels)
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    print(f"\n⚖️ DISTRIBUIÇÃO DE CLASSES (IMBALANCEAMENTO):")
    max_count = max(counts)
    for label, count in zip(unique_labels, counts):
        class_name = [k for k, v in label_map.items() if v == label][0]
        percentage = (count / total_sequences) * 100
        imbalance_ratio = max_count / count if count > 0 else float('inf')
        print(f"   {class_name}: {count} sequências ({percentage:.1f}%) - Ratio: 1:{imbalance_ratio:.1f}")
    
    print("="*80)

    np.savez_compressed(OUTPUT_FILE, X=np.array(sequences), y=np.array(labels))
    print(f"\n✅ Processamento concluído! {len(sequences)} sequências salvas em '{OUTPUT_FILE}'.")
    
    # Save statistics for later use
    with open("preprocessing_stats.json", "w") as f:
        json.dump(class_stats, f, indent=4)
    print("📊 Estatísticas detalhadas salvas em 'preprocessing_stats.json'")


if __name__ == "__main__":
    create_sequences()
