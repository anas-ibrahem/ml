import sys
import os
import time
import pickle
import numpy as np
import librosa
import noisereduce as nr
from scipy import signal
import cv2
import tensorflow as tf
import xgboost as xgb

# --- MODEL PIPELINE CHOICE ---
# Options: "ALEX" (1D AlexNet) or "XG" (SVM Router + XGBoost)
MODEL_CHOICE = "XG"

TARGET_SR = 48000
TRIM_TOP_DB = 40
HP_CUTOFF = 50
HP_ORDER = 4
NOISE_PROP = 0.75
TARGET_RMS = 0.05

# --- CUSTOM LOSS FUNCTION FOR ALEXNET ---
@tf.keras.utils.register_keras_serializable()
class FocalLossTF(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, name='focal_loss_tf', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        pt = tf.exp(-ce_loss)
        return tf.reduce_mean(((1.0 - pt) ** self.gamma) * ce_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config
# ----------------------------------------

def _find_quietest_clip(y, sr, clip_dur=0.3):
    frame_len = int(sr * clip_dur)
    hop = frame_len // 2
    if len(y) <= frame_len: return y
    rms_vals = [np.sqrt(np.mean(y[i:i+frame_len]**2)) for i in range(0, len(y)-frame_len, hop)]
    best_start = int(np.argmin(rms_vals)) * hop
    return y[best_start : best_start + frame_len]

def clean_audio(y, sr, trim_top_db=TRIM_TOP_DB, hp_cutoff=HP_CUTOFF, hp_order=HP_ORDER, noise_prop=NOISE_PROP, target_rms=TARGET_RMS):
    y_trim, _ = librosa.effects.trim(y, top_db=trim_top_db, frame_length=2048, hop_length=512)
    noise_clip = _find_quietest_clip(y_trim, sr)
    y_nr = nr.reduce_noise(y=y_trim, sr=sr, y_noise=noise_clip, stationary=True, prop_decrease=noise_prop)
    rms = np.sqrt(np.mean(y_nr**2))
    if rms > 1e-9: y_nr = y_nr * (target_rms / rms)
    sos = signal.butter(hp_order, hp_cutoff, btype="hp", fs=sr, output="sos")
    y_hp = signal.sosfilt(sos, y_nr)
    return np.clip(y_hp, -1.0, 1.0)

def extract_features(y, sr):
    if len(y) < 1024:
        return np.zeros((128, 128, 1))
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_resized = cv2.resize(mel_db, (128, 128), interpolation=cv2.INTER_AREA)
    mel_expanded = np.expand_dims(mel_resized, axis=-1) 
    
    return mel_expanded

def main():
    # Get the data directory path from the command line argument
    if len(sys.argv) < 2:
        print("Error: Please provide the path to the data directory.")
        sys.exit(1)
        
    data_dir = sys.argv[1]
    
    # Read all .wav files in the data directory sorted by their numeric prefix
    files = os.listdir(data_dir)
    wav_files = [f for f in files if f.endswith('.wav')]
    
    # Sort files numerically (1.wav, 2.wav, 10.wav...)
    wav_files.sort(key=lambda x: int(x.split('.')[0])) 

    results = []
    times = []

    # ---------------------------------------------------------
    # LOAD YOUR MODEL HERE (Before the loop starts)
    # ---------------------------------------------------------
    print("Loading models and scalers...")
    try:
        # 1. Load Autoencoder MinMax Scaler
        if os.path.exists('scaler_params.npz'):
            scaler_data = np.load('scaler_params.npz')
            GLOBAL_X_MIN = scaler_data['X_min']
            GLOBAL_X_MAX = scaler_data['X_max']
        else:
            # Mathematical certainty: librosa.power_to_db(ref=np.max) yields [-80.0, 0.0]
            # Since you checked X_scaled.min() returning 0.0 and 1.0, these are the TRUE raw limits!
            print("Warning: scaler_params.npz not found for Autoencoder. Using librosa true limits [-80, 0].")
            GLOBAL_X_MIN, GLOBAL_X_MAX = -80.0, 0.0
        
        # 2. Load the trained Autoencoder bottleneck (Encoder)
        encoder = tf.keras.models.load_model('encoder_model.h5')
        
        if MODEL_CHOICE == "ALEX":
            # 3. Load StandardScaler for AlexNet
            if os.path.exists('alexnet/scaler_alex.pkl'):
                with open('alexnet/scaler_alex.pkl', 'rb') as f:
                    scaler_alex = pickle.load(f)
            elif os.path.exists('scaler_alex.pkl'):
                with open('scaler_alex.pkl', 'rb') as f:
                    scaler_alex = pickle.load(f)
            else:
                print("Warning: scaler_alex.pkl not found! AlexNet inputs won't be scaled correctly.")
                class DummyScaler:
                    def transform(self, X): return X
                scaler_alex = DummyScaler()
                
            # 4. Load the final supervised model (AlexNet 1D)
            # We pass in custom_objects since it was trained with a custom Focal Loss
            classifier = tf.keras.models.load_model(
                'alexnet/tf_alexnet.keras', 
                custom_objects={'FocalLossTF': FocalLossTF}
            )
            
        elif MODEL_CHOICE == "XG":
            # 3. Load SVM Router
            if os.path.exists('stage1_svm/best_router.pkl'):
                with open('stage1_svm/best_router.pkl', 'rb') as f:
                    svm_router = pickle.load(f)
            else:
                print("Warning: stage1_svm/best_router.pkl not found! Falling back to M1.")
                class DummySVM:
                    def predict(self, X): return [0]
                svm_router = DummySVM()

            # 4. Load XGBoost Models (one per machine)
            xgb_models = {}
            for i in range(3):
                model_path = f'stage2_xg/xgb_machine_{i}.pkl'
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        xgb_models[i] = pickle.load(f)
                else:
                    print(f"Warning: {model_path} not found! Falling back to class 0.")
                    class DummyXGB:
                        def predict(self, X): return [0]
                    xgb_models[i] = DummyXGB()

        print("Models loaded successfully.")
    except Exception as e:
        print(f"File not found for model/scaler loading: {e}\nFalling back to Dummy Models for placeholder.")
        GLOBAL_X_MIN, GLOBAL_X_MAX = 0.0, 1.0
        
        class DummyEncoder:
            def predict(self, X, verbose=0): return np.zeros((1, 256))
            
        class DummyClassifier:
            def predict(self, X, verbose=0): return np.zeros((1, 6)) # 6 classes
            
        encoder = DummyEncoder()
        classifier = DummyClassifier()

    # ---------------------------------------------------------
    # PROCESS EACH AUDIO FILE
    # ---------------------------------------------------------
    for wav_file in wav_files:
        # Get the path
        file_path = os.path.join(data_dir, wav_file)
        
        audio_data, sr = librosa.load(file_path, sr=TARGET_SR, duration=10.0) 
        
        # =========================================================
        # START TIMER HERE! (After reading the file)
        # =========================================================
        start_time = time.time()
        
        # STEP 2: Preprocess the loaded audio (Noise reduction, silence removal, etc.)
        cleaned_data = clean_audio(audio_data, sr) 
        
        # STEP 3: Extract features
        features = extract_features(cleaned_data, sr)
        
        # STEP 3.5: Apply True Global MinMax Scaling
        features = features.astype(np.float32)
        if GLOBAL_X_MAX - GLOBAL_X_MIN > 0:
            features = (features - GLOBAL_X_MIN) / (GLOBAL_X_MAX - GLOBAL_X_MIN)
        else:
            features = features - GLOBAL_X_MIN
            
        # STEP 4: Predict class (0-5)
        # Expand dims for batch size of 1: shape becomes (1, 128, 128, 1)
        batch_features = np.expand_dims(features, axis=0)
        
        # 1. Get 256D embedding from encoder
        embedding = encoder.predict(batch_features, verbose=0)
        
        # 2. Route and Predict based on chosen pipeline
        if MODEL_CHOICE == "ALEX":
            # 2a. Apply StandardScaler (Fit earlier on X_cv during Model 4 AlexNet training)
            try:
                embedding_scaled = scaler_alex.transform(embedding)
            except NameError:
                embedding_scaled = embedding
            
            # Predict final class (AlexNet softmax output)
            pred_probs = classifier.predict(embedding_scaled, verbose=0)
            prediction = np.argmax(pred_probs, axis=-1)[0] 
            
        elif MODEL_CHOICE == "XG":
            try:
                embedding_svm = embedding.astype(np.float64)
                
                # Predict Machine (0, 1, or 2) using Router
                mach_id = svm_router.predict(embedding_svm)[0]
                
                # Predict State (0 for Normal, 1 for Abnormal) using XGBoost
                state_pred = xgb_models[mach_id].predict(embedding_svm)[0]
                
                # Combine Machine & State into Final Label (0-5)
                # Map logic: Machine 1 (0 * 2) + state -> 0 or 1
                #            Machine 2 (1 * 2) + state -> 2 or 3
                #            Machine 3 (2 * 2) + state -> 4 or 5
                prediction = int((mach_id * 2) + state_pred)
            except Exception as e:
                # Fallback if Dummy Models took over and failed to index
                prediction = 0
        
        # =========================================================
        # END TIMER HERE! 
        # =========================================================
        end_time = time.time()
        
        # Store results
        results.append(str(prediction))
        times.append(f"{end_time - start_time:.3f}")

    # ---------------------------------------------------------
    # WRITE OUTPUTS
    # ---------------------------------------------------------
    with open("results.txt", "w") as f:
        f.write("\n".join(results) + "\n")
        
    with open("time.txt", "w") as f:
        f.write("\n".join(times) + "\n")
        
    print("Processing complete. results.txt and time.txt generated.")

if __name__ == "__main__":
    main()