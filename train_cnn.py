import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, GlobalMaxPooling1D, Dense, Concatenate, Multiply, Add, ReLU, Softmax, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import h5py
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import class_weight
from tensorflow.keras.regularizers import l1_l2

# Constants
SAMPLE_SIZE = 1024
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)



# Data Loading and Cleaning (unchanged)
def load_combined_data(filenames):
    local_inputs_list, global_inputs_list, labels_list, df_list = [], [], [], []
    for filename in filenames:
        with h5py.File(filename, "r") as f:
            local_inputs_list.append(np.array(f["local_inputs"][:]))
            global_inputs_list.append(np.array(f["global_inputs"][:]))
            labels_list.append(np.array(f["labels"][:], dtype=int))
            metadata = f["metadata"][:]
            df = pd.DataFrame({
                'TIC': metadata['TIC'].astype(int),
                'sector': metadata['sector'].astype(int),
                'path_to_fits': metadata['path_to_fits'].astype(str),
                'TOI Disposition': metadata['TOI Disposition'].astype(str),
                'label': metadata['label'].astype(int)
            })
            df_list.append(df)
    local_inputs = np.concatenate(local_inputs_list, axis=0)
    global_inputs = np.concatenate(global_inputs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    df_merged = pd.concat(df_list, ignore_index=True)
    assert len(local_inputs) == len(global_inputs) == len(labels) == len(df_merged)
    local_inputs, global_inputs, labels, df_merged = clean_loaded_data(local_inputs, global_inputs, labels, df_merged)
    return local_inputs, global_inputs, labels, df_merged

def clean_loaded_data(local_inputs, global_inputs, labels, df_merged):
    valid_mask = (~np.isnan(local_inputs).any(axis=(1,2)) & 
                  ~np.isnan(global_inputs).any(axis=(1,2)))
    if not np.all(valid_mask):
        print(f"Removing {len(valid_mask) - np.sum(valid_mask)} invalid samples")
    local_clean = local_inputs[valid_mask]
    global_clean = global_inputs[valid_mask]
    labels_clean = labels[valid_mask]
    df_clean = df_merged.iloc[valid_mask].copy()
    local_clean = (local_clean - np.mean(local_clean, axis=1, keepdims=True)) / (np.std(local_clean, axis=1, keepdims=True) + 1e-8)
    global_clean = (global_clean - np.mean(global_clean, axis=1, keepdims=True)) / (np.std(global_clean, axis=1, keepdims=True) + 1e-8)
    assert len(local_clean) == len(global_clean) == len(labels_clean) == len(df_clean)
    assert not np.isnan(local_clean).any()
    assert not np.isnan(global_clean).any()
    return local_clean, global_clean, labels_clean, df_clean

# Model Architecture
def residual_block(x, filters, kernel_size):
    y = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5))(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l1_l2(l1=1e-6, l2=1e-5))(y)
    y = BatchNormalization()(y)
    out = Add()([x, y])
    return ReLU()(out)

def build_model():
    local_input = Input(shape=(1024, 1), name='local_lightcurve')
    global_input = Input(shape=(1024, 1), name='global_lightcurve')
    kernel_reg = l1_l2(l1=1e-6, l2=1e-5)

    # Local branch (deeper)
    x = Conv1D(64, kernel_size=9, padding='same', activation='relu', kernel_regularizer=kernel_reg)(local_input)
    x = BatchNormalization()(x)
    x = residual_block(x, 64, 5)
    x = Conv1D(128, kernel_size=5, padding='same', activation='relu', kernel_regularizer=kernel_reg)(x)
    x = BatchNormalization()(x)
    x = residual_block(x, 128, 3)
    x = Conv1D(256, kernel_size=3, padding='same', activation='relu', kernel_regularizer=kernel_reg)(x)
    x = BatchNormalization()(x)
    x = residual_block(x, 256, 3)
    x = Dropout(0.3)(x)
    
    # Multi-head attention for local branch
    x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)  # Self-attention
    x = GlobalMaxPooling1D()(x)

    # Global branch (deeper)
    y = Conv1D(64, kernel_size=9, padding='same', activation='relu', kernel_regularizer=kernel_reg)(global_input)
    y = BatchNormalization()(y)
    y = residual_block(y, 64, 5)
    y = Conv1D(128, kernel_size=5, padding='same', activation='relu', kernel_regularizer=kernel_reg)(y)
    y = BatchNormalization()(y)
    y = residual_block(y, 128, 3)
    y = Conv1D(256, kernel_size=3, padding='same', activation='relu', kernel_regularizer=kernel_reg)(y)
    y = BatchNormalization()(y)
    y = residual_block(y, 256, 3)
    y = Dropout(0.3)(y)
    
    # Multi-head attention for global branch
    y = MultiHeadAttention(num_heads=4, key_dim=64)(y, y)  # Self-attention
    y = GlobalMaxPooling1D()(y)

    # Merge and classify
    combined = Concatenate()([x, y])
    combined = Dense(256, activation='relu', kernel_regularizer=kernel_reg)(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(128, activation='relu', kernel_regularizer=kernel_reg)(combined)
    combined = Dropout(0.2)(combined)
    output = Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[local_input, global_input], outputs=output)
    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss='binary_crossentropy',  # Reverted to binary crossentropy
        metrics=['accuracy', tf.keras.metrics.AUC(name='roc_auc'), tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
    )
    return model

# Data Preparation
def fetch_data():
    local_inputs, global_inputs, labels, df_merged = load_combined_data(["/mnt/data/LCs_1024_CNN_Input.h5"])
    local_b, global_b, labels_b, df_b = load_combined_data(["/mnt/data/TCEs_LCs_1024_CNN_Input.h5"])

    local_clean = np.concatenate([local_inputs, local_b])
    global_clean = np.concatenate([global_inputs, global_b])
    labels_clean = np.concatenate([labels, labels_b])
    df_clean = pd.concat([df_merged, df_b], axis=0, ignore_index=True)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, val_idx = next(sgkf.split(df_clean, labels_clean, groups=df_clean['TIC']))

    local_train, local_val = local_clean[train_idx], local_clean[val_idx]
    global_train, global_val = global_clean[train_idx], global_clean[val_idx]
    labels_train, labels_val = labels_clean[train_idx], labels_clean[val_idx]
    
    return local_train, local_val, global_train, global_val, labels_train, labels_val

# Custom Callback to Save in Float32
class Float32ModelCheckpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        original_policy = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy('float32')
        super().on_epoch_end(epoch, logs)
        tf.keras.mixed_precision.set_global_policy(original_policy)

# Main Execution
if __name__ == "__main__":
    local_train, local_val, global_train, global_val, labels_train, labels_val = fetch_data()
    print(f"Training samples: {len(local_train)}, Validation samples: {len(local_val)}")
    
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    lr_scheduler = ReduceLROnPlateau(monitor='val_roc_auc', factor=0.5, patience=5, min_lr=1e-6, mode='max')
    early_stopping = EarlyStopping(monitor='val_roc_auc', patience=20, restore_best_weights=True, mode='max')
    checkpoint = Float32ModelCheckpoint(
        "/mnt/data/modelV3.h5",
        monitor='val_roc_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks = [lr_scheduler, early_stopping, checkpoint]
    
    model = build_model()
    model.summary()
    
    history = model.fit(
        (local_train, global_train),
        labels_train,
        validation_data=((local_val, global_val), labels_val),
        batch_size=32,
        epochs=100,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )
    
    val_loss, val_acc, val_roc_auc, val_pr_auc = model.evaluate((local_val, global_val))
    print(f"Final Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, ROC AUC: {val_roc_auc:.4f}, PR AUC: {val_pr_auc:.4f}")
    print("Best model saved to modelV3.h5")