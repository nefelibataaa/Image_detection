import os
import numpy as np
import tensorflow as tf
from keras import layers, models, Input
from keras.layers import Reshape
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from tensorflow.python.keras.utils.np_utils import to_categorical
from PIL import Image
import time

# 从路径加载灰度图像数据并打标签
def load_images_and_labels(benign_path, malware_path):
    X = []
    y = []

    for filename in os.listdir(benign_path):
        if filename.endswith('.png'):
            file_path = os.path.join(benign_path, filename)
            img = Image.open(file_path).convert('L')
            img = img.resize((64, 64))
            img_array = np.array(img)
            X.append(img_array)
            y.append(0)

    for filename in os.listdir(malware_path):
        if filename.endswith('.png'):
            file_path = os.path.join(malware_path, filename)
            img = Image.open(file_path).convert('L')
            img = img.resize((64, 64))
            img_array = np.array(img)
            X.append(img_array)
            y.append(1)

    return np.array(X), np.array(y)

# 双注意力机制模块
class DualAttentionModule(tf.keras.layers.Layer):
    def __init__(self):
        super(DualAttentionModule, self).__init__()
        self.channel_attention = layers.Conv2D(1, (1, 1), activation='sigmoid')
        self.spatial_attention = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')

    def call(self, inputs):
        # Channel attention
        channel_avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        channel_max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        channel_attention = self.channel_attention(channel_avg_pool + channel_max_pool)
        channel_refined = inputs * channel_attention

        # Spatial attention
        spatial_avg_pool = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        spatial_max_pool = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        spatial_attention = self.spatial_attention(spatial_avg_pool + spatial_max_pool)
        refined_features = channel_refined * spatial_attention
        
        return refined_features

# 定义CNN模型
def create_cnn_model():
    input_layer = Input(shape=(64, 64, 1))
    x = input_layer

    filters = [32, 64, 128, 256, 128]
    for f in filters:
        x = layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

    model = models.Model(inputs=input_layer, outputs=x)
    return model

# 定义特征提取模型
def create_feature_extraction_model():
    cnn_model = create_cnn_model()
    input_layer = cnn_model.input
    feature_map = cnn_model.output

    dual_attention = DualAttentionModule()(feature_map)
    local_features = layers.Flatten()(dual_attention)

    reshaped_attention = Reshape((dual_attention.shape[1] * dual_attention.shape[2], dual_attention.shape[3]))(dual_attention)
    bilstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(reshaped_attention)
    global_features = layers.Flatten()(bilstm)

    fused_features = layers.concatenate([local_features, global_features], axis=-1)
    return models.Model(inputs=input_layer, outputs=fused_features)

# 定义完整的分类模型
def create_classification_model(num_classes):
    feature_extraction_model = create_feature_extraction_model()
    input_layer = feature_extraction_model.input
    features = feature_extraction_model.output

    x = layers.Dense(512, activation='relu')(features)
    x = layers.Dense(256, activation='relu')(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

# 数据加载和预处理
benign_path = "/home/xyq/Image_detection/image_benign"
malware_path = "/home/xyq/Image_detection/image_malicious"

X, y = load_images_and_labels(benign_path, malware_path)
X = np.expand_dims(X, axis=-1)
y = to_categorical(y, num_classes=2)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建和编译模型
num_classes = 2 
model = create_classification_model(num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 模型结构摘要
model.summary()

# 训练模型
model.fit(X_train, y_train, epochs=5, batch_size=64)
model.save('/home/xyq/Image_detection/model_file.keras')