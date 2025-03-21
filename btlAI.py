import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
import joblib
import tensorflow as tf
from tensorflow.keras.losses import Loss

# Định nghĩa Focal Loss
class FocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        bce = - y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        focal_factor = tf.pow(1 - y_pred, self.gamma) * y_true + tf.pow(y_pred, self.gamma) * (1 - y_true)
        return tf.reduce_mean(self.alpha * focal_factor * bce)

# 1. Chuẩn bị dữ liệu
try:
    data = pd.read_csv('C:\\Users\\PC\\Downloads\\WA_Fn-UseC_-HR-Employee-Attrition.csv')
except FileNotFoundError:
    print("Không tìm thấy tệp. Vui lòng kiểm tra đường dẫn!")
    exit()

# Loại bỏ các cột không mang thông tin phân biệt
data = data.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])

# Mã hóa biến mục tiêu 'Attrition' (Yes=1, No=0)
label_encoder = LabelEncoder()
data['Attrition'] = label_encoder.fit_transform(data['Attrition'])

# Chọn các đặc trưng đầu vào (thêm các đặc trưng mới)
features = ['Age', 'Education', 'TotalWorkingYears', 'JobRole', 'MonthlyIncome', 
            'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany', 'OverTime', 
            'DistanceFromHome', 'PercentSalaryHike', 'EnvironmentSatisfaction', 
            'TrainingTimesLastYear', 'YearsSinceLastPromotion', 'NumCompaniesWorked', 
            'RelationshipSatisfaction', 'StockOptionLevel', 'JobInvolvement']
X = data[features]
y = data['Attrition']

# Tách các cột số và cột phân loại
numeric_features = ['Age', 'Education', 'TotalWorkingYears', 'MonthlyIncome', 
                    'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany', 
                    'DistanceFromHome', 'PercentSalaryHike', 'EnvironmentSatisfaction', 
                    'TrainingTimesLastYear', 'YearsSinceLastPromotion', 'NumCompaniesWorked', 
                    'RelationshipSatisfaction', 'StockOptionLevel', 'JobInvolvement']
categorical_features = ['JobRole', 'OverTime']

# Chuẩn hóa các cột số
scaler = StandardScaler()
X_numeric = scaler.fit_transform(X[numeric_features])

# Mã hóa one-hot các cột phân loại
X_categorical = pd.get_dummies(X[categorical_features], drop_first=True)

# Kết hợp lại các đặc trưng
X = np.hstack((X_numeric, X_categorical))

# Lưu tên các đặc trưng để sử dụng sau (cho tính toán tầm quan trọng của đặc trưng)
feature_names = numeric_features + list(X_categorical.columns)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiển thị phân phối lớp trước SMOTE
print("Phân phối lớp trước SMOTE (Tập huấn luyện):")
print(pd.Series(y_train).value_counts())

# Sử dụng SMOTE để xử lý mất cân bằng (có thể bỏ nếu không hiệu quả)
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# Hiển thị phân phối lớp sau SMOTE
# print("\nPhân phối lớp sau SMOTE (Tập huấn luyện):")
# print(pd.Series(y_train).value_counts())

# Tính trọng số lớp
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {0: 1.0, 1: 3.0}  # Tăng trọng số lớp 1

# 2. Xây dựng mô hình ANN
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(units=32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(units=1, activation='sigmoid'))

# Biên dịch mô hình với Focal Loss
model.compile(optimizer=Adam(learning_rate=0.0005), loss=FocalLoss(alpha=0.25, gamma=2.0), metrics=['accuracy'])

# Xem tóm tắt mô hình
print("\nTóm tắt mô hình:")
model.summary()

# 3. Huấn luyện mô hình với Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
history = model.fit(X_train, y_train, 
                    epochs=150, 
                    batch_size=32, 
                    validation_split=0.3, 
                    class_weight=class_weights_dict, 
                    callbacks=[early_stopping], 
                    verbose=1)

# 4. Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f'\nĐộ chính xác trên tập kiểm tra: {accuracy:.4f}')

# 5. Dự đoán trên tập kiểm tra
y_pred = (model.predict(X_test) > 0.3).astype("int32")  # Giảm ngưỡng để tăng recall
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=['Không nghỉ việc', 'Nghỉ việc']))

# 6. Hiển thị Ma trận Nhầm lẫn
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Không', 'Có'], yticklabels=['Không', 'Có'])
plt.title('Ma trận Nhầm lẫn')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()

# 7. Hiển thị Đường cong ROC và AUC
y_pred_prob = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Đường cong ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tỷ lệ Dương tính Giả (False Positive Rate)')
plt.ylabel('Tỷ lệ Dương tính Thật (True Positive Rate)')
plt.title('Đường cong Đặc tính Vận hành Người nhận (ROC)')
plt.legend(loc="lower right")
plt.show()

# 8. Hiển thị Tầm quan trọng của Đặc trưng (dùng Permutation Importance)
# Tạo một wrapper để sử dụng permutation_importance với mô hình Keras
from sklearn.base import BaseEstimator, ClassifierMixin

class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")
    
    def predict_proba(self, X):
        return self.model.predict(X)

# Tính tầm quan trọng của đặc trưng
keras_wrapper = KerasClassifierWrapper(model)
perm_importance = permutation_importance(keras_wrapper, X_test, y_test, n_repeats=10, random_state=42)

# Hiển thị tầm quan trọng của đặc trưng
sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.barh(np.array(feature_names)[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel('Tầm quan trọng của Đặc trưng (Permutation Importance)')
plt.title('Tầm quan trọng của Đặc trưng')
plt.show()

# 9. Trực quan hóa kết quả huấn luyện (Mất mát và Độ chính xác)
plt.figure(figsize=(12, 4))

# Biểu đồ Mất mát
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Mất mát Huấn luyện')
plt.plot(history.history['val_loss'], label='Mất mát Xác thực')
plt.title('Mất mát của Mô hình')
plt.xlabel('Vòng lặp (Epoch)')
plt.ylabel('Mất mát (Loss)')
plt.legend()

# Biểu đồ Độ chính xác
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Độ chính xác Huấn luyện')
plt.plot(history.history['val_accuracy'], label='Độ chính xác Xác thực')
plt.title('Độ chính xác của Mô hình')
plt.xlabel('Vòng lặp (Epoch)')
plt.ylabel('Độ chính xác (Accuracy)')
plt.legend()

plt.tight_layout()
plt.show()

# Lưu scaler để sử dụng sau
joblib.dump(scaler, 'scaler.pkl')
print("\nĐã lưu scaler vào 'scaler.pkl' để sử dụng sau này.")