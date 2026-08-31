import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Tạo dữ liệu mô phỏng
np.random.seed(42)
n_samples = 200

area = np.random.randint(50, 300, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
location_score = np.random.uniform(1, 10, n_samples)

price = (2.5 * area + 
         8000 * bedrooms + 
         5000 * bathrooms + 
         15000 * location_score + 
         np.random.normal(0, 5000, n_samples))

# 2. Tạo DataFrame
data = pd.DataFrame({
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'location_score': location_score,
    'price': price
})

# 3. ĐỊNH NGHĨA X VÀ Y TRƯỚC KHI CHIA (ĐÂY LÀ PHẦN EM BỊ SAI)
X = data[['area', 'bedrooms', 'bathrooms', 'location_score']]
y = data['price']

# 4. Chia dữ liệu train/test (SAU KHI CÓ X VÀ Y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Huấn luyện model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 7. Đánh giá model
print(f"Độ chính xác trên tập train: {model.score(X_train_scaled, y_train):.4f}")
print(f"Độ chính xác trên tập test: {model.score(X_test_scaled, y_test):.4f}")

# 8. Lưu model và scaler
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Đã lưu model thành công!")