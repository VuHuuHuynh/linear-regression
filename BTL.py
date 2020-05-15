import numpy as np
import pandas as pd
# from show import ShowFig, ShowInfo
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('insurance.csv', encoding='utf-8', sep=',')

# Chuyển đổi cột 'sex', 'smoke', 'region' sang dạng số
cols = ['sex', 'smoke', 'region']
new_data = pd.get_dummies(data, cols, drop_first=True)
# print(new_data.head())

# ShowInfo(data)
# ShowFig(new_data)

'''
Biểu đồ phân tán của age, bmi theo charges
    - charges tăng khi age tăng tuyến tính
    - chỉ số bmi xác định được sự phân bố của chi phí bảo hiểm đa số được chi trả cho những người có chỉ số từ 20 cho tới hơn 40
'''
plt.figure(figsize=(20, 5))

features = ['age', 'bmi']
target = data['charges']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = data[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('charges')
# plt.show()

# Chuẩn bị data training
# Nối 3 cột age, bmi, smoke_yes lại bằng cách sử dụng np.c_ của numpy
X = pd.DataFrame(np.c_[new_data['age'], new_data['bmi'], new_data['smoke_yes']], columns=['age', 'bmi', 'smoke_yes'])
Y = new_data['charges']

# Tách data thành training và test(tỉ lệ: training=0.8, test=0.2)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Training và test mẫu
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# Đánh giá mô hình bằng R2
# 1. Đánh giá mô hình training
y_train_predict = lin_model.predict(X_train)
r2 = r2_score(Y_train, y_train_predict)
print("The model performance for training set")
print("-------------------------------------")
print(f'R2 score is {r2}', '\n')

# 2. Đánh giá mô hình test
y_test_predict = lin_model.predict(X_test)
r2 = r2_score(Y_test, y_test_predict)
print("The model performance for testing set")
print("-------------------------------------")
print(f'R2 score is {r2}', '\n')

'''
    - 0 < r2 < 1
    - Giá trị của R2 thu được (training) = 0.7386685402148381, có nghĩa là từ 3 biến độc lập age, bmi, smoke_yes, có ảnh hưởng tác động đến charges là 73.87%. Trong khi các phần còn lại bị ảnh hưởng bởi các biến khác ngoài mô hình. Giá trị của r2 càng nhỏ thì tác động của các biến độc lập đến biến phụ thuộc càng ít và ngược lại.
    - Tương tự, R2(test) = 0.7793451556442221 => tác động của các biến độc lập lên biến charges là 77.93%.
    =>> Kết quả của r2 training và r2 test có chêch lệch thấp điều này khẳng định việc sử dụng mô hình training có thể sẽ thu được kết quả mong muốn.

'''

# Tính w0 trong công thức đường hồi quy tuyến tính: y = w0 + w1*x1 + w2*x2 +...
intercept = lin_model.intercept_
print(f'Intercept value is {intercept}', '\n')

# Tính hệ số hồi quy
coeff_df = pd.DataFrame(lin_model.coef_, X.columns, columns=['Coefficient'])
print(coeff_df, '\n')

'''
-***- Nhận định đánh giá từ hệ số trong pt hồi quy -**-
    - Tính được các giá trị của w1, w2, w3 lần lượt là 260.073, 321.391, 23621.229
    - Ta nhận thấy giá trị của w3 quá lớn và không cân xứng, lí do vì nó là hệ số của x3 là của feature 'smoke_yes'. Điều này biểu thị rằng tình trạng hút thuốc của cá nhân đóng vai trò không tương xứng trong việc xác định chi phí bảo hiểm của một cá nhân.
    - Chính vì điều đó, nếu bạn là người hút thuốc thì sẽ làm tăng rủi ro cho cuộc sống của bạn nên chi phí bảo hiểm cao
    - Ngoài ra, w1 ứng với x1 là 'age' có giá trị là 260.073. Điều này có nghĩa là cứ tăng thêm 1 tuổi thì chi phí bảo hiểm sẽ tăng thêm 260.073$
'''

# # Kiểm tra khác biệt giữa giá trị dự đoán và giá trị thật
# y_pred = lin_model.predict(X_test)
# data = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
# print(data.head(10))


# Xây dựng hàm dự đoán chi phí bảo hiểm theo độ tuổi, bmi và tình trạng hút thuốc
def calc_insurance(age, bmi, smoking_status):
    if smoking_status == 'smoker':
        s = 1
    else:
        s = 0
    insurance_cost = (intercept + coeff_df.iloc[0] * age + coeff_df.iloc[1] * bmi + coeff_df.iloc[2] * s)
    print(f'Insurance Cost Predict is {insurance_cost}')
    return insurance_cost


# Dự đoán chi phí bảo hiểm của một người 27 tuổi, có chỉ số bmi = 42.13 và có hút thuốc.
calc_insurance(27, 42.13, 'smoker')
