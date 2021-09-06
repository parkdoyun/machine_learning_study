# 3장 피마 인디언 당뇨병 예측

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LogisticRegression

diabetes_data = pd.read_csv('./diabetes.csv')
#print(diabetes_data['Outcome'].value_counts()) # negative : 500, positive : 268
#print(diabetes_data.head(3))
#print(diabetes_data.info()) # Null 값 없으며 피처의 타입 모두 숫자 -> 인코딩 필요 X

# 피처 데이터 세트 X, 레이블 데이터 세트 y 추출
# 맨 끝이 Outcome 칼럼으로 레이블 값, 칼럼 위치 -1 이용해 추출
X = diabetes_data.iloc[:, :-1] # 처음부터 마지막 열 전까지
y = diabetes_data.iloc[:, -1] # 마지막 열

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156, stratify=y)

# 학습, 예측 및 평가 수행
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train) # 학습
pred = lr_clf.predict(X_test) # 예측
pred_proba = lr_clf.predict_proba(X_test)[:, 1] # 평가

def get_clf_eval(y_test, pred=None, pred_proba=None): # 성능 평가 지표 출력 함수
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)

    print('오차 행렬')
    print(confusion)

    print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현율 : {2:.4f}, F1 : {3:.4f}, AUC : {4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))

def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    # thresholds list 객체 내의 값 차례로 iteration하면서 평가 수행
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print('임곗값 : ', custom_threshold)
        get_clf_eval(y_test, custom_predict)

def precision_recall_curve_plot(y_test, pred_proba_c1):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

    # X축을 threshold, Y축은 정밀도, 재현율 값으로, 정밀도는 점선 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision') # 정밀도
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall') # 재현율

    # threshold 값 X축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    # X축, Y축 label과 legend, grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()



#get_clf_eval(y_test, pred, pred_proba) # 전체 데이터의 65%가 Negative이므로 정확도보다는 재현율에 초점!

pred_proba_c1 = lr_clf.predict_proba(X_test)[:, 1]
#precision_recall_curve_plot(y_test, pred_proba_c1) # 정밀도 재현율 곡선 보기
# 재현율 곡선 보면 임곗값 0.42정도로 낮추면 균형 맞춤, 하지만 두개 지표 모두 0.7 안 됨 -> 임곗값 인위적 조작 필요

#print(diabetes_data.describe()) # 피처 값 분포도 확인 -> min 값 0이 많다

# 0값 검사
# 0값 검사할 피처명 테스트
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# 전체 데이터 건수
total_count = diabetes_data['Glucose'].count()

# 피처별로 반복하면서 데이터 값이 0인 데이터 건수 추출하고, 퍼센트 계산
for feature in zero_features:
    zero_count = diabetes_data[diabetes_data[feature] == 0][feature].count()
    #print('{0} 0 건수는 {1}, 퍼센트는 {2:.2f} %'.format(feature, zero_count, 100*zero_count/total_count))
# -> SkinThickness, Insulin 0값 많다, 전체 데이터 수가 많지 않아서 삭제하면 학습 효과적으로 수행 어려움 -> 평균값으로 대체

# zero_features 리스트 내부에 저장된 개별 피처들에 대해서 0값 평균값으로 대체
mean_zero_features = diabetes_data[zero_features].mean()
diabetes_data[zero_features]=diabetes_data[zero_features].replace(0, mean_zero_features)

# 0값을 평균값으로 대체한 데이터 세트에 피처 스케일링 적용해 변환
# 로지스틱 회귀의 경우, 일반적으로 숫자 데이터에 스케일링 적용하는 것이 좋다

X = diabetes_data.iloc[:, :-1] # 피처 : 나머지
y = diabetes_data.iloc[:, -1] # 레이블 정보 : 맨 끝 칼럼

# 일괄적으로 피처 데이터 세트(X)에 스케일링 적용
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=156, stratify=y)

# 로지스틱 회귀로 학습, 예측 및 평가 수행
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

#get_clf_eval(y_test, pred, pred_proba) # 데이터 변환과 스케일링 통해 성능 개선됐으나, 재현율 수치 개선 필요
# -> 분류 결정 임곗값 변화시키면서 재현율 수치가 어느 정도 개선되는지 확인 -> 0.48이 가장 적절
thresholds = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.50]
pred_proba = lr_clf.predict_proba(X_test)
# get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1, 1), thresholds) # 에러, get_clf_eval 함수를 내부적으로 호출할 때 인수 적어서 그런듯

# 임곗값 변환해서 다시 예측 -> predict()는 마음대로 변환 X, 별도의 로직으로 구현해야 함
# predict_proba로 추출한 예측 결과 확률 값을 변환해서 변경된 임곗값에 따른 예측 클래스 값 구한다

# 임곗값 0.48로 설정한 Binarizer 생성
binarizer = Binarizer(threshold=0.48)

# 위에서 구한 lr_clf의 predict_proba() 예측 확률 array에서 1에 해당하는 칼럼 값 Binarizer로 변환
pred_threshold_048 = binarizer.fit_transform(pred_proba[:, 1].reshape(-1, 1))

get_clf_eval(y_test, pred_threshold_048, pred_proba[:, 1])

