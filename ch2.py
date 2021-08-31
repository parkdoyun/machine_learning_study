# ch2 - 타이타닉 생존자 예측
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
"""
titanic_df = pd.read_csv('./titanic_train.csv') # 파일 가져오기
titanic_df.head(3)

#print('학습 데이터 정보')
#print(titanic_df.info())

# Null 값 없애기 -> 사이킷런 머신러닝에서는 Null 값 허용 X
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True) # 나이 평균값
titanic_df['Cabin'].fillna('N', inplace=True) # 'N'
titanic_df['Embarked'].fillna('N', inplace=True)
#print('데이터 세트 Null 값 개수 ', titanic_df.isnull().sum().sum()) # Null 값 확인

#print('Sex 값 분포 : \n', titanic_df['Sex'].value_counts())
#print('\nCabin 값 분포 : \n', titanic_df['Cabin'].value_counts())
#print('\nEmbarked 값 분포 : \n', titanic_df['Embarked'].value_counts())

# Cabin (선실) -> 방번호보다는 등급이 중요!, 앞 문자만 추출
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]

#print(titanic_df.groupby(['Sex','Survived'])['Survived'].count())

#sns.barplot(x='Sex', y='Survived', data=titanic_df) # 성별과 생존자 관련 그래프
#sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df) #객실 등급별 성별에 따른 생존자 그래프

# 입력(age)에 따라 구분 값 반환하는 함수 설정.
def get_category(age):
    cat = ''
    if age <= -1: cat='Unknown'
    elif age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'

    return cat

#plt.figure(figsize=(10, 6)) # 막대그래프의 figure 더 크게 설정

# X축의 값을 순차적으로 표시하기 위한 설정
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

titanic_df['Age_cat']=titanic_df['Age'].apply(lambda x : get_category(x)) # 'Age' 칼럼 값 받아서 적절한 문자열로 대치한 뒤에 반환
# 그래프 그리기
#sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
titanic_df.drop('Age_cat', axis=1, inplace=True)

# 분석 결과, Sex, Age, PClass 등이 생존 좌우하는 피처임
"""
# 문자열 카테고리를 숫자형 카테고리 피처로 변환
from sklearn import preprocessing
"""
# 여러 칼럼 한번에 변환하는 함수
def encode_features(dataDF):
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder() # make encoder
        le = le.fit(dataDF[feature]) # 기준 정보 설정
        dataDF[feature] = le.transform(dataDF[feature]) # 각 컬럼 변환

    return dataDF

titanic_df = encode_features(titanic_df)
"""
# Null 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True) # 평균
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

# 불필요한 속성 제거 함수
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

# 레이블 인코딩 수행
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1] # 객실 정보는 맨 앞자리만 남김(등급)
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder() # 인코더 생성
        le = le.fit(df[feature]) # 기준 정보 설정
        df[feature] = le.transform(df[feature]) # 변환
    return df

# 데이터 전처리 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

# 원본 데이터 재로딩하고 피처 데이터 세트와 레이블 데이터 세트 추출
titanic_df = pd.read_csv('./titanic_train.csv')
y_titanic_df = titanic_df['Survived'] # 'Survived' 속성 분리
X_titanic_df = titanic_df.drop('Survived', axis=1) # 'Survived' 분리해서 피처 데이터 세트 생성

X_titanic_df = transform_features(X_titanic_df) # 피처 데이터 세트 가공

# 별도의 테스트 데이터 추출
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)

# 결정트리, 랜덤 포레스트, 로지스틱 회귀 사용하여 생존자 예측
# 학습(fit), 예측(predict)
# 예측 성능 평가는 정확도(accuracy)로 함
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
"""
# 결정트리, 랜덤 포레스트, 로지스틱 회귀 위한 사이킷런 Classifier 클래스 생성
dt_clf = DecisionTreeClassifier(random_state=11) # random_state=11은 예제 수행할 때마다 같은 결과 출력하려고! 실제에서는 제거해도 됨
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()

# 결정트리 학습/예측/평가
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
#print('DecisionTreeClassifier 정확도 : {0:.4f}'.format(accuracy_score(y_test, dt_pred)))

# 랜덤 포레스트 학습/예측/평가
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
#print('RandomForestClassifier 정확도 : {0:.4f}'.format(accuracy_score(y_test, rf_pred)))

# 로지스틱 회귀 학습/예측/평가

lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
print('LogisticRegression 정확도 : {0:.4f}'.format(accuracy_score(y_test, lr_pred)))


# 결정트리 교차 검증
from sklearn.model_selection import KFold

def exec_kfold(clf, folds=5):
    # 폴드 세트를 5개인 KFold 객체 생성, 폴드 수만큼 예측 결과 저장 위한 리스트 객체 생성
    kfold = KFold(n_splits=folds)
    scores = []

    # 교차 검증 수행
    for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
        # X_titanic_df 데이터에서 교차 검증별로 학습과 검증 데이터 가리키는 인덱스 생성
        X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]
        # Classifier 학습/예측/정확도 계산
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)
        print('교차 검증 {0} 정확도 :{1:.4f}'.format(iter_count, accuracy))

    # 5개 fold에서의 평균 정확도 계산
    mean_score = np.mean(scores)
    print('평균 정확도 : {0:.4f}'.format(mean_score))

# exec_kfold 호출
exec_kfold(dt_clf, folds=5)

# 교차 검증 cross_val_score 이용해 수행
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt_clf, X_titanic_df, y_titanic_df, cv=5)
for iter_count, accuracy in enumerate(scores):
    print('교차 검증 {0} 정확도 {1:.4f}'.format(iter_count, accuracy))

print('평균 정확도 : {0:.4f}'.format(np.mean(scores))) # kfold와 살짝 다름 -> StratifiedKFold 사용해서 폴드 세트 분할하기 때문

# GridSearchCV 이용해서 최적 하이퍼 파라미터 찾고 예측 성능 측정
# cv는 5개의 폴드 세트, 하이퍼 파라미터는 max_depth, min_samples_split, min_samples_leaf 변경하면서 성능 측정
# 최적 하이퍼 파라미터와 그때의 예측 출력하고 최적 하이퍼 파라미터로 학습된 Estimator 이용해서
# 분리된 테스트 데이터 세트에 예측 수행해 예측 정확도 출력
"""

dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)


parameters = {'max_depth': [2,3,5,10], 'min_samples_split': [2,3,5], 'min_samples_leaf': [1,5,8]}

grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)
grid_dclf.fit(X_train, y_train)

print('GridSearchCV 최적 하이퍼 파라미터 : ', grid_dclf.best_params_)
print('GridSearchCV 최고 정확도 : {0:.4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_ # 최적 하이퍼 파라미터로 학습된 Estimator

# 예측 및 평가 수행
dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dpredictions)
print('테스트 세트에서의 DecisionTreeClassifier 정확도 : {0:.4f}'.format(accuracy))
