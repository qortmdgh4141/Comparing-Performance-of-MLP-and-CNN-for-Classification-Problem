# 📉  Comparing Performance of MLP and CNN for Classification Problem
<br/>
  
### 1. &nbsp; Research Objective <br/><br/>

- _The objective of this research is to train MLP-based models and CNN-based models using the Fashion-MNIST dataset and compare the classification performance of these two models. The main hypothesis of this study is as follows:_  <br/>

  - _"The CNN-based model will show superior performance in the classification task of fashion item images compared to the MLP-based model."_ <br/><br/>

- _This hypothesis is based on the following prior background knowledge:_  <br/>

  - _Fashion-MNIST is a dataset consisting of fashion item images, which include various patterns and textures. To handle such complex visual features, it is expected that CNN models, which can effectively utilize spatial information, would be more suitable._ <br/>
  
  - _MLP is an artificial neural network with fully connected hidden layers between input and output. This model is suitable for sequential data processing, but it may have difficulty effectively extracting visual features of fashion item images._ <br/>
  
  - _CNN is a deep learning model specialized in image processing, which handles local information through convolution and pooling operations using shared parameters. Such a structure can better capture the visual features of fashion item images and be more suitable for classification tasks._ <br/><br/>
  
- _Based on this background, we will train MLP-based models and CNN-based models, respectively, and apply them to the classification task of the Fashion-MNIST dataset. It is expected that these results will contribute to the improvement of performance in image classification tasks and aid in model selection in practical applications._ <br/><br/><br/> 

### 2. &nbsp; Key Components of the Neural Network Model and Experimental Settings  <br/><br/>

- _Convolutional Layer_<br/>

  - _Number of filters : (32 or 64)  /  Kernel size: (3 x 3)._ <br/>
  
  - _The convolutional layer extracts local features using filters (kernels) and generates feature maps._ <br/><br/>

- _Pooling Layer_<br/>

  - _Pooling size : (2, 2)._<br/>
  
  - _The pooling layer provides spatial invariance, reduces the size of feature maps to decrease computational complexity, and emphasizes abstracted features._ <br/><br/>

- _Dense Layer_ <br/>

  - _Number of nodes: (512 or 10)._ <br/>

  - _The dense layer is a traditional neural network layer that connects all inputs and outputs. It learns abstract features and outputs probability distribution for various classes._<br/><br/>
  
- _Dropout Layer_ <br/>

  - _The dropout layer is one of the regularization techniques used to reduce overfitting during the neural network training process._ <br/>

  - _Dropout randomly deactivates some units (neurons) of the neural network during training, preventing the model from relying too heavily on specific units and improving generalization capability._<br/><br/>

- _Activation function for hidden layers : ReLU Function_ <br/>

  - _The ReLU function is a non-linear function that outputs 0 for negative input values and keeps the output as is for positive input values._ <br/>

  - _To alleviate the issue of gradient vanishing caused by weight initialization when using ReLU activation function, the weights of the hidden layers were initialized using He initialization._<br/><br/>

- _Activation function for the output layer : Softmax Function_ <br/>

  - _The softmax function is commonly used as the activation function for the output layer in multi-class classification problems._ <br/>

  - _The softmax function normalizes the input values to calculate the probability of belonging to each class, and the sum of probabilities for all classes is 1._<br/><br/>

- _Optimization Algorithm : Adam (Adaptive Moment Estimation)_ <br/>

  - _The Adam optimization algorithm, which combines the advantages of Momentum, which adjusts the learning rate considering the direction of gradients, and RMSProp, which adjusts the learning rate considering the magnitude of gradients, was used._ <br/>

  - _The softmax function normalizes the input values to calculate the probability of belonging to each class, and the sum of probabilities for all classes is 1._<br/><br/>

- _Loss Function : Cross-Entropy Loss Function_ <br/>

  - _When using the softmax function in the output layer, the cross-entropy loss function is commonly used as the loss function._ <br/>

  - _The cross-entropy loss function calculates the error only for the classes corresponding to the actual target values and updates the model in the direction of minimizing the error._<br/><br/>

- _Evaluation Metric : Accuracy_ <br/>

  - _Accuracy is one of the evaluation metrics used to assess the performance of a classification model._ <br/>

  - _Accuracy considers the prediction as correct if it matches the actual target class and calculates it by dividing it by the total number of samples._<br/><br/>

- _Batch Size & Maximum Number of Learning Iterations_ <br/>

  - _In this experiment, the batch size is 128, and the model is trained by iterating up to a maximum of 100 times._<br/>
  
  - _The number of batch size and iterations during training affects the speed and accuracy of the model, and I, as the researcher conducting the experiment, have set the number of batch size and iterations based on my experience of tuning deep learning models._<br/><br/> <br/> 

### 3. &nbsp; Data Preprocessing and Analysis <br/><br/>

- _**Package Settings**_ <br/> 
  
  ```
  from sklearn.datasets import load_diabetes
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import MinMaxScaler, StandardScaler
  from sklearn.metrics import mean_absolute_percentage_error

  from keras import initializers
  from keras.optimizers import Adam
  from keras.models import Sequential
  from keras.layers import Dense, Dropout,BatchNormalization

  import numpy as np
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  from IPython.display import display
  ```

- _**Data Preparation**_ <br/> 
  
  ```
  # 당뇨병 데이터 셋트 로딩 : 입력 데이터(data), 목표 데이터(target)
  diabetes = load_diabetes()

  # 입력 데이터와 목표 데이터를 각각 데이터 프레임으로 변환
  x_data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
  y_data = pd.DataFrame(diabetes.target, columns=['target'])

  # 당뇨병 데이터 셋트에 NaN값이 존재하는지 확인
  if x_data.isnull().values.any() or y_data.isnull().values.any():
      print("- 당뇨병 데이터 셋트에는 NaN값이 존재합니다. -", end= "\n\n")
  else:
      print("- 당뇨병 데이터 셋트에는 NaN값이 존재하지 않습니다. -", end= "\n\n")
  ```
  
  ```
  # 입력 데이터를 출력
  print(f"< 입력 데이터의 구성 : {x_data.shape[0]}행 x {x_data.shape[1]}열 >")
  display(x_data)
  ```
  
  ```
  # 분석 대상에서 제외할 변수인 "age & sex" 열을 삭제  
  x_data = x_data.drop(['age', 'sex'], axis=1) 

  #  분석 대상에서 제외할 변수인 "age & sex" 열을 삭제한 입력 데이터를 출력
  print(f"\n< 'age & sex' 열을 삭제한 입력 데이터의 구성 : {x_data.shape[0]}행 x {x_data.shape[1]}열 >")
  display(x_data)
  ```
  
  ```
  # 목표 데이터를 출력
  print(f"\n< 목표 데이터의 구성 : {y_data.shape[0]}행 x {y_data.shape[1]}열 >")
  display(y_data)
  ```
  
- _**Exploratory Data Analysis (EDA)**_ <br/> 
  
  ```
  # 그래프로 데이터 분포를 파악하기 위해 입출력 데이터를 하나의 테이터 프레임으로 병합
  concat_data = pd.concat([x_data, y_data], axis=1)
  display(concat_data)
  ```

  ```
  # 목표변수인 당뇨병 진행 상태(Diabetes Progression) 값을 10개의 계급으로 하는 밀도그래프를 출력
  # 평균적으로 당뇨병 진행 상태(Diabetes Progression) 값은 100에 많이 분포 
  sns.set(rc={'figure.figsize' : (15, 3)})
  sns.kdeplot(data=concat_data, x='target', shade=True)
  plt.xlabel('Diabetes Progression')
  plt.show()
  ```
  
  <img src="https://github.com/qortmdgh4141/Performance-Optimization-of-MLP-Model-for-Regression-Problem/blob/main/image/density_graph.png?raw=true" height="320">
  
  ```
  # 각 변수 간 상관계수를 히트맵 그래프로 출력 
  # s1 변수와 s2 변수들은 양의 선형적 관계를 가지는 매우 강한 상관관계를 가지고 있음
  # s3 변수와 s4 변수들은 음의 선형적 관계를 가지는 매우 강한 상관관계를 가지고 있음
  corr_matrix = concat_data.corr().round(2)

  sns.set(rc={'figure.figsize' : (8, 5)})
  sns.heatmap(data=corr_matrix, xticklabels=True, annot=True)
  plt.xticks(rotation=0)
  plt.xlabel('\n< Correlation coefficient between each variable >')
  plt.show()
  ```
  
  <img src="https://github.com/qortmdgh4141/Performance-Optimization-of-MLP-Model-for-Regression-Problem/blob/main/image/corr_heatmap_graph.png?raw=true" width="640">
  
  ```
  # 독립 변수 간 매우 강한 상관관계를 가지는 변수가 있는 경우, 다중공선성(multicollinearity) 문제가 발생함 
  # 따라서 변수 선택 기법을 사용하여 상관관계가 높은 변수를 제거
  x_data = x_data.drop(['s2','s3'], axis=1) 
  display(x_data)
  ``` 
  
- _**Splitting Data**_ <br/>  
  
  ```
  # 학습용과 테스트용 데이터를 7:3으로 분리
  x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=20183047)
  x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=20183047)

  print(f"- 학습용 입력 데이터(X) 형상 : {x_train.shape}")
  print(f"- 학습용 정답 데이터(Y) 형상 : {y_train.shape}", end="\n\n")
  print(f"- 검증용 입력 데이터(X) 형상 : {x_val.shape}")
  print(f"- 검증용 정답 데이터(Y) 형상 : {y_val.shape}", end="\n\n") 
  print(f"- 평가용 입력 데이터(X) 형상 : {x_test.shape}")
  print(f"- 평가용 정답 데이터(Y) 형상 : {y_test.shape}"))   
  ```
  
- _**Feature Scaling**_ <br/> 
  
  ```
  # 최솟값은 0, 최댓값은 1이 되도록 데이터에 대해 정규화
  # 최소-최대 정규화 스케일러 생성
  minmax_scalerX = MinMaxScaler()
  minmax_scalerY = MinMaxScaler()

  # 정규화 스케일러를 학습용 데이터에 맞춤
  minmax_scalerX.fit(x_train)
  minmax_scalerY.fit(y_train)

  # 정규화 스케일러로 학습 데이터를 변환
  x_train_minmax = minmax_scalerX.transform(x_train)
  y_train_minmax = minmax_scalerY.transform(y_train)

  # 정규화 스케일러로 검증용 데이터를 변환
  x_val_minmax = minmax_scalerX.transform(x_val)
  y_val_minmax = minmax_scalerY.transform(y_val)

  # 정규화 스케일러로 테스트 데이터를 변환
  x_test_minmax = minmax_scalerX.transform(x_test)
  y_test_minmax = minmax_scalerY.transform(y_test)
  ```
  
  ```
  # 최솟값은 0, 최댓값은 1이 되도록 학습 데이터에 대해 정규화
  # 피처 스케일링 : 학습 데이터의 입력 값
  scalerX = MinMaxScaler()
  scalerX.fit(x_train)
  x_train_norm = scalerX.transform(x_train)

  # 피처 스케일링 : 학습 데이터의 출력 값
  scalerY = MinMaxScaler()
  scalerY.fit(y_train)
  y_train_norm = scalerY.transform(y_train)

  # 피처 스케일링 : 테스트 데이터의 입출력 값
  x_test_norm = scalerX.transform(x_test)
  y_test_norm = scalerY.transform(y_test)
  ```
  <br/> 

### 4. &nbsp; Training and Testing MLP Model <br/><br/>

- _Optimized MLP Model_

  ```
  """
  1. 입출력 노드 : 6개 / 1개
     - 학습 시에 입력 변수의 특성 갯수가 8개이고, 목표 변수 갯수가 1개이기 때문에, 그에 대응하는 입출력 노드로 구성

  2. 은닉층 개수 (노드 수) : 3개 (60, 120, 60)
      - 총 3개의 은닉층이 존재하며, 제 1 은닉층과 제 3 은닉층은 6개의 노드가 존재하고 제 2 은닉층에는 12개의 노드가 존재

  3. 배치 정규화
      - 각 층(layer)을 거칠 때마다 입력 데이터의 분포가 변화함에 따라 학습이 불안정해지는 문제인 내부 공변량(internal covariate shift)를 막기 위해 사용
      - 각 층에서 입력 데이터를 정규화하고, 학습 중에 이에 대한 평균과 분산을 조절하여 입력 데이터의 분포를 안정화 가능

  4. 활성화 함수 :  Relu
     - 입력값이 0보다 작을 경우는 0으로 출력하고, 0보다 큰 경우는 그대로 출력하는 비선형 함수인 Relu 함수로 설정
     - ReLU 활성화 함수를 사용할 때, 가중치 초기화에 따른 그래디언트 소실 문제를 완화하기 위해 은닉층의 가중치는 He 초깃값을 사용

  5. 최적화 알고리즘 
     - Momentum과 RMSProp의 장점을 결합한 최적화 알고리즘인 Adam(Adaptive Moment Estimation)을 사용
     - Momentum은 : 기울기의 방향을 고려하여 학습 속도를 조절 
     - RMSProp : 기울기 크기를 고려하여 학습 속도를 조절

  6. 손실 함수 : 
     - 예측값과 실제값의 차이를 제곱한 값의 평균을 계산함으로써, 
       예측값과 실제값 사이의 오차를 잘 나타내는 MSE(Mean Squared Error)를 사용

  7. 정확도 평가 지표
     - 예측값과 실제값의 백분율 차이의 절대값을 평균하는 MAPE(Mean Absolute Percentage Error)를 사용
       회귀분석에서 가장 일반적으로 사용되는 평가지표 중 상대적인 오차의 크기를 평가하므로, 
       이 평가지표의 오차 값은 예측값과 실제값이 클수록 더 커지는 경향이 있음

  8. 배치 사이즈 / 최대 학습 반복 횟수 : 64 / 1000
  """

  # 모형 구조
  model = Sequential()

  model.add(Dense(60, input_dim=6, activation='relu', kernel_initializer=initializers.HeNormal()))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(120, activation='relu', kernel_initializer=initializers.HeNormal()))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(60, activation='relu', kernel_initializer=initializers.HeNormal()))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(1, bias_initializer=initializers.Constant(value=0.01)))

  model.compile(optimizer=Adam(lr=0.0001), loss='mse')

  results_standard = model.fit(x_train_minmax, y_train_minmax, validation_data=(x_val_minmax, y_val_minmax)
              , epochs=1000, batch_size=64)
  ```

  ```
  # MAPE 값 출력 
  y_pred = model.predict(x_test_minmax)
  y_pred_inverse = minmax_scalerY.inverse_transform(y_pred)

  minmax_mape = mean_absolute_percentage_error(y_test, y_pred_inverse)
  print("MAPE based on min-max normalization : {:.2%}".format(minmax_mape))
  ```
  <br/> 
  
### 5. &nbsp; Research Results  <br/><br/>
    
- _The purpose of this study was to train and evaluate a multilayer perceptron model on the Diabetes 130-US hospitals for years 1999-2008 Data Set. I set the number of hidden racers by referring to the optimal model structure proposed in the paper "Diabetes Mellitus Diagnosis Using Artificial Neural Networks with Fewer Features", increased the model complexity by making the number of nodes in each layer about 10 times larger than the number of nodes set in the paper, and added batch regularization and dropout layers to solve the problems of internal covariate shift and overfitting. In addition, I improved the performance of the existing model by using variable selection techniques, scaling techniques, and initialization techniques to solve multicollinearity problems and to prevent problems such as gradients vanishing._ <br/> <br/>

  ```
  # loss 그래프 출력
  train_loss = results_standard.history['loss']
  val_loss = results_standard.history['val_loss']

  epochs = range(1, len(train_loss) + 1)

  plt.plot(epochs, train_loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.ylim([0,4])
  plt.legend()
  plt.show()
  ```
  
  ```
  # 예측값 대비 실게값의 산포도
  y_pred = model.predict(x_test_minmax)
  diff = np.abs(y_pred - y_test_minmax)

  plt.figure(figsize=(5, 5))
  plt.scatter(y_test_minmax, y_pred, c=diff, cmap='viridis')
  plt.plot([0, 1], [0, 1], c='r')
  plt.xlabel('True Values')
  plt.ylabel('Predictions')
  plt.colorbar()
  plt.show()
  ```
  <br/>
  <img src="https://github.com/qortmdgh4141/Performance-Optimization-of-MLP-Model-for-Regression-Problem/blob/main/image/scatter_plot_line_graph.png?raw=true" weight="1280">

- _The following graph shows the evolution of the loss with increasing epoch, and I can see that overfitting did not occur due to the impact of the aforementioned batch regularization and dropout layer._ <br/>

- _However, it was found that underfitting occurs, where the predictive performance on the training data does not improve because it does not sufficiently reflect the complexity of the data after some learning progress._ <br/>

- _This means that the model has been oversimplified, and I believe that the following reasons contributed to this:_ <br/>

  - _Low Model Complexity_<br/>
  
    - _I believe that underfitting occurs because the model is too simple or limited._<br/>
    
    - _This does not mean that making the current model structure more complex is a good solution. This is because the amount of training data is currently small, and making the model structure more complex is very likely to lead to overfitting._ <br/><br/>
    
  - _Lack of Variable Diversity_<br/>
  
    - _It is determined that underfitting occurred due to a lack of variable diversity in the dataset._<br/>
    
    - _To be more specific, I believe that excessive normalization was applied to a dataset that lacks diversity, causing the model to fail to adequately reflect the patterns in the training data._<br/> <br/> <br/>

### 6. &nbsp; Suggestions for Future Research  <br/><br/>
    
- _Based on the scatter plot of predicted values compared to actual values, it can be seen that the performance of the prediction model is not very accurate. Therefore, to improve the performance of the regression model in the future, the following solutions can be suggested:_

  - _Diversity and Quality of Data_<br/>
  
    - _The unbalanced distribution of data can greatly affect the performance of the model.Therefore, if the data is collected considering the diversity of the data and preprocessed so as not to harm the diversity of the collected data set, the performance of the model will be improved._<br/>

  - _Feature Engineering_<br/>
  
    - _Feature engineering is known to have a significant impact on the performance of a prediction model.Therefore, by extracting more diverse and sophisticated features or introducing new variables, it is expected to improve the performance of the model._<br/> <br/> <br/>
 
--------------------------
### 💻 S/W Development Environment
<p>
  <img src="https://img.shields.io/badge/Windows 10-0078D6?style=flat-square&logo=Windows&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google Colab-black?style=flat-square&logo=Google Colab&logoColor=yellow"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
</p>
<p>
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit learn-blue?style=flat-square&logo=scikitlearn&logoColor=F7931E"/>
  <img src="https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=Numpy&logoColor=blue"/>
</p>

### 🚀 Machine Learning Model
<p>
  <img src="https://img.shields.io/badge/MLP-5C5543?style=flat-square?"/>
</p> 

### 💾 Datasets used in the project
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Diabetes Dataset <br/>
