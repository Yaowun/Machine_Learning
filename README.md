# Machine_Learning
## 專案介紹
本專案使用 Python 程式語言，利用 Tensorflow Keras 套件對 MNIST 手寫數字進行機器學習辨識。

## 專案技術
Python 3.9、Tesorflow 2.11.0。

## 資料夾說明
* fig - 圖片放置處
* model - 機器學習模型放置處
  * assets - 靜態資源放置處
  * variables - 模型變數放置處

## 實作
### 範例
以下使用 MNIST 的第 0 筆測試資料作為範例
``` python
plt.imshow(data_test_x[0], cmap='Greys')
```
![AI digital recognition](./fig/number_example.png)
``` python
predict = model.predict(data_test_x)
print("Probability:\n", predict[0])
print("Predicted number is:", predict[0].argmax())
```
```
Probability:
[0.0000000e+00 0.0000000e+00 7.0817629e-31 2.0787209e-33 0.0000000e+00
 0.0000000e+00 0.0000000e+00 1.0000000e+00 0.0000000e+00 2.1166898e-18]
Predicted number is: 7
```

### 訓練過程
![machine learning training](./fig/train_history.png)  
將測試資料集丟入模型做評估
``` python
score = model.evaluate(data_test_x, data_test_y, verbose=0)
```
得到的模型評分為 Loss = 0.034 、Accuracy = 99.46 %。
