# AISecurity-Course Assignment
## Instructions
### step Ⅰ
` download and unzip the file`
### step Ⅱ
` cd <the same folder as main.py>`
### step Ⅲ
` chmod 777 train_test.sh`
### step Ⅳ
`./train_test.sh <lr> <epoches>`

例： ` ./train_test.sh 0.01 10 # 以0.01学习率训练10轮`
## Tuning learning rate 
epoches = 10
In experiment,I train the model for 10 epoches using Adam optimizer.
### lr = 0.01
` 最大准确率为68.00%`
### lr = 0.001
` 最大准确率为72.96%`
### lr = 0.0001
` 最大准确率为70.49%`

We can see that learning rate is not linear dependent with test accuracy.
