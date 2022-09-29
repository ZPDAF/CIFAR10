# CIFAR hyperparameter tune
## Instructions
### first step
` cd <main.py所在文件夹>`
### second step
` chmod 777 train_test.sh`
### third step
`./train_test.sh <学习率> <epochs>`

例： ` ./train_test.sh 0.01 10 # 以0.01学习率训练10轮`
## Tune learning rate 
epochs = 10
### 0.01
` 最大准确率为68.00%`
### 0.001
` 最大准确率为72.96%`
### 0.0001
` 最大准确率为70.49%`

