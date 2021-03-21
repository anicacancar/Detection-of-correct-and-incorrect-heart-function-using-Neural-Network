# Detection-of-correct-and-incorrect-heart-function-using-Neural-Network

Dataset used in this project is called ECG Heartbeat Categorization Dataset from Kaggle:

https://www.kaggle.com/shayanfazeli/heartbeat

We are using the following datasets: ptdb_abnormal.csv and ptdb_normal.csv. The first dataset represents abnormal and the second normal heart function. 

In this project main goal is to make a neural network that after training will be able to recognize normal and abnormal heartbeat. This dataset has 187 features for 14452 samples of data. The neural network will have 187 inputs and 1 output.

# Results

This dataset is not balanced so we are using the f1 score as a rate of performance. 

Parameters of the best neural network for this problem are:

- Number of layers and nodes: [30 30 30 30]
- Activation function: tansig
- Coefficient of regularization: 0.3

For this type of neural network, we get f1_score = 0.97126, which are very good results.

For more details check out the report in the files above.

