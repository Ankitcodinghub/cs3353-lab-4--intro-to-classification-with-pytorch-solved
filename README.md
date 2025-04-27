# cs3353-lab-4--intro-to-classification-with-pytorch-solved
**TO GET THIS SOLUTION VISIT:** [CS3353 Lab 4- Intro to Classification with PyTorch Solved](https://www.ankitcodinghub.com/product/aiml-cs-335-solved-3/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;121100&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS3353 Lab 4- Intro to Classification with PyTorch Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
Lab 4: Intro to Classification with PyTorch

Important: Please read the instructions mentioned in the questions carefully. We have provided boilerplate code for each question. Please ensure that you make changes in the areas marked with TODO.

Please read the comments in the code carefully for detailed information regarding input and output format.

1 Classification

1.1 Dataset

You have been given synthetic dataset for classification. The dataset contains 2000 data points, each of which has 80 features. the shape of the dataset is 2000* 8, where the last column indicates the output label (since its a binary classification problem output is 0/1).

You will have to read the given dataset.csv file ,convert it to a pandas dataframe and then split the entire dataset into training and validation data as per the split ratio for evaluation purpose.

1.2 Task

Complete the following functions:

• def __init__(self, args, input_dim):

This is the constructor. It takes as input the args, input_dim variables. Use the args variable to specify all the hyperparameters of the model, and the neural layers as per your discretion.

• def forward(self, data):

This implements the forward pass of the algorithm.

Input data is batched data of shape BATCH_SIZE * num_features

Returns predictions of shape BATCH_SIZE * 1

• def loss(self, pred, label): This implements the loss function. The inputs pred and label, are each tensors of shape BATCH_SIZE*1 . Pred contains model prediction outputs for each of the batched inputs. Label contains 0/1 values only. Function returns

a single loss value. We will be implementing different variations of the loss function as described in the next section.

• def evaluate(loader, model): This implements the evaluation function for binary classification. Input loader is of “torch.data_utils.DataLoader” type. It is an ordered pair where the first item is the feature matrix of shape BATCH_SIZE * num_features, and the second item is the label tensor of shape BATCH_SIZE * 1. Function returns an evaluation score based on the following definition: it will be ratio of the number of samples which are correctly classified/total number of sample points in that batch

We will be be implementing the three loss functions as specified next. While executions, we need to specify the type of loss function using the args variable “model_type”.

1.2.1 Negative Log Likelihood (NLL)

The Binary Cross Entropy Function is defined as follows, for a set of N data points:

Here, for the ith data point: ti is the true label (0 for class 0 and 1 for class 1) and pi is the predicted probability of the point belonging to class 1.

When the observation belongs to class 1 the first part of the formula becomes active and the second part vanishes, and vice versa in the case observation’s actual class are 0. This is how we calculate the binary cross-entropy.

The probabiltiy scores for the forward pass can be computed using a Sigmoid function as follows:

z is the score of the item x as given by the neural model.

The sigmoid function outputs a S(z) ∈ [0,1] and indicates the probability of how close to a class the item belongs (in the case of binary classification). Therefore, having a threshold 0.5, the binary classification output Class(x) can be formulated as

(

1 if S(z) &gt; 0.5 Class(x) =

0 otherwise

Your model will output a sigmoid score for each input. Subsequently, these real valued predictions will be converted to binary labels using Class(x) function. Finally, the accuracy is computed no. of label matches

using the binary predictions and binary labels, and is defined as total no. of items in both labels

1.2.2 SVM Loss

For ith data point, the Hinge Loss Function is given by:

(1 − y (wT xi + b) if 1 − yi(wT xi + b)) &gt; 0 i

LHL(xi,yi) =

0 otherwise

where, yi is the truth label for the corresponding input instance xi and parameter w. Here: w,xi ∈Rn, b ∈R, yi ∈{−1,1}

Note: you will have to change the output labels as (1,-1) rather than (1,0) as given in the dataset before calculating the hinge loss.

Here, LHL can take any float value depending on the sign of yi and wT xi. If both signs are same (indicating correct class prediction), LHL = 0. And if the signs are opposite (indicating misclassification), value of LHL increases. In other words, it finds the classification boundary that guarantees the maximum margin between the data points of the different classes.

Hence during evaluation, you need to calculate the number of items on each side of the classification boundary w.r.t to the actual classes they belong to. The formula is given as follows:

(1 if yi(wT xi + b) ≥ 0

Class(xi) =

−1 otherwise

As before, the final accuracy is computed using the binary predictions of Class(x) and binary no. of label matches

labels, and is defined as

total no. of items in both labels

1.2.3 Ranking Loss

The Ranking Loss is defined as follows (recap from earlier lab):

L(P,N) = X max(0,n − p)

p∈P,n∈N

Here, P denotes the model predictions for the set of positive items labelled 1. N denotes the model predictions for the set of negative items labelled 0. We want to impose the constraint that the positive scores should be greater than the negative scores. This is enforced by the above functions.

Your model will output a score for each input. During Evaluation, we will count the number of times positive items are scored higher than the negative ones. Higher the count, better is the model. The counting formula is specified as below:

L(P,N) = X 1[p &gt; n]

p∈P,n∈N

The final evaluation score is the output of the counting formula.

2 Other Resources:

We have provided the following resources to help you train your models:

2.1 Training

def train(args, Xtrain, Ytrain, Xval, Yval, model)

Use this as a black box function to train your code. The inputs are the args variable, training labels and features, validation features and labels, and the model object. Do not change any part of this function.

2.2 Visualization

def plot(val_accs, losses):

Use this function to plot the training loss, and validation accuracy across epochs.

3 Assessment

We will be checking the following:

• End-to-End working of the ML model

• correctness of the loss function implementations • correctness of the evaluation function implementations

• execution time of the all functions.

Make sure to avoid for loops in the loss functions.

4 Submission instructions

Complete the functions in assignment.py. Make changes only in the places mentioned in comments. Do not modify the function signatures. Keep the file in a folder named &lt;ROLL_NUMBER&gt;_L4 and compress it to a tar file named &lt;ROLL_NUMBER&gt;_L4.tar.gz using the command

tar -zcvf &lt;ROLL_NUMBER&gt;_L4.tar.gz &lt;ROLL_NUMBER&gt;_L4

Submit the tar file on Moodle. The directory structure should be –

&lt;ROLL_NUMBER&gt;_L4

| – – – – assignment.py

Replace ROLL_NUMBER with your own roll number. If your Roll number has alphabets, they should be in “small” letters.
