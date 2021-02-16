# Bayesian Network Structure Learning with Genetic Algorithm

## Abstract
Bayesian networks structure learning has been always in the focus of researchers. There are many approaches presented for this matter. Genetic algorithm is an 
effective approach in problems facing with a large number of possible answers. In this study, we perform genetic algorithm on Asia dataset to find a graph that 
describes the dataset in the best way. Asia dataset contains some medical information about patients and the goal is finding relation between these information,
so it would be easier for doctors to predict the existence of some disease in patients. We used BIC score to evaluate the structures using the dataset. Comparing
the final achieved structures with the one that is achieved with medical knowledge, there were some differences between them, which might be because of improper
scoring function or improper knowledge-based structure. All in all, genetic algorithm succeeded in optimizing the fitness function in a reasonable time.

## Introduction
The problem of classification is one of the most important topics in the current world. There are hundreds of machine learning approaches that are applied to 
different problems. Bayesian networks is a very powerful tool for classification problems. They use the dataset to find the probabilistic relationship between 
different variables of the dataset. One of the most import benefits of Bayesian networks is that they can give us more accurate estimate of the probability of 
variables being at different states as the number of observed variables increased. Normally, in classification problems, we face a number of inputs and a number
of outputs predicted based on input values. However, in Bayesian networks, we have more degrees of freedom in the selection of inputs and outputs. We know that 
the Bayesian network gives us probabilistic connections between the variables. Therefore, if we have value of some of the variables, we will have a probabilistic
prediction of other variables. As the number of observed variables increases, the prediction of other variables would be more accurate. Figure 1 shows a simple 
naïve Bayesian network structure. In naïve structures, one of the variables is directly connected to all others. 

[Figure 1: The structure of a naïve Bayes network [1]](/images/1.png)
