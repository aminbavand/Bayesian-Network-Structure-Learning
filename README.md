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


<p align="center">
  <img width="460" height="300" src="/images/1.png">
</p>

<p align="center">
  Figure 1: The structure of a naïve Bayes network [1]
</p>


In classification problems, we have a matrix of raw data used for training our classifier. In case of Bayesian network classifiers, we have to build our network with the available dataset. The first step is to find the best structure describing the connection between variables and the second step is to find the exact probabilistic relationship between connected variables. If we assume that the first step is done, it is not so hard to perform the second step, meaning finding the probabilistic relationship between connected variables. Because, that can directly be inferred from the dataset. The most important challenge is to find the best structure. As there are a very large number of possible structures for the network, it is nearly impossible to consider every possible structure separately, evaluate them based on the dataset and see which one can describe the dataset the best. Instantly, that lead us to think of heuristic methods.


There are many heuristic search methods presented in the literature like random search, particle swarm optimization (PSO), genetic algorithm (GA), etc. In particular, GA is one of the mostly used heuristic methods in different problems. In most cases, GA is able to find global optima or at least a local optimum closely enough to the global optimum. It is also fast enough and easy to implement. All these reasons add together and make GA one of the most popular heuristic searches. In this project, we choose GA to find the best structures in the Bayesian networks. We search the space of all possible structures through GA to find the optimum graph for our dataset. The dataset that we use is Asia or sometimes called lung cancer dataset [2].


In section 2, we describe the dataset that we use in this project and then bring brief explanation about our methods, which are Bayesian network and genetic algorithm and explain how we are going to use these tools in our problem. In section 3, we bring the result of our method on the dataset. Finally, in section 4 we have a discussion about the results and how compatible they are with the expected knowledge-based result. 

## Materials and Methods

In this section, we first provide the detail of our used dataset in this project. Then we briefly discuss about the context of Bayesian networks and Genetic algorithm and how to use them in our dataset.

### Dataset


As we mentioned in section 1, we us Asia dataset for this study. The dataset is available in R package “bnlearn” [2]. It consists of eight binary variables (nodes in Bayesian network) named and described as follows:

    1. “Visit to Asia”: Whether or not the patient has visited Asia recently.
    2. “Tuberculosis”: Whether or not the patient has tuberculosis.
    3. “Smoke”: Whether the patient smokes or not.
    4. “Lung Cancer”: Whether or not the patient has lung cancer.
    5. “Bronchitis”: Whether or not the patient has bronchitis.
    6. “Either tuberculosis or lung cancer” Whether or not the patient has either one of tuberculosis or lung cancer.
    7. “Positive X-ray”: Whether or not the result of chest X-ray on the patient is positive.
    8. “Dyspnoea”: Whether or not the patient suffers from dyspnea.



The number of data instances is 10000. Therefore, we have a 10000*8 matrix of zero and one elements. In order to have better insights over the dataset and the possible connections between its variables, let us look at some medical facts about the relation of these eight variables together.
“Shortness of breath (dyspnea) may be due to tuberculosis, lung cancer or bronchitis, or none of them, or more than one of them. A recent visit to Asia increases the chances of tuberculosis, while smoking in known to be a risk factor for both lung cancer and bronchitis. The results of a single chest X-ray do not discriminate between lung cancer and tuberculosis, as neither does the presence of absence of dyspnea.” [3]



In order to better understand how the connection between variables can help us, let us look at a simple example from [3]. If a patient has dyspnoea and recently visited Asia and the smoking history and X-ray results would not be available, then how much is the chance that patient has one of the mentioned disease. And would knowing one of the missed information help us in knowing other variables? 



In this project, we are going to construct a network that shows the relation between these variables only by using the dataset and see whether or not that structure satisfies the above explanations about Asia dataset.


### Bayesian Network and Its Structure Learning

As we have explained in the introduction, Bayesian network is a powerful classification tool that can be applied to many problems. To see how it works, we will explain it with a simple example of three variables A, B, and C. Suppose A and B are binary and both are directly connected to C with an arrow pointed to C. Then the Bayesian network containing these three variable with their probability distributions is presented in Figure 2:



<p align="center">
  <img width="460" height="300" src="/images/2.png">
</p>

<p align="center">
  Figure 2: A simple Bayesian network illustration
</p>



In this network, we call variables A and B, the parents for variable C, as they directly affect the variable C. Since A and B have no parents, their probability distributions only contain the states of these variables. For example, in 60% of the times, variable A takes 0 and in 40% of the times, in takes 1. These probabilities directly comes from the dataset. For variables like A, which have no parents, their probability distributions come from that column of dataset related to variable A, without considering any other variable. However, in case of variable like C, which have some parents, their probability distributions would be conditional to their parents and we call that conditional probability distribution (CPD). In CPDs, there are one probability distribution for the variable, for every combination of its parents. As in Figure 2, since A and B each can take two values, then all possible combinations of these variables are 4 and for all these 4 combinations, we would have a probability distribution for variable C, which by the way takes 3 values. Now we explain how we can fill the CPD for variable C. Suppose we want to fill the first row. First, we consider all the rows of the dataset, which their corresponding columns for variables A and B be both zeros. Then in all that rows, the number of rows which their C is 1 divided by all the rows with A=0 and B=0 would be the probability of variable C being at state 1 given A and B are both in state zero.


Above, we explained how to fill the CPDs for a known structure and given dataset. That is a straightforward task and does not seem so hard. The most important task is to find the best structure that is matched with the dataset. If we assume that the network is connected (there is at least one path between every two variables in the graph), even for the simple network of Figure 2, there are 18 possible valid structures in total. Valid structure means a connected graph without any cycle, meaning there are no path from one node back to itself. For networks with larger number of variables, the total number of possible structures grows very fast and it is practically impossible to check all the structures. So, the first task would be to find the best structure describing the dataset. But how can we cay one structure is better than the other. There should be some evaluation metrics for this matter. There are many evaluation metrics for structure of Bayesian networks. In [4], some of them are presented. Bayesian Dirichlet (BD), Bayesian Dirichlet equivalence (BDe), K2, log-likelihood (LL), and Minimum description length/ Bayesian information criterion (MIC/BIC) are some of most used metrics. In this project, we use BIC metric to evaluate the structures for two reasons. BIC is one of the most used methods in literature, so it has been validated so many times before and we can be sure of its accuracy. BIC contains two terms in it. First term is the log-likelihood of the data, which in simple words give us intuition about what is the likelihood of the dataset given this structure. Of course if we have a structure in which all variables are directly connected to each other, whatever the data would be, the likelihood of that would be probably too much, because every connection exist in this network. However, log-likelihood metric does not consider the redundant edges at all. So, we have to add another term to penalize the structure if it has some connections that should not exist.


The added term is  -d*N/2 which d is the number of independent parameters in the structure and N is the total number of data instances. The details are available in [5]. I added another penalty term to the BIC score to penalize those structures which are not connected. For every individual in the population, after checking the connected status of the graph, if it was not connected, a very large negative amount would be added to the BIC score. This way, the non-connected graphs would have a very small chance of being selected as the parents of the next generation, so they would be removed in the selection process.



Now that we have a metric to evaluate the structure of Bayesian network based on the dataset, we can think of finding a way of constructing the structure that gives us the best performance metric (here, BIC). As we explained in section 1, since there are a very large number of structures, we should think of heuristic search methods and in this project, GA is our method of interest. In section 2.3, we introduce genetic algorithm and in section 2.4, we explain how we are going to use that on our dataset.


### Genetic Algorithm



Genetic algorithm is a metaheuristic search method that tries to find the optimum solution in the search space. It starts with an initial random population of possible answers in the search space and tries to make some changes in the population by combining them together and also making some changes individually to lead the population in the direction which an objective function (fitness) would be improved. The procedure of GA is as follows. After generating an initial population, it chooses some individuals of this population (individuals with better fitness have higher probability of being chosen) to be parents for the next generation. This is called selection. After that each of these parents mutate with a small probability. Mutation is a little change in the structure of an answer. Next step is crossover which two parents swap some genes between them. Now we have the next generation. Note that also we transfer the best answer of each generation without any change to the next generation. This way, we can make sure that the best solution found so far, would not be destroyed. We repeat this process for multiple times until for some consecutive steps, the best answer does not change. This way we know that GA has converged to an answer, which could be either local, or the global optimum answer.



### Performing Genetic Algorithm for Bayesian network


In case of our problem, we want to find a Bayesian network structure for Asia dataset which BIC score would be maximum for that structure. The steps in the procedure are as follows:

1. Phenotype genotype mapping:

The first thing we have to do is find a vectorized
representation of Bayesian network that uniquely maps every structure to a vectorized form.
Assume we have n variables named X_1, X_2,..., X_n in our network. We can represent the network
with the set of parents for each variable. If the set of parents for variable X_i are represented by
Pa(X_i) , then the set of all parent sets for all variables {Pa(X_1),...,Pa(X_n)} would be a unique
representation of that network [6]. We are able to use this representation and perform operations
on it.


2. Initialization:

We randomly generate a set of populations of the form {Pa(X_1),...,Pa(X_n)}
considering there should not be any cycle in the graph. One important point in the construction
of initial population is that how many parent one node should have in average. If we do not have
any presumption over the average number of parents per node, then some of the constructed
networks might have very little number of edges in the network and even the network might not
be connected. I used a Gaussian distribution with mean equal to 1.2 times the number of nodes
and with variance 0.2 times the number of nodes and use this distribution as the distribution of
the total number of edges in the network.


3. Selection:

We can assign a probability for each individual of the population, which is the
probability of that individual being chosen for the next generation parent. As we said, this
probability should be higher for those who have better fitness than the others. In the case of
Bayesian network in this project, our metric is BIC score and the greater the BIC score, the better
that structure is for the dataset. So, we can calculate BIC score for all the population and subtract
the minimum of them from all other scores and then add a small constant value to all of them (so
there won’t be any zero in the values). Now if we normalized these values so that sum of them
would be 1, then we have a probability distribution over all individuals in the population and this
probability is more for those individuals which have better BIC score. Now, we choose some of
the individuals based on these probability distributions and use them as the parents for the next
generation to perform mutation and crossover on them.



4. Mutation: 

Mutation in our problem would be a little change in the network, which can be
either adding or removing one edge to/from the graph. We have to be careful about producing
graphs that have cycle, so we have to check after each mutation that if the new graph has cycle in
it, redo the operation. Note that mutation does not happened to all individuals in population, but
there should be a constant small probability that determines whether the mutation should happen
for every individual or not. Also, with 50% probability, the type of mutation is removing an edge
and with 50% probability, the type is adding an edge. The added edge should be uniformly
chosen from the available ones.


5. Crossover: 

Generally, k point crossover between two parents means that we set k points in
random places in string and start from the beginning of strings. String have divided into k+1
substrings. Starting from the first area, we swap this area between two parents. In the next area,
we do not do anything, but in the one after that and generally in all areas that have even distance
with first are, we swap the elements of that area between two parents. In the case of our problem,
we use one point crossover. If we have two networks represent with <img src="https://render.githubusercontent.com/render/math?math={Pa_i(X_1),...,Pa_i(X_n)}"> and <img src="https://render.githubusercontent.com/render/math?math={Pa_j(X_1),...,Pa_j(X_n)}"> after performing crossover with swapping point X_k , the new generated
networks would be as follows: 

1. <img src="https://render.githubusercontent.com/render/math?math={Pa_i(X_1),...,Pa_i(X_k), Pa_j(X_{k+1}),...,Pa_j(X_n)}">



2. <img src="https://render.githubusercontent.com/render/math?math={Pa_j(X_1),...,Pa_j(X_k), Pa_i(X_{k+1}),...,Pa_i(X_n)}">





Note that the swapping point X_k is chosen randomly between all possible points. After
performing crossover, we have to check for existence of cycles in the generated graphs.
We started with an initial population of size 200 and ran the algorithm for 50 iterations. After
that there were no improvement in the results. After setting the probability of mutation to several
values, the value of 0.2 seemed to make the algorithm able to find the optimum answers in faster
time, so we set it to 0.2.

We have explained all the necessary materials that are used in performing genetic algorithm on
Bayesian networks. Now it is time to jump and see the result of our method to the dataset.


## Results

After performing genetic algorithm to find the best structure describing Asia dataset, using BIC
score, the top three achieved network are shown in Figure 3, Figure 4, and Figure 5.



<p align="center">
  <img width="460" height="300" src="/images/3.png">
</p>

<p align="center">
  Figure 3: The best achieved structure with BIC score = -26492.4
</p>



<p align="center">
  <img width="460" height="300" src="/images/4.png">
</p>

<p align="center">
  Figure 4: The second best achieved structure with BIC score = -26710.9
</p>



<p align="center">
  <img width="460" height="300" src="/images/5.png">
</p>

<p align="center">
  Figure 5: The third best achieved structure with BIC score = -27049.9
</p>









