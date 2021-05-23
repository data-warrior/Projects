# Databricks notebook source
# MAGIC %md # HW 5 - Page Rank
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Fall 2018`__
# MAGIC 
# MAGIC In Weeks 8 and 9 you discussed key concepts related to graph based algorithms and implemented SSSP.   
# MAGIC In this final homework assignment you'll implement distributed PageRank using some data from Wikipedia.
# MAGIC By the end of this homework you should be able to:  
# MAGIC * ... __compare/contrast__ adjacency matrices and lists as representations of graphs for parallel computation.
# MAGIC * ... __explain__ the goal of the PageRank algorithm using the concept of an infinite Random Walk.
# MAGIC * ... __define__ a Markov chain including the conditions underwhich it will converge.
# MAGIC * ... __identify__ what modifications must be made to the web graph inorder to leverage Markov Chains.
# MAGIC * ... __implement__ distributed PageRank in Spark.
# MAGIC 
# MAGIC __Please refer to the `README` for homework submission instructions and additional resources.__ 

# COMMAND ----------

# MAGIC %md # Notebook Set-Up
# MAGIC Before starting your homework run the following cells to confirm your setup.   

# COMMAND ----------

# imports
import re
import ast
import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md ### Run the next cell to create your directory in dbfs
# MAGIC You do not need to understand this scala snippet. It simply dynamically fetches your user directory name so that any files you write can be saved in your own directory.

# COMMAND ----------

# RUN THIS CELL AS IS
# This code snippet reads the user directory name, and stores is in a python variable.
# Next, it creates a folder inside your home folder, which you will use for files which you save inside this notebook.
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
print(userhome)
hw5_path = userhome + "/HW5/" 
hw5_path_open = '/dbfs' + hw5_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(hw5_path)

# COMMAND ----------

# RUN THIS CELL AS IS. 
tot = 0
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
for item in dbutils.fs.ls(DATA_PATH):
  tot = tot+item.size
tot
# ~4.7GB

# COMMAND ----------

# RUN THIS CELL AS IS. You should see all-pages-indexed-in.txt, all-pages-indexed-out.txt and indices.txt in the results. If you do not see these, please let an Instructor or TA know.
display(dbutils.fs.ls(DATA_PATH))

# COMMAND ----------

# RUN THIS CELL AS IS - A test to make sure your directory is working as expected.
# You should see a result like:
# dbfs:/user/youremail@ischool.berkeley.edu/HW5/test.txt
dbutils.fs.put(hw5_path+'test.txt',"hello world",True)
display(dbutils.fs.ls(hw5_path))


# COMMAND ----------

sc = spark.sparkContext
spark

# COMMAND ----------

# MAGIC %md # Question 1: Distributed Graph Processing
# MAGIC Chapter 5 from Lin & Dyer gave you a high level introduction to graph algorithms and concernts that come up when trying to perform distributed computations over them. The questions below are designed to make sure you captured the key points from this reading and your async lectures. 
# MAGIC 
# MAGIC ### Q1 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ Give an example of a dataset that would be appropriate to represent as a graph. What are the nodes/edges in this dataset? Is the graph you describe 'directed' or 'undirected'? What would the average "in-degree" of a node mean in the context of your example? 
# MAGIC 
# MAGIC * __b) short response:__ Other than their size/scale, what makes graphs uniquely challenging to work with in the map-reduce paradigm? *(__HINT__: Do not respond in terms of any specific algorithm. Think in terms of the nature of the graph datastructure itself).*
# MAGIC 
# MAGIC * __c) short response:__ Briefly describe Dijskra's algorithm (goal/approach). What specific design component makes this approach hard to parallelize?
# MAGIC 
# MAGIC * __d) short response:__ How does parallel breadth-first-search get around the problem that you identified in part `c`? At what expense?

# COMMAND ----------

# MAGIC %md ### Q1 Student Answers:
# MAGIC > __a)__ Type your answer here!
# MAGIC 
# MAGIC > __b)__ Type your answer here!
# MAGIC 
# MAGIC > __c)__ Type your answer here!  
# MAGIC 
# MAGIC > __d)__ Type your answer here!  

# COMMAND ----------

# MAGIC %md ### <--- SOLUTION --->
# MAGIC __SOLUTION__
# MAGIC > __a)__ Answers will vary but should clearly identify an example with nodes/edges, it's directed-ness and in-degree (eg. people on a social media site who are 'friends' with other people on that site, undirected, average indegree = average size of someone's network) 2) 
# MAGIC 
# MAGIC > __b)__ Because of their structure, it can be hard to represent graphs as distinct records that can be processed separately from each other -- in particular information about one node's edges is inherently also information about other nodes.
# MAGIC 
# MAGIC > __c)__ Dijsktra's algorithm is a way to find the single source shortest path for a *weighted* graph. It works by maintaining a priority queue while it traverses the graph (starting at the source node). At each step, it loads neighboring nodes into the priority queue in the order of their total distance from the source (replacing longer path distances with shorter ones as appropriate). It then choses which node to visit next based on what is at the front of the queue. By the time the queue is empty all shortest paths have been found. Unfortunately for a large graph the size of this priority queue (which has to be held in memory on all nodes) becomes a problem if we wanted to parallelize the algorithm.
# MAGIC 
# MAGIC > __d)__ Parallel BFS conceptually works similarly to Dijsktras in that it sequentially visits neighboring nodes of previously visited nodes and computes total distances based on edge weights. However instead of loading the resulting distances into a priority queue of nodes "to be visited" it encodes a node state on a per-record basis -- which effectively allows the information about which nodes are 'enqueued' to exist in parallel. However there is a cost with this design choice - namely that in order to sequentially visit nodes in order of their edges parallel BFS must make multiply passes over the entire data set. 

# COMMAND ----------

# MAGIC %md # Question 2: Representing Graphs 
# MAGIC 
# MAGIC In class you saw examples of adjacency matrix and adjacency list representations of graphs. These data structures were probably familiar from HW3, though we hadn't before talked about them in the context of graphs. In this question we'll discuss some of the tradeoffs associated with these representations. __`NOTE:`__ We'll use the graph from Figure 5.1 in Lin & Dyer as a toy example. For convenience in the code below we'll label the nodes `A`, `B`, `C`, `D`, and `E` instead of $n_1$, $n_2$, etc but otherwise you should be able to follow along & check our answers against those in the text.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/HW5/Lin-Dyer-graph-Q1.png?raw=true" width=50%>
# MAGIC 
# MAGIC ### Q2 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ Relatively speaking, is the graph you described in Figure 5.1 in Lin & Dyer "sparse" or "dense"?  Explain how sparsity/density impacts the adjacency matrix and adjacency list representations of a graph.
# MAGIC 
# MAGIC * __b) short response:__ Run the provided code to create and plot our toy graph. Is this graph directed or undirected? Explain how the adjacency matrices for directed graphs will differ from those of undirected graphs.
# MAGIC 
# MAGIC * __c) code:__ Fill in the missing code to complete the function `get_adj_matr()`.
# MAGIC 
# MAGIC * __d) code:__ Fill in the missing code to complete the function `get_adj_list()`.

# COMMAND ----------

# MAGIC %md ### Q2 Student Answers:
# MAGIC > __a)__ Type your answer here!
# MAGIC 
# MAGIC > __b)__ Type your answer here!

# COMMAND ----------

# MAGIC %md ### <--- SOLUTION --->
# MAGIC __SOLUTION__
# MAGIC > __a)__ This is a relatively sparse matrix. If self loops are allowed, then we could have a total of N^2 edges, or 5^2 = 25. Otherwise, we could have N(N-1) = 20 edges. Out of a total possible 20 or 25 edges, this graph only has 9. For sparse graphs, their adjacency list representation will be much more (memory) efficient than their adjacency matrix (because storing 0's is a use of space without encoding any information).
# MAGIC 
# MAGIC > __b)__ This is a directed graph. We can see this because we see the edge `('A', 'B')` in the graph but no corresponding `('B','A')`. For directed graphs the adjacency matrix will not be symmetric wherease for undirected graphs it will be symmetric.

# COMMAND ----------

# part a - a graph is just a list of nodes and edges (RUN THIS CELL AS IS)
TOY_GRAPH = {'nodes':['A', 'B', 'C', 'D', 'E'],
             'edges':[('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'D'), 
                      ('D', 'E'), ('E', 'A'),('E', 'B'), ('E', 'C')]}

# COMMAND ----------

# part a - simple visualization of our toy graph using nx (RUN THIS CELL AS IS)
G = nx.DiGraph()
G.add_nodes_from(TOY_GRAPH['nodes'])
G.add_edges_from(TOY_GRAPH['edges'])
display(nx.draw(G, pos=nx.circular_layout(G), with_labels=True, alpha = 0.5))

# COMMAND ----------

# part c - adjacency matrix function
def get_adj_matr(graph):
    """
    Function to create an adjacency matrix representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        pd.DataFrame with entry i,j representing an edge from node i to node j
    """
    n = len(graph['nodes'])
    adj_matr = pd.DataFrame(0, columns = graph['nodes'], index = graph['nodes'])
    ############### YOUR CODE HERE ##################
    
    ############### (END) YOUR CODE #################
    return adj_matr

# COMMAND ----------

# <--- SOLUTION --->
# part c - adjacency matrix function
def get_adj_matr(graph):
    """
    Function to create an adjacency matrix representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        pd.DataFrame with entry i,j representing an edge from node i to node j
    """
    n = len(graph['nodes'])
    adj_matr = pd.DataFrame(0, columns = graph['nodes'], index = graph['nodes'])
    ############### YOUR CODE HERE ##################
    for edge in graph['edges']:
        in_, out_ = edge
        adj_matr[out_][in_] = 1
    ############### (END) YOUR CODE #################
    return adj_matr

# COMMAND ----------

# part c - take a look (RUN THIS CELL AS IS)
TOY_ADJ_MATR = get_adj_matr(TOY_GRAPH)
print(TOY_ADJ_MATR)

# COMMAND ----------

# part d - adjacency list function
def get_adj_list(graph):
    """
    Function to create an adjacency list representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        dictionary of the form {node : [list of edges]}
    """
    adj_list = {node: [] for node in graph['nodes']}
    ############### YOUR CODE HERE ##################

    
    ############### (END) YOUR CODE #################
    return adj_list

# COMMAND ----------

# <--- SOLUTION --->
# part d - adjacency list function
def get_adj_list(graph):
    """
    Function to create an adjacency list representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        dictionary of the form {node : [list of edges]}
    """
    adj_list = {node: [] for node in graph['nodes']}
    ############### YOUR CODE HERE ##################
    for in_, out in graph['edges']:
        adj_list[in_].append(out)
    ############### (END) YOUR CODE #################
    return adj_list

# COMMAND ----------

# part d - take a look (RUN THIS CELL AS IS)
TOY_ADJ_LIST = get_adj_list(TOY_GRAPH)
print(TOY_ADJ_LIST)

# COMMAND ----------

# MAGIC %md # Question 3: Markov Chains and Random Walks
# MAGIC 
# MAGIC As you know from your readings and in class discussions, the PageRank algorithm takes advantage of the machinery of Markov Chains to compute the relative importance of a webpage using the hyperlink structure of the web (we'll refer to this as the 'web-graph'). A Markov Chain is a discrete-time stochastic process. The stochastic matrix has a principal left eigen vector corresponding to its largest eigen value which is one. A Markov chain's probability distribution over its states may be viewed as a probability vector. This steady state probability for a state is the PageRank of the corresponding webpage. In this question we'll briefly discuss a few concepts that are key to understanding the math behind PageRank. 
# MAGIC 
# MAGIC ### Q3 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ It is common to explain PageRank using the analogy of a web surfer who clicks on links at random ad infinitum. In the context of this hypothetical infinite random walk, what does the PageRank metric measure/represent?
# MAGIC 
# MAGIC * __b) short response:__ What is the "Markov Property" and what does it mean in the context of PageRank?
# MAGIC 
# MAGIC * __c) short response:__ A Markov chain consists of $n$ states plus an $n\times n$ transition probability matrix. In the context of PageRank & a random walk over the WebGraph what are the $n$ states? what implications does this have about the size of the transition matrix?
# MAGIC 
# MAGIC * __d) code + short response:__ What is a "right stochastic matrix"? Fill in the code below to compute the transition matrix for the toy graph from question 2. [__`HINT:`__ _It should be right stochastic. Using numpy this calculation can be done in one line of code._]
# MAGIC 
# MAGIC * __e) code + short response:__ To compute the stable state distribution (i.e. PageRank) of a "nice" graph we can apply the power iteration method - repeatedly multiplying the transition matrix by itself, until the values no longer change. Apply this strategy to your transition matrix from `part d` to find the PageRank for each of the pages in your toy graph. Your code should print the results of each iteration. How many iterations does it take to converge? Which node is most 'central' (i.e. highest ranked)? Does this match your intuition? 
# MAGIC     * __`NOTE 1:`__ _this is a naive approach, we'll unpack what it means to be "nice" in the next question_.
# MAGIC     * __`NOTE 2:`__ _no need to implement a stopping criteria, visual inspection should suffice_.

# COMMAND ----------

# MAGIC %md ### Q3 Student Answers:
# MAGIC > __a)__ Type your answer here!
# MAGIC 
# MAGIC > __b)__ Type your answer here!
# MAGIC 
# MAGIC > __c)__ Type your answer here! 
# MAGIC 
# MAGIC > __d)__ Type your answer here!
# MAGIC 
# MAGIC > __e)__ Type your answer here! 

# COMMAND ----------

# MAGIC %md ### <--- SOLUTION --->
# MAGIC __SOLUTION__
# MAGIC 
# MAGIC > __a)__ PageRank measures the relative importance of a webpage based on how connected that page is to other pages. In the analogy of the random surfer we can think of PageRank as the proportion of time the surfer will spend at a given page. More precisely, PageRank is a probability distribution over nodes in the graph representing the likelihood that a random walk over the link structure will arrive at a particular node. In the context of web search PageRank is a way to compare web pages in terms of their credibility or relative importance which is a useful way to sort query results.
# MAGIC 
# MAGIC 
# MAGIC > __b)__ The markov property is the property of ‘memorylessness’… in the context of Page Rank this is the assumption that the probability of transitioning from one page to another is stable regardless of past browsing history.
# MAGIC 
# MAGIC > __c)__ The states are the webpages — at any given time a random walker can only be at one page. This means the transition matrix will be huge because ’n’ will be the number of webpages on the internet
# MAGIC 
# MAGIC 
# MAGIC > __d)__ A "right stochastic matrix" is a real square matrix with each row summing to 1.
# MAGIC 
# MAGIC 
# MAGIC > __e)__ The steady state coverges by the 7th iteration. Node 'E' is highest ranked and yes, it does seem most central.

# COMMAND ----------

# part d - recall what the adjacency matrix looked like (RUN THIS CELL AS IS)
TOY_ADJ_MATR

# COMMAND ----------

# part d - use TOY_ADJ_MATR to create a right stochastic transition matrix for this graph
################ YOUR CODE HERE #################
transition_matrix = None # replace with your code

################ (END) YOUR CODE #################
print(transition_matrix)

# COMMAND ----------

# <--- SOLUTION --->
# part d - use TOY_ADJ_MATR to create a right stochastic transition matrix for this graph
################ YOUR CODE HERE #################
transition_matrix = np.array(TOY_ADJ_MATR) / np.array(TOY_ADJ_MATR).sum(axis=1, keepdims=True)
################ (END) YOUR CODE #################
print(transition_matrix)

# COMMAND ----------

# part e - compute the steady state using the transition matrix 
def power_iteration(xInit, tMatrix, nIter, verbose = True):
    """
    Function to perform the specified number of power iteration steps to 
    compute the steady state probability distribution for the given
    transition matrix.
    
    Args:
        xInit     - (n x 1 array) representing inial state
        tMatrix  - (n x n array) transition probabilities
        nIter     - (int) number of iterations
    Returns:
        state_vector - (n x 1 array) representing probability 
                        distribution over states after nSteps.
    
    NOTE: if the 'verbose' flag is on, your function should print the step
    number and the current matrix at each iteration.
    """
    state_vector = None
    ################ YOUR CODE HERE #################

    
    
    
    ################ (END) YOUR CODE #################
    return state_vector

# COMMAND ----------

# <--- SOLUTION --->
# part e - compute the steady state using the transition matrix 
def power_iteration(xInit, tMatrix, nIter, verbose = True):
    """
    Function to perform the specified number of power iteration steps to 
    compute the steady state probability distribution for the given
    transition matrix.
    
    Args:
        xInit     - (n x 1 array) representing inial state
        tMatrix  - (n x n array) transition probabilities
        nIter     - (int) number of iterations
    Returns:
        state_vector - (n x 1 array) representing probability 
                        distribution over states after nSteps.
    
    NOTE: if the 'verbose' flag is on, your function should print the step
    number and the current matrix at each iteration.
    """
    state_vector = None
    ################ YOUR CODE HERE #################
    state_vector = xInit.dot(tMatrix)
    for ix in range(nIter):    
        if verbose:
            print(f'Step {ix}:\n {state_vector}')
        tMatrix = tMatrix.dot(tMatrix)
        state_vector = xInit.dot(tMatrix)
    ################ (END) YOUR CODE #################
    return state_vector

# COMMAND ----------

# part e - run 10 steps of the power_iteration (RUN THIS CELL AS IS)
xInit = np.array([1.0, 0, 0, 0, 0]) # note that this initial state will not affect the convergence states
states = power_iteration(xInit, transition_matrix, 10, verbose = True)

# COMMAND ----------

# MAGIC %md __`Expected Output for part e:`__  
# MAGIC >Steady State Probabilities:
# MAGIC ```
# MAGIC Node A: 0.10526316  
# MAGIC Node B: 0.15789474  
# MAGIC Node C: 0.18421053  
# MAGIC Node D: 0.23684211  
# MAGIC Node E: 0.31578947  
# MAGIC ```

# COMMAND ----------

# MAGIC %md # Question 4: Page Rank Theory
# MAGIC 
# MAGIC Seems easy right? Unfortunately applying this power iteration method directly to the web-graph actually runs into a few problems. In this question we'll tease apart what we meant by a 'nice graph' in Question 3 and highlight key modifications we'll have to make to the web-graph when performing PageRank. To start, we'll look at what goes wrong when we try to repeat our strategy from question 3 on a 'not nice' graph.
# MAGIC 
# MAGIC __`Additional References:`__ http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html
# MAGIC 
# MAGIC ### Q4 Tasks:
# MAGIC 
# MAGIC * __a) code + short response:__ Run the provided code to create and plot our 'not nice' graph. Fill in the missing code to compute its transition matrix & run the power iteration method from question 3. What is wrong with what you see? [__`HINT:`__ _there is a visible underlying reason that it isn't converging... try adding up the probabilities in the state vector after each iteration._]
# MAGIC 
# MAGIC * __b) short response:__  Identify the dangling node in this 'not nice' graph and explain how this node causes the problem you described in 'a'. How could we modify the transition matrix after each iteration to prevent this problem?
# MAGIC 
# MAGIC * __c) short response:__ What does it mean for a graph to be irreducible? Is the webgraph naturally irreducible? Explain your reasoning briefly.
# MAGIC 
# MAGIC * __d) short response:__ What does it mean for a graph to be aperiodic? Is the webgraph naturally aperiodic? Explain your reasoning briefly.
# MAGIC 
# MAGIC * __e) short response:__ What modification to the webgraph does PageRank make in order to guarantee aperiodicity and irreducibility? Interpret this modification in terms of our random surfer analogy.

# COMMAND ----------

# MAGIC %md ### Q4 Student Answers:
# MAGIC > __a)__ Type your answer here!
# MAGIC 
# MAGIC > __b)__ Type your answer here!
# MAGIC 
# MAGIC > __c)__ Type your answer here!
# MAGIC 
# MAGIC > __d)__ Type your answer here!  
# MAGIC 
# MAGIC > __e)__ Type your answer here!  

# COMMAND ----------

# MAGIC %md ### <--- SOLUTION --->
# MAGIC __SOLUTION__
# MAGIC 
# MAGIC > __a)__ Each column should represent a probability distribution but these don't sum to 1. In fact the more iterations we run, the lower their sum is.
# MAGIC 
# MAGIC > __b)__ A dangling node is a node with inlinks but no outlinks. We need to redistribute the dangling mass to maintain stochasticity.
# MAGIC   
# MAGIC > __c)__ _irriducibility_: There must be a sequence of tansitions of non-zero probability from any state to any other. In other words, the graph has to be connected (a path exists from all vetices to all vertices). No the webgraph will have disconnected segments.
# MAGIC 
# MAGIC > __d)__ _aperiodicity_: States are not partitioined into sets such that all state transitions occur cyclicly from one set to another. Yes. All we need is a single page with a link back to itself to make the web graph aperiodic. An anchor link is one such link.
# MAGIC 
# MAGIC > __e)__ To compensate for the lack of these properties, Page and Brin introduced the Random Surfer model, whereby a damping factor as well as a teleportation factor are applied to each node to account for sinks and sources, or in other words, dangling nodes and disconnected components.

# COMMAND ----------

# part a - run this code to create a second toy graph (RUN THIS CELL AS IS)
TOY2_GRAPH = {'nodes':['A', 'B', 'C', 'D', 'E'],
              'edges':[('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'D'), 
                       ('B', 'E'), ('C', 'A'), ('C', 'E'), ('D', 'B')]}

# COMMAND ----------

# part a - simple visualization of our test graph using nx (RUN THIS CELL AS IS)
G = nx.DiGraph()
G.add_nodes_from(TOY2_GRAPH['nodes'])
G.add_edges_from(TOY2_GRAPH['edges'])
display(nx.draw(G, pos=nx.circular_layout(G), with_labels=True, alpha = 0.5))

# COMMAND ----------

# part a - run 10 steps of the power iteration method here
# HINT: feel free to use the functions get_adj_matr() and power_iteration() you wrote above
################ YOUR CODE HERE #################




################ (END) YOUR CODE #################

# COMMAND ----------

# <--- SOLUTION --->
# part a - run 10 steps of the power iteration method here
# HINT: feel free to use the functions get_adj_matr() and power_iteration() you wrote above
################ YOUR CODE HERE #################
adj_matr = np.array(get_adj_matr(TOY2_GRAPH))
trans_matr = np.zeros(adj_matr.shape)
for idx, row in enumerate(adj_matr):
  if np.sum(row) > 0.0:
        trans_matr[idx] = np.array(row) / np.sum(row)
xInit = np.array([1.0,0,0,0,0])
state = power_iteration(xInit, trans_matr, 10)
################ (END) YOUR CODE #################

# COMMAND ----------

# MAGIC %md # About the Data
# MAGIC The main dataset for this data consists of a subset of a 500GB dataset released by AWS in 2009. The data includes the source and metadata for all of the Wikimedia wikis. You can read more here: 
# MAGIC > https://aws.amazon.com/blogs/aws/new-public-data-set-wikipedia-xml-data. 
# MAGIC 
# MAGIC As in previous homeworks we'll be using a 2GB subset of this data, which is available to you in this dropbox folder: 
# MAGIC > https://www.dropbox.com/sh/2c0k5adwz36lkcw/AAAAKsjQfF9uHfv-X9mCqr9wa?dl=0. 
# MAGIC 
# MAGIC Use the cells below to download the wikipedia data and a test file for use in developing your PageRank implementation(note that we'll use the 'indexed out' version of the graph) and to take a look at the files.

# COMMAND ----------

dbutils.fs.ls(DATA_PATH)

# COMMAND ----------

# open test_graph.txt file to see format (RUN THIS CELL AS IS)
with open('/dbfs/mnt/mids-w261/HW5/test_graph.txt', "r") as f_read:
  for line in f_read:
    print(line)

# COMMAND ----------

# load the data into Spark RDDs for convenience of use later (RUN THIS CELL AS IS)
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
testRDD = sc.textFile(DATA_PATH +'test_graph.txt')
indexRDD = sc.textFile(DATA_PATH + '/indices.txt')
wikiRDD = sc.textFile(DATA_PATH + '/all-pages-indexed-out.txt')

# COMMAND ----------

# display testRDD (RUN THIS CELL AS IS)
testRDD.take(10)

# COMMAND ----------

# display indexRDD (RUN THIS CELL AS IS)
indexRDD.take(10)

# COMMAND ----------

# display wikiRDD (RUN THIS CELL AS IS)
wikiRDD.take(10)

# COMMAND ----------

# MAGIC %md # Question 5: EDA part 1 (number of nodes)
# MAGIC 
# MAGIC As usual, before we dive in to the main analysis, we'll peform some exploratory data anlysis to understand our dataset. Please use the test graph that you downloaded to test all your code before running the full dataset.
# MAGIC 
# MAGIC ### Q5 Tasks:
# MAGIC * __a) short response:__ In what format is the raw data? What does the first value represent? What does the second part of each line represent? [__`HINT:`__ _no need to go digging here, just visually inspect the outputs of the head commands that we ran after loading the data above._]
# MAGIC 
# MAGIC * __b) code + short response:__ Run the provided bash command to count the number of records in the raw dataset. Explain why this is _not_ the same as the number of total nodes in the graph.
# MAGIC 
# MAGIC * __c) code:__ In the space provided below write a Spark job to count the _total number_ of nodes in this graph. 
# MAGIC 
# MAGIC * __d) short response:__ How many dangling nodes are there in this wikipedia graph? [__`HINT:`__ _you should not need any code to answer this question._]

# COMMAND ----------

# MAGIC %md ### Q5 Student Answers:
# MAGIC > __a)__ Type your answer here!  
# MAGIC 
# MAGIC > __b)__ Type your answer here! 
# MAGIC 
# MAGIC > __d)__ Type your answer here!  

# COMMAND ----------

# MAGIC %md ### <--- SOLUTION --->
# MAGIC __SOLUTION__
# MAGIC > __a)__ This data structure is called an 'adjacency list'. The first value is the node-id (webpage number) the second is a dictionary of linked pages (neighbor nodes)  and the number of times that page is linked.
# MAGIC 
# MAGIC > __b)__ Webpages (i.e. nodes) that don't have any hyperlinks (i.e. out-edges) won't have a record in this raw representation of the graph.
# MAGIC 
# MAGIC > __d)__ 9410987

# COMMAND ----------

# part b - count the number of records in the raw data (RUN THIS CELL AS IS)
# 5781290
print(wikiRDD.count())

# COMMAND ----------

# part c - write your Spark job here (compute total number of nodes)
def count_nodes(dataRDD):
    """
    Spark job to count the total number of nodes.
    Returns: integer count 
    """    
    ############## YOUR CODE HERE ###############

    
    
    
    ############## (END) YOUR CODE ###############   
    return totalCount

# COMMAND ----------

# <--- SOLUTION --->
# part c - write your Spark job here (compute total number of nodes)
def count_nodes(dataRDD):
    """
    Spark job to count the total number of nodes.
    Returns: integer count 
    """    
    ############## YOUR CODE HERE ###############
    def get_node_ids(line):
        node, edges = line.split('\t')
        edges = ast.literal_eval(edges)
        return [str(node)] + list(edges.keys())
    
    totalCount = dataRDD.flatMap(get_node_ids).distinct().count()
    ############## (END) YOUR CODE ###############   
    return totalCount

# COMMAND ----------

# part c - run your counting job on the test file (RUN THIS CELL AS IS)
start = time.time()
tot = count_nodes(testRDD)
print(f'... completed job in {time.time() - start} seconds.')
print(f'Total Nodes: {tot}')

# COMMAND ----------

# part c - run your counting job on the full file (RUN THIS CELL AS IS)
start = time.time()
tot = count_nodes(wikiRDD)
print(f'... completed job in {time.time() - start} seconds.')
print(f'Total Nodes: {tot}')

# COMMAND ----------

# <--- SOLUTION --->
# part e - number of dangling nodes
15192277 - 5781290

# COMMAND ----------

# <--- SOLUTION --->
# part e - number of dangling nodes

num_dangling = 15192277 - 5781290

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

data = [5781290, num_dangling]
labels = ["regular", "dangling"]

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n{:d}".format(pct, absolute)

wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))
ax.legend(wedges, labels,
          title="Node Type",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=12, weight="bold")
ax.set_title("Ratio of dangling nodes")
display(plt.show())

# COMMAND ----------

# MAGIC %md # Question 6 - EDA part 2 (out-degree distribution)
# MAGIC 
# MAGIC As you've seen in previous homeworks the computational complexity of an implementation depends not only on the number of records in the original dataset but also on the number of records we create and shuffle in our intermediate representation of the data. The number of intermediate records required to update PageRank is related to the number of edges in the graph. In this question you'll compute the average number of hyperlinks on each page in this data and visualize a distribution for these counts (the out-degree of the nodes). 
# MAGIC 
# MAGIC ### Q6 Tasks:
# MAGIC * __a) code:__ In the space provided below write a Spark job to stream over the data and compute all of the following information:
# MAGIC  * count the out-degree of each non-dangling node and return the names of the top 10 pages with the most hyperlinks
# MAGIC  * find the average out-degree for all non-dangling nodes in the graph
# MAGIC  * take a 1000 point sample of these out-degree counts and plot a histogram of the result. 
# MAGIC  
# MAGIC  
# MAGIC * __b) short response:__ In the context of the PageRank algorithm, how is information about a node's out degree used?
# MAGIC 
# MAGIC * __c) short response:__ What does it mean if a node's out-degree is 0? In PageRank how will we handle these nodes differently than others?
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md ### Q6 Student Answers:
# MAGIC 
# MAGIC > __b)__ Type your answer here! 
# MAGIC 
# MAGIC > __c)__ Type your answer here! 

# COMMAND ----------

# MAGIC %md ### <--- SOLUTION --->
# MAGIC __SOLUTION__
# MAGIC 
# MAGIC > __b)__ The outdegree is used in calculating each node's "contribution" to its neighbors upon each iteration/update step. Specifically we take the node's current rank, divide it by the out-degree and then redistribute that partial sum to each neighbor to get added up.
# MAGIC 
# MAGIC > __b)__ Nodes with out-degree 0 are dangling nodes also called 'sinks'. Without modification their mass doesn't get redistributed which is a problem for the 'stochasticity' requirement for a Markov chain to converge. To avoide this problem we have to accumulate the mass from these dangling nodes & redistribute it evenly across the rest of the graph.

# COMMAND ----------

# part a - write your Spark job here (compute average in-degree, etc)
def count_degree(dataRDD, n):
    """
    Function to analyze out-degree of nodes in a a graph.
    Returns: 
        top  - (list of 10 tuples) nodes with most edges
        avgDegree - (float) average out-degree for non-dangling nodes
        sampledCounts - (list of integers) out-degree for n randomly sampled non-dangling nodes
    """
    # helper func
    def parse(line):
        node, edges = line.split('\t')
        return (node, ast.literal_eval(edges))
    
    ############## YOUR CODE HERE ###############

    
    
    
    
    ############## (END) YOUR CODE ###############
    
    return top, avgDegree, sampledCounts

# COMMAND ----------

# <--- SOLUTION --->
# part a - write your Spark job here (compute average in-degree, etc)
def count_degree(dataRDD, n):
    """
    Function to analyze out-degree of nodes in a a graph.
    Returns: 
        top  - (list of 10 tuples) nodes with most edges
        avgDegree - (float) average out-degree for non-dangling nodes
        sampledCounts - (list of integers) out-degree for n randomly sampled non-dangling nodes
    """
    # helper func
    def parse(line):
        node, edges = line.split('\t')
        return (node, ast.literal_eval(edges))
    
    ############## YOUR CODE HERE ###############
    degree = dataRDD.map(parse) \
                    .mapValues(lambda x: np.sum(list(x.values()))) \
                    .cache()
    
    top = degree.takeOrdered(10, key=lambda x: -x[1])
    avgDegree = degree.map(lambda x: x[1]).mean()
    sampledCounts = degree.map(lambda x: x[1]).takeSample(False, n)
    ############## (END) YOUR CODE ###############
    
    return top, avgDegree, sampledCounts

# COMMAND ----------

# part a - run your job on the test file (RUN THIS CELL AS IS)
start = time.time()
test_results = count_degree(testRDD,10)
print(f"... completed job in {time.time() - start} seconds")
print("Average out-degree: ", test_results[1])
print("Top 10 nodes (by out-degree:)\n", test_results[0])

# COMMAND ----------

# part a - plot results from test file (RUN THIS CELL AS IS)
plt.hist(test_results[2], bins=10)
plt.title("Distribution of Out-Degree")
display(plt.show())

# COMMAND ----------

# part a - run your job on the full file (RUN THIS CELL AS IS)
start = time.time()
full_results = count_degree(wikiRDD,1000)

print(f"... completed job in {time.time() - start} seconds")
print("Average out-degree: ", full_results[1])
print("Top 10 nodes (by out-degree:)\n", full_results[0])

# COMMAND ----------

# part a - plot results from full file (RUN THIS CELL AS IS)
plt.hist(full_results[2], bins=50)
plt.title("Distribution of Out-Degree")
display(plt.show())

# COMMAND ----------

# MAGIC %md # Question 7 - PageRank part 1 (Initialize the Graph)
# MAGIC 
# MAGIC One of the challenges of performing distributed graph computation is that you must pass the entire graph structure through each iteration of your algorithm. As usual, we seek to design our computation so that as much work as possible can be done using the contents of a single record. In the case of PageRank, we'll need each record to include a node, its list of neighbors and its (current) rank. In this question you'll initialize the graph by creating a record for each dangling node and by setting the initial rank to 1/N for all nodes. 
# MAGIC 
# MAGIC __`NOTE:`__ Your solution should _not_ hard code \\(N\\).
# MAGIC 
# MAGIC ### Q7 Tasks:
# MAGIC * __a) short response:__ What is \\(N\\)? Use the analogy of the infinite random web-surfer to explain why we'll initialize each node's rank to \\(\frac{1}{N}\\). (i.e. what is the probabilistic interpretation of this choice?)
# MAGIC 
# MAGIC * __b) short response:__ Will it be more efficient to compute \\(N\\) before initializing records for each dangling node or after? Explain your reasoning.
# MAGIC 
# MAGIC * __c) code:__ Fill in the missing code below to create a Spark job that:
# MAGIC   * parses each input record
# MAGIC   * creates a new record for any dangling nodes and sets it list of neighbors to be an empty set
# MAGIC   * initializes a rank of 1/N for each node
# MAGIC   * returns a pair RDD with records in the format specified by the docstring
# MAGIC 
# MAGIC 
# MAGIC * __d) code:__ Run the provided code to confirm that your job in `part a` has a record for each node and that your should records match the format specified in the docstring and the count should match what you computed in question 5. [__`TIP:`__ _you might want to take a moment to write out what the expected output should be fore the test graph, this will help you know your code works as expected_]
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md ### Q7 Student Answers:
# MAGIC 
# MAGIC > __a)__ Type your answer here! 
# MAGIC 
# MAGIC > __b)__ Type your answer here! 

# COMMAND ----------

# MAGIC %md ### <--- SOLUTION --->
# MAGIC __SOLUTION__
# MAGIC 
# MAGIC > __a)__ N is the total number of nodes in the graph. Setting the initial proability ot 1/N is like saying that the random surfer is equally likely to start her random walk anywhere on the graph.
# MAGIC 
# MAGIC > __b)__ If we wait to compute N after initializing records then we can avoid a second shuffle, however in this case it might in fact be more efficient to use distinct() to quickly find potential dangling nodes and compute n before initializing the bulky dictionaries.

# COMMAND ----------

# part c - job to initialize the graph (RUN THIS CELL AS IS)
def initGraph(dataRDD):
    """
    Spark job to read in the raw data and initialize an 
    adjacency list representation with a record for each
    node (including dangling nodes).
    
    Returns: 
        graphRDD -  a pair RDD of (node_id , (score, edges))
        
    NOTE: The score should be a float, but you may want to be 
    strategic about how format the edges... there are a few 
    options that can work. Make sure that whatever you choose
    is sufficient for Question 8 where you'll run PageRank.
    """
    ############## YOUR CODE HERE ###############

    # write any helper functions here
    
    
    
    
    
    
    
    
    
    # write your main Spark code here
    
    
    
    
    
    ############## (END) YOUR CODE ##############
    
    return graphRDD

# COMMAND ----------

# <--- SOLUTION --->
# part c - job to initialize the graph (RUN THIS CELL AS IS)
def initGraph(dataRDD):
    """
    Spark job to read in the raw data and initialize an 
    adjacency list representation with a record for each
    node (including dangling nodes).
    
    Returns: 
        graphRDD -  a pair RDD of (node_id , (score, edges))
        
    NOTE: The score should be a float, but you may want to be 
    strategic about how format the edges... there are a few 
    options that can work. Make sure that whatever you choose
    is sufficient for Question 8 where you'll run PageRank.
    """
    ############## YOUR CODE HERE ###############
    
    # helper function
    def parse(line):
        """
        Helper function to identify potential danglers and
        write edges as a csv string for efficient aggregation.
        """
        node, edges = line.split('\t')
        edge_string = ''       
        for edge, count in ast.literal_eval(edges).items():
            # emit potential danglers w/ empty string
            yield (int(edge), '')
            # add this edge to our string of edges
            if edge_string:
                edge_string += ','
            edge_string += ','.join([edge] * int(count))
        # finally yield this node w/ its string formatted edge list
        yield (int(node), edge_string)
    
    # main Spark code
    initRDD = dataRDD.flatMap(parse).reduceByKey(lambda a, b: a + b) 
    N = sc.broadcast(initRDD.count())
    graphRDD = initRDD.mapValues(lambda x: (1/float(N.value), x)).cache()

    ############## (END) YOUR CODE ###############
    
    return graphRDD

# COMMAND ----------

# part c - run your Spark job on the test graph (RUN THIS CELL AS IS)
start = time.time()
testGraph = initGraph(testRDD).collect()
print(f'... test graph initialized in {time.time() - start} seconds.')
testGraph

# COMMAND ----------

# part c - run your code on the main graph (RUN THIS CELL AS IS)
start = time.time()
wikiGraphRDD = initGraph(wikiRDD)
print(f'... full graph initialized in {time.time() - start} seconds')

# COMMAND ----------

# part c - confirm record format and count (RUN THIS CELL AS IS)
start = time.time()
print(f'Total number of records: {wikiGraphRDD.count()}')
print(f'First record: {wikiGraphRDD.take(1)}')
print(f'... initialization continued: {time.time() - start} seconds')

# COMMAND ----------

# MAGIC %md # Question 8 - PageRank part 2 (Iterate until convergence)
# MAGIC 
# MAGIC Finally we're ready to compute the page rank. In this last question you'll write a Spark job that iterates over the initialized graph updating each nodes score until it reaches a convergence threshold. The diagram below gives a visual overview of the process using a 5 node toy graph. Pay particular attention to what happens to the dangling mass at each iteration.
# MAGIC 
# MAGIC <img src='https://github.com/kyleiwaniec/w261_assets/blob/master/images/HW5/PR-illustrated.png?raw=true' width=50%>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC __`A Note about Notation:`__ The formula above describes how to compute the updated page rank for a node in the graph. The $P$ on the left hand side of the equation is the new score, and the $P$ on the right hand side of the equation represents the accumulated mass that was re-distributed from all of that node's in-links. Finally, $|G|$ is the number of nodes in the graph (which we've elsewhere refered to as $N$).
# MAGIC 
# MAGIC ### Q8 Tasks:
# MAGIC * __a) short response:__ In terms of the infinite random walk analogy, interpret the meaning of the first term in the PageRank calculation: $\alpha * \frac{1}{|G|}$
# MAGIC 
# MAGIC * __b) short response:__ In the equation for the PageRank calculation above what does $m$ represent and why do we divide it by $|G|$?
# MAGIC 
# MAGIC * __c) short response:__ Keeping track of the total probability mass after each update is a good way to confirm that your algorithm is on track. How much should the total mass be after each iteration?
# MAGIC 
# MAGIC * __d) code:__ Fill in the missing code below to create a Spark job that take the initialized graph as its input then iterates over the graph and for each pass:
# MAGIC   * reads in each record and redistributes the node's current score to each of its neighbors
# MAGIC   * uses an accumulator to add up the dangling node mass and redistribute it among all the nodes. (_Don't forget to reset this accumulator after each iteration!_)
# MAGIC   * uses an accumulator to keep track of the total mass being redistributed.( _This is just for your own check, its not part of the PageRank calculation. Don't forget to reset this accumulator after each iteration._)
# MAGIC   * aggregates these partial scores for each node
# MAGIC   * applies telportation and damping factors as described in the formula above.
# MAGIC   * combine all of the above to compute the PageRank as described by the formula above.
# MAGIC   * 
# MAGIC   
# MAGIC    __WARNING:__ Some pages contain multiple hyperlinks to the same destination, please take this into account when redistributing the mass.
# MAGIC 
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md ### Q8 Student Answers:
# MAGIC 
# MAGIC > __a)__ Type your answer here!
# MAGIC 
# MAGIC > __b)__ Type your answer here! 
# MAGIC 
# MAGIC > __c)__ Type your answer here! 

# COMMAND ----------

# MAGIC %md ### <--- SOLUTION --->
# MAGIC __SOLUTION__
# MAGIC 
# MAGIC > __a)__ That first term is the probability of randomly jumping to this node... \\(\alpha\\) is the teleportation factor (i.e. the probability of a random jump), and we divide it by \\(|G|\\) because if the surfer does jump randomly there are \\(|G|\\) equally likely nodes he could land on.
# MAGIC 
# MAGIC > __b)__ \\(m\\) is the dangling mass which we also distribute to all \\(|G|\\) nodes equally.
# MAGIC 
# MAGIC > __c)__ The total mass should add to 1 at all times.

# COMMAND ----------

# part d - provided FloatAccumulator class (RUN THIS CELL AS IS)

from pyspark.accumulators import AccumulatorParam

class FloatAccumulatorParam(AccumulatorParam):
    """
    Custom accumulator for use in page rank to keep track of various masses.
    
    IMPORTANT: accumulators should only be called inside actions to avoid duplication.
    We stringly recommend you use the 'foreach' action in your implementation below.
    """
    def zero(self, value):
        return value
    def addInPlace(self, val1, val2):
        return val1 + val2

# COMMAND ----------

# part d - job to run PageRank (RUN THIS CELL AS IS)
def runPageRank(graphInitRDD, alpha = 0.15, maxIter = 10, verbose = True):
    """
    Spark job to implement page rank
    Args: 
        graphInitRDD  - pair RDD of (node_id , (score, edges))
        alpha         - (float) teleportation factor
        maxIter       - (int) stopping criteria (number of iterations)
        verbose       - (bool) option to print logging info after each iteration
    Returns:
        steadyStateRDD - pair RDD of (node_id, pageRank)
    """
    # teleportation:
    a = sc.broadcast(alpha)
    
    # damping factor:
    d = sc.broadcast(1-a.value)
    
    # initialize accumulators for dangling mass & total mass
    mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    
    ############## YOUR CODE HERE ###############
    
    # write your helper functions here, 
    # please document the purpose of each clearly 
    # for reference, the master solution has 5 helper functions.


            
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    # write your main Spark Job here (including the for loop to iterate)
    # for reference, the master solution is 21 lines including comments & whitespace

    
    
    
    
    
    
    
    
    
    
    ############## (END) YOUR CODE ###############
    
    return steadyStateRDD

# COMMAND ----------

# <--- SOLUTION --->
# part d - job to run PageRank (RUN THIS CELL AS IS)
def runPageRank(graphInitRDD, alpha = 0.15, maxIter = 10, verbose = True):
    """
    Spark job to implement page rank
    Args: 
        graphInitRDD  - pair RDD of (node_id , (score, edges))
        alpha         - (float) teleportation factor
        maxIter       - (int) stopping criteria (number of iterations)
        verbose       - (bool) option to print logging info after each iteration
    Returns:
        steadyStateRDD - pair RDD of (node_id, pageRank)
    """
    # teleportation:
    a = sc.broadcast(alpha)
    
    # damping factor:
    d = sc.broadcast(1-a.value)
    
    # initialize accumulators for dangling mass & total mass
    mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    
    ############## YOUR CODE HERE ###############
    # helper functions
    def getEdgeCount(payload):
        "MapValues: add an edge count to the payload for efficiency upon iteration"
        score, edges = payload
        if edges == '':
            count = 0
        else:
            count = len(edges.split(','))
        return (score, count, edges)
    
    def calculateMissingMass(record, mmAccumulator, totAccumulator):
        "Foreach:  function to run with accumulators"
        node, (score, count, edges) = record
        if count == 0:
            mmAccumulator.add(score)
        totAccumulator.add(score)       
    
    def redistributeMass(record):
        "flatMap: emit partial mass to each edge & preserve graph"
        node, (score, count, edges) = record
        # emit partials for non-danglers
        for edge in edges.split(','):
            if edge != '':
                yield (int(edge), score / float(count))
        # preserve graph
        yield (node, (0, count, edges))
        
    def combineIncoming(a, b):
        "CombineByKey: add up partial scores and preserve payload"
        score, count, edges = a
        if type(b) == type(1.0):
            # add mass
            score += b    
        else:
            # combine other params
            score += b[0]
            count = max(a[1], b[1])
            edges = max(a[2], b[2])
        return (score, count, edges)
    
    def recalculateScores(record,N,a,d,mm): 
        # N = number of nodes, 
        # a = teleportation factor, 
        # d = 1-a = damping factor,
        # mm = missing (dangling) mass
        node, (score, count, edges) = record
        score = a*(1/N) + (d)*(mm/N + score)
        return (node, (score, count, edges))
            
    # main Spark JOb         
    graph = graphInitRDD.mapValues(getEdgeCount)
    N = sc.broadcast(graph.count())

    for idx in range(maxIter):      
        # calculate and broadcast missing mass & node counts
        if verbose:
            graph.foreach(lambda x: calculateMissingMass(x, mmAccum, totAccum))
#           mm = sc.broadcast(mmAccum.value)
        
        mm = sc.broadcast(graph.filter(lambda x:x[1][1]==0).map(lambda x:x[1][0]).sum())
        
        # run page rank
        graph = graph.flatMap(redistributeMass) \
                     .aggregateByKey((0, 0, ''), combineIncoming, combineIncoming) \
                     .map(lambda x: recalculateScores(x,N.value,a.value,d.value,mm.value)) \
                     .cache()
        
        # log progress & reset accumulators
        if verbose:
            print(f'STEP {idx}: missing mass = {mm.value}, total = {totAccum.value}')
            totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
#           mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
        
            
    steadyStateRDD = graph.mapValues(lambda x: x[0])
    ############## (END) YOUR CODE ###############
    
    return steadyStateRDD

# COMMAND ----------

# part d - run PageRank on the test graph (RUN THIS CELL AS IS)
# NOTE: while developing your code you may want turn on the verbose option
nIter = 20
testGraphRDD = initGraph(testRDD)
start = time.time()
test_results = runPageRank(testGraphRDD, alpha = 0.15, maxIter = nIter, verbose = False)
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
test_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

# MAGIC %md __`expected results for the test graph:`__
# MAGIC ```
# MAGIC [(2, 0.3620640495978871),
# MAGIC  (3, 0.333992700474142),
# MAGIC  (5, 0.08506399429624555),
# MAGIC  (4, 0.06030963508473455),
# MAGIC  (1, 0.04255740809817991),
# MAGIC  (6, 0.03138662354831139),
# MAGIC  (8, 0.01692511778009981),
# MAGIC  (10, 0.01692511778009981),
# MAGIC  (7, 0.01692511778009981),
# MAGIC  (9, 0.01692511778009981),
# MAGIC  (11, 0.01692511778009981)]
# MAGIC ```

# COMMAND ----------

# part d - run PageRank on the full graph (RUN THIS CELL AS IS)
# NOTE: wikiGraphRDD should have been computed & cached above!
nIter = 10
start = time.time()
full_results = runPageRank(wikiGraphRDD, alpha = 0.15, maxIter = nIter, verbose = True)
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
full_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

top_20 = full_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

# view record from indexRDD (RUN THIS CELL AS IS)
# title\t indx\t inDeg\t outDeg
indexRDD.take(1)

# COMMAND ----------

# map indexRDD to new format (index, name) (RUN THIS CELL AS IS)
namesKV_RDD = indexRDD.map(lambda x: (int(x.split('\t')[1]), x.split('\t')[0]))

# COMMAND ----------

# see new format (RUN THIS CELL AS IS)
namesKV_RDD.take(2)

# COMMAND ----------

# We should have saved these above, but it takes too long to run in the cloud ($$$), so for expedience:
top_20 = [(13455888, 0.0015447247129832947),
 (4695850, 0.0006710240718906518),
 (5051368, 0.0005983856809747697),
 (1184351, 0.0005982073536467391),
 (2437837, 0.0004624928928940748),
 (6076759, 0.00045509400641448284),
 (4196067, 0.0004423778888372447),
 (13425865, 0.00044155351714348035),
 (6172466, 0.0004224002001845032),
 (1384888, 0.0004012895604073632),
 (6113490, 0.00039578924771805474),
 (14112583, 0.0003943847283754762),
 (7902219, 0.000370098784735699),
 (10390714, 0.0003650264964328283),
 (12836211, 0.0003619948863114985),
 (6237129, 0.0003519555847625285),
 (6416278, 0.00034866235645266493),
 (13432150, 0.00033936510637418247),
 (1516699, 0.00033297500286244265),
 (7990491, 0.00030760906265869104)]

# COMMAND ----------

# (RUN THIS CELL AS IS)
top_20_RDD = sc.parallelize(top_20)

# COMMAND ----------

# (RUN THIS CELL AS IS)
top_20_RDD.take(1)

# COMMAND ----------

# MAGIC %md # OPTIONAL
# MAGIC ### The rest of this notebook is optional and doesn't count toward your grade.
# MAGIC The indexRDD we created earlier from the indices.txt file contains the titles of the pages and thier IDs.
# MAGIC 
# MAGIC * __a) code:__ Join this dataset with your top 20 results.
# MAGIC * __b) code:__ Print the results

# COMMAND ----------

# MAGIC %md ## Join with indexRDD and print pretty

# COMMAND ----------

# part a
joinedWithNames = None
############## YOUR CODE HERE ###############

############## END YOUR CODE ###############

# COMMAND ----------

# <--- SOLUTION --->
# part a
joinedWithNames = None
############## YOUR CODE HERE ###############
joinedWithNames = namesKV_RDD.join(top_20_RDD) \
                             .sortBy(ascending=False, keyfunc=lambda k: k[1][1]) \
                             .collect()
############## END YOUR CODE ###############

# COMMAND ----------

# part b
# Feel free to modify this cell to suit your implementation, but please keep the formatting and sort order.
print("{:10s}\t| {:10s}\t| {}".format("PageRank","Page id","Title"))
print("="*100)
for r in joinedWithNames:
    print ("{:6f}\t| {:10d}\t| {}".format(r[1][1],r[0],r[1][0]))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## OPTIONAL - GraphFrames
# MAGIC GraphFrames is a graph library which is built on top of the Spark DataFrames API.
# MAGIC 
# MAGIC * __a) code:__ Using the same dataset, run the graphframes implementation of pagerank.
# MAGIC * __b) code:__ Join the top 20 results with indices.txt and display in the same format as above.
# MAGIC * __c) short answer:__ Compare your results with the results from graphframes.
# MAGIC 
# MAGIC __NOTE:__ Feel free to create as many code cells as you need. Code should be clear and concise - do not include your scratch work. Comment your code if it's not self annotating.

# COMMAND ----------

# imports
import re
import ast
import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from graphframes import *
from pyspark.sql import functions as F

# COMMAND ----------

dbutils.fs.ls("/mnt/")

# COMMAND ----------

# load the data into Spark RDDs for convenience of use later (RUN THIS CELL AS IS)
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
testRDD = sc.textFile(DATA_PATH +'test_graph.txt')
indexRDD = sc.textFile(DATA_PATH + '/indices.txt')
wikiRDD = sc.textFile(DATA_PATH + '/all-pages-indexed-out.txt')

# COMMAND ----------

DF = wikiRDD.map(lambda x: (x.split('\t')[0], ast.literal_eval(x.split('\t')[1]))).toDF()

# COMMAND ----------

# MAGIC %%time
# MAGIC DF.take(1)

# COMMAND ----------

v = DF.select('_1').withColumnRenamed('_1','id').distinct().cache()

# COMMAND ----------

# MAGIC %%time
# MAGIC v.show(1)

# COMMAND ----------

import ast
def getEdges(row):
    node_id, nodes = row
    for node in nodes: 
        yield int(node_id), int(node)

# COMMAND ----------

e = spark.createDataFrame(DF.rdd.flatMap(getEdges), ["src", "dst"]).cache()

# COMMAND ----------

# MAGIC %%time
# MAGIC e.show(1)

# COMMAND ----------

# Create a GraphFrame
from graphframes import *
g = GraphFrame(v, e)

# Query: Get in-degree of each vertex.
# g.inDegrees.show()

# COMMAND ----------

# Run PageRank algorithm, and show results.
results = g.pageRank(resetProbability=0.15, maxIter=10)

# COMMAND ----------

start = time.time()
top_20 = results.vertices.orderBy(F.desc("pagerank")).limit(20)
print(f'... completed job in {time.time() - start} seconds.')

# COMMAND ----------

# MAGIC %%time
# MAGIC top_20.show()

# COMMAND ----------

namesKV_RDD = indexRDD.map(lambda x: (int(x.split('\t')[1]), x.split('\t')[0]))

# COMMAND ----------

namesKV_DF = namesKV_RDD.toDF()

# COMMAND ----------

namesKV_DF = namesKV_DF.withColumnRenamed('_1','id')
namesKV_DF = namesKV_DF.withColumnRenamed('_2','title')
namesKV_DF.take(1)

# COMMAND ----------

resultsWithNames = namesKV_DF.join(top_20, namesKV_DF.id==top_20.id).orderBy(F.desc("pagerank")).collect()

# COMMAND ----------

# TODO: use f' for string formatting
print("{:10s}\t| {:10s}\t| {}".format("PageRank","Page id","Title"))
print("="*100)
for r in resultsWithNames:
    print ("{:6f}\t| {:10s}\t| {}".format(r[3],r[2],r[1]))

# COMMAND ----------

# MAGIC %md Our RDD implementaion takes about 35 minutes, whereas the GraphFrame one takes around 8 minutes. GraphFrames doesn't normalize the ranks. 

# COMMAND ----------

# MAGIC %md ### Congratulations, you have completed HW5! Please refer to the readme for submission instructions.
# MAGIC 
# MAGIC If you would like to provide feedback regarding this homework, please use the survey at: https://docs.google.com/forms/d/e/1FAIpQLSce9feiQeSkdP43A0ZYui1tMGIBfLfzb0rmgToQeZD9bXXX8Q/viewform

# COMMAND ----------

# MAGIC %md # Alternative solution to Q8 - PageRank

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative Pagerank which does not care about dangling mass
# MAGIC The following implementation borrows from the Spark example pagerank 'naive' implementation which utilizes joins. This implementation has been modified to work with our input format. The algorithm is based on the slides by Reza Zadeh at Stanford. http://stanford.edu/~rezab/dao/notes/Partitioning_PageRank.pdf
# MAGIC 
# MAGIC The ranks are slightly different from the above implementation which does take inot consideration dangling mass. However, the perfomance speedup is two-fold, and is probably a fair trade-off.

# COMMAND ----------

# <--- SOLUTION --->

from operator import add
def runPageRank(graphInitRDD, alpha = 0.15, maxIter = 10, verbose = True):
     
    def parseNeighbors(line):
        """
        Helper function to identify potential danglers and
        write edges as a list for efficient aggregation.
        input: 4	{'1': 1, '2': 2}
        output: 4 [1,2,2]
        """
        node, edges = line.split('\t')
        edge_list = []       
        for edge, count in ast.literal_eval(edges).items():
            # emit potential danglers w/ empty string
            yield (int(edge), [])
            # add this edge to our string of edges
            for cnt in range(int(count)):
                edge_list.append(int(edge))
        # finally yield this node w/ its string formatted edge list
        yield (int(node), edge_list)



    def computeContribs(row):
        """
        Calculates URL contributions to the rank of other URLs.
        row: (1, ([], 1.0))
        """
        node, (edges,rank) = row
        num_edges = len(edges)

        for edge in edges:
            yield (int(edge), rank/num_edges)

        yield (node,0)    


    # Loads all URLs from input file and initialize their neighbors.
    links = graphInitRDD.flatMap(lambda urls: parseNeighbors(urls))\
                        .reduceByKey(lambda a, b: a + b)\
                        .partitionBy(96)\
                        .cache()

    # Loads all URLs with other URL(s) link to from input file and initialize ranks of them to one.
    ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0),
                       preservesPartitioning=True)

    # Calculates and updates URL ranks continuously using PageRank algorithm.
    for iteration in range(int(maxIter)):
        # Calculates URL contributions to the rank of other URLs.
        contribs = links.join(ranks).flatMap(lambda url_urls_rank: computeContribs(url_urls_rank))
        # Re-calculates URL ranks based on neighbor contributions.
        ranks = contribs.reduceByKey(add, numPartitions=links.getNumPartitions())\
                        .mapValues(lambda rank: rank * 0.85 + 0.15)

    return ranks

# COMMAND ----------

nIter = 20
start = time.time()
test_results = runPageRank(testRDD, alpha = 0.15, maxIter = nIter, verbose = True)
print(f'Top 20 ranked nodes:')
print(test_results.takeOrdered(20, key=lambda x: - x[1]))
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')

# COMMAND ----------

nIter = 10
start = time.time()
# wikiGraphRDD = initGraph(wikiRDD)
full_results = runPageRank(wikiRDD, alpha = 0.15, maxIter = nIter, verbose = False)
print(f'Top 20 ranked nodes:')
print(full_results.takeOrdered(20, key=lambda x: - x[1]))
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster - 7.4 ML (includes Apache Spark 3.0.1, Scala 2.12), Nov 2020:
# MAGIC * workers:  m5.2xlarge - Min Workers 4 - Max Workers 8
# MAGIC * Driver:  c5.18xlarge - 144.0 GB Memory, 72 Cores, 10.93 DBU

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster (SCALA_standard, July 2020):
# MAGIC * workers:  13.xlarge, 30.5GB, 4cores, 1 BDU -- min workers 2, max workers 7
# MAGIC * Driver:  13.xlarge, 30.5GB, 4cores, 1 BDU

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster (KyleHW4Test, April 2020):
# MAGIC <img src="https://raw.githubusercontent.com/kyleiwaniec/w261_assets/master/images/hw4/KyleHW4test-cluster.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ### with partitioinig using 32
# MAGIC Command took 16.93 minutes -- by kylehamilton@ischool.berkeley.edu at 3/3/2020, 10:14:16 PM on KyleHW4Test
# MAGIC ### with partitioinig using 200    
# MAGIC Command took 15.04 minutes -- by kylehamilton@ischool.berkeley.edu at 3/3/2020, 11:35:54 PM on KyleHW4Test    
# MAGIC ### with partitioinig using 100 
# MAGIC Command took 11.10 minutes -- by kylehamilton@ischool.berkeley.edu at 3/3/2020, 11:52:06 PM on KyleHW4Test
# MAGIC ### with partitioinig using 96     
# MAGIC Command took 10.30 minutes -- by kylehamilton@ischool.berkeley.edu at 3/4/2020, 12:06:42 AM on KyleHW4Test
# MAGIC ### with partitioinig using 64
# MAGIC Command took 14.06 minutes -- by kylehamilton@ischool.berkeley.edu at 3/4/2020, 5:19:28 PM on KyleHW4Test

# COMMAND ----------

# MAGIC %md
# MAGIC ### with partitioinig using 96  
# MAGIC Command took 14.38 minutes -- by kylehamilton@ischool.berkeley.edu at 7/29/2020, 9:00:15 AM on SCALA_standard

# COMMAND ----------

# MAGIC %md
# MAGIC ## Partitioning
# MAGIC By default Spark will create 200 partitions. However, that may not be the ideal number for your cluster. In this case, 96 partitions appears to be the optimal number. 

# COMMAND ----------

