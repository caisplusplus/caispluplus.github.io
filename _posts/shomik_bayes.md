---
layout: post
published: true
title: "Naive Bayes Classifiers"
mathjax: true
featured: true

comments: true
---

### Table of Contents
0. [Introduction](#introduction)
1. [Bayes' Theorem](#bt)
2. [Naive Bayes Classifier](#nbc)
3. [Example: Text Classification](#ex)
4. [Conclusion](#conclusion)


<h2><a name="introduction"></a>Introduction</h2>

In a classification problem, we are given certain features/attributes about our sample and are interested in classifying the sample, or assigning a discrete class. Naive Bayes Classifiers are simple probabilistic classifiers that are based on Bayes' Theorem for conditional probability. An important ("naive") assumption for these classifiers is the features are independent. Naive Bayes Classifiers work surprisingly well for many classification problems

<h2><a name='bt'></a>Bayes' Theorem</h2>

In probability theory and statistics, [**Bayes' Theorem**](https://en.wikipedia.org/wiki/Bayes%27_theorem) is a useful formula for working with conditional probabilities:

$$\mathbb{P}(A | B) = \frac{\mathbb{P}(B | A) \times \mathbb{P}(A)}{\mathbb{P}(B)}$$

where $A$ and $B$ are events and $\mathbb{P}(B) \neq 0$. 

 **Note:** $\mathbb{P}(X | Y)$ is a conditional probability. It is the probability that event $X$ will occur given the knowledge that event $Y$ has occurred. 

<h2><a name='nbc'></a>Naive Bayes Classifier</h2>

Abstractly, the Naive Bayes Classifier is a conditional probability model for a classification problem, where there are:

*  $n$ features about the sample: $X = (x_{1} ... x_{n})$
* $k$ possible classes: $C_{1} ... C_{k}$

Applying Bayes' Theorem, we can find the conditional probability that a sample $i$ belongs to class $C_{j}$ given the sample's features $X_{i}$:
$$\mathbb{P}(C_{j} | X_{i}) = \frac{\mathbb{P}(X_{i} | C_{j}) \times \mathbb{P}(C_{j})}{\mathbb{P}(X_{i})} $$

In practice, the denominator $\mathbb{P}(X_{i})$ can be ignored because it does not depend on the class $C_{j}$. Thus, the conditional probability can be written as:
$$\mathbb{P}(C_{j} | X_{i}) \propto  \mathbb{P}(X_{i} | C_{j})  \times \mathbb{P}(C_{j})$$

**Note:**  $\propto$  means "directly proportional to". Informally, it is used to denote equivalence "up to constant factors".

Now comes the "naive" assumption that the features are conditionally independent. In other words, we will assume that each feature contributes independently to the probability that the sample is in the class $C_{j}$. Formally, this is written as:
$$ \forall a \ \in\ [1, n]:  \ \mathbb{P}(x_{a} |\  x_{1},  ... \ , x_{a-1},x_{a+1}, ... \ , x_{n}, C_{j}) =   \mathbb{P}(x_{a} | C_{j}) $$

Given this assumption, the conditional probability that a sample $i$ belongs to class $C_{j}$ given the sample’s features $X_{i}$ can be written as:
$$\mathbb{P}(C_{j} | X_{i}) \propto  \mathbb{P}(C_{j}) \times \prod_{a=1}^{n}  \mathbb{P}(x_{a} | C_{j})$$

Finally, to classify our sample using this Naive Bayes model, we simply choose the class $C_{j}$ with the highest conditional probability:

$$ argmax_{j\in[1, k]} \ \mathbb{P}(C_{j}) \times \prod_{a=1}^{n}  \mathbb{P}(x_{a} | C_{j})$$

<h2><a name='ex'></a>Example: Text Classification</h2>

A common application of Naive Bayes classifiers is text classification, particularly for identifying spam emails. 

Here, we will discuss a more simple text classification example for positive vs negative movie reviews. Our training set has 4 sentences:
| Text  | Category |
|---|---|---|---|---|
| a great movie  |   Positive
| very good film  |   Positive
| a pathetic film  |   Negative
| very disappointing movie | Negative

Let's try to predict which category "a very good movie" belongs to. 

In this example, we can use individual words as features, and word frequencies to calculate conditional probabilities. Note that we must make the "naive" assumption that every word in a sentence contributes independently to the probability that the sentence is positive or negative. 

Using our formula for Naive Bayes classification in the previous section, we can get the conditional probability that this sentence is positive (similar for negative):
$$\mathbb{P}(Positive| a \ very \ good \ movie ) $$

$$\propto  \mathbb{P}(a \ very \ good \ movie  | Positive)  \times \mathbb{P}(Positive)$$

$$\propto  \mathbb{P}(a|Positive) \times \mathbb{P}(very|Positive) \times \mathbb{P}(good|Positive) \times \mathbb{P}(movie|Positive) \times \mathbb{P}(Positive)$$

Now, we must calculate these conditional probabilities using word frequencies from our training set. However, there is an added caveat that we must account for zero probabilities. Thus, the following formula is commonly used in practice:

$$\mathbb{P}(word|class) \approx \frac{\#\ times \ word \ appears\ in \ class+ 1}{\#\ total \ words\ in\ class + \#\ unique\ words \ in \ training \ set}$$
  
 For our example, the # of words is 6 for both classes, and the # of unique words is 8.
 
| Words  | $\mathbb{P}$(word&#124;Positive) | $\mathbb{P}$(word&#124;Negative) |
|---|---|---|---|---|
| a, movie, very, film  |   $\frac{1+1}{6+8}$ = $\frac{1}{7}$| $\frac{1+1}{6+8}$ = $\frac{1}{7}$ |
| great, good |   $\frac{1+1}{6+8}$ = $\frac{1}{7}$| $\frac{0+1}{6+8}$ = $\frac{1}{14}$|
| pathetic, disappointing | $\frac{0+1}{6+8}$ = $\frac{1}{14}$| $\frac{1+1}{6+8}$ = $\frac{1}{7}$ |

Additionally, since each class has an equal number of words:
$$\mathbb{P}(Positive) = \mathbb{P}(Negative) = \frac{1}{2}$$

Now, plugging into our formula above, we find:  
$$\mathbb{P}(Positive| a \ very \ good \ movie ) = \frac{1}{2} \times \frac{1}{7} \times \frac{1}{7} \times \frac{1}{7} \times \frac{1}{7} = \frac{1}{2\times7^{4}}$$ $$\mathbb{P}(Negative| a \ very \ good \ movie ) = \frac{1}{2} \times \frac{1}{7} \times \frac{1}{7} \times \frac{1}{14} \times \frac{1}{7} = \frac{1}{2^{2}\times7^{4}}$$

Therefore, since $\mathbb{P}(Positive| a \ very \ good \ movie )$ is greater than $\mathbb{P}(Negative| a \ very \ good \ movie )$ we conclude that our sample movie review sentence is positive! 

<h2><a name='conclusion'></a>Conclusion</h2>

As we've seen, Naive Bayes Classifiers are simple probabilistic classifiers that use Bayes' Theorem for conditional probability. The simplicity of Naive Bayes Classifiers make it a popular choice for smaller datasets and text classification problems. However, while Naive Bayes Classifiers can learn the importance of individual features, ignoring the relationship among features may be a critical shortcoming for many real-world problems. 

**Note:**  As an aside, although Naive Bayes Classifiers use Bayes' Theorem, it is important to note that these classifiers are not necessarily a [**Bayesian Method**](https://en.wikipedia.org/wiki/Bayesian_probability). Generally, there are two approaches to probability and statistics: Bayesian and Frequentist. 
<br>

<h2>Sources</h2>
<ul>
  <li><a href='https://en.wikipedia.org/wiki/Bayes%27_theorem'>wikipedia.org/BayesTheorem</a></li>
  <li><a href='http://www.ee.columbia.edu/~vittorio/Lecture5.pdf'>ee.columbia.edu</a></li>
  <li><a href='https://web.stanford.edu/class/cs124/lec/naivebayes.pdf'>web.stanford.edu</a></li>
  <li><a href='https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/'>monkeylearn.com</a></li>
</ul>
