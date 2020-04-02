# Measurements

# Test of Hypothesis Based on Samples
In order to increase the performance of our program, a careful analysis must be needed.
We cannot only change lines of our code, see that our measures change a little bit, and conclude
that our program runs faster. It is not a game of increasing or decreasing a number.

So, how can we claim that an improvement has strong evidence to be acknowledged?
In our case, there are two contradictory hypotheses under consideration, a **current program**,
and a **potential program**, where the last one assures an improvement on the previous one.
It is initially assumed that the current program has an average performance. This is the
"prior belief" claim. The assertion that the program will increase in speed by means of our approach
is a contradiction to the prior claim. Nevertheless, an assertion of that contradiction is not wanted
unless and until data can provide strong support for it. Hypothesis, claims, contradictions, strong
supports. It sounds like a **hypothesis-testing problem!**

A test of hypotheses is a method for using sample data to decide whether a **null hypothesis** 
should be rejected in terms of an **alternative hypothesis**. On one hand, the prior belief that
the current program has an average performance $$\mu_0 = \mu_0$$, where $$mu_0$$ is the average ns needed to update a cell, is the null hypothesis denoted by $$H_0$$. On the other hand, the alternative hypothesis, denoted by $$H_a$$ is the claim $$\mu = \mu_0$$, i.e, the average performance of the program decreases with the proposed approach.

Since the true average performance of the population mu is unknown, a sample mean is used by means of
a set of observations in order to make an approximation.
These observations are going to be the number of ns per cell needed to perform the functions mentioned above, which are:

* *react*
* *vel_step*
* *dens_step*

Each of one will lead to sets of observations 
$${x_i: i=1, ..., n}$$ 
$${y_i: 1, ... , n}$$ 
$${w_i: i=1, ..., n}$$ respectively.
Finally we are going to observe the set 
$${z_i: i=1 ,...,n}$$. Where $${z_i = x_i + y_i + z_i}$$
Since these sets of samples might differ at least a bit every execution, there is uncertainty about the value of each sample. Because of this uncertainty, before the data becomes available we view each observation as a random variable.

* $${X_1, X_2, . . . , X_n}$$: Where $$X_i$$ is the number of ns per cell needed to perform react in the step i.
* $${Y_1, Y_2, . . . , Y_n}$$: Where $$Y_i$$ is the number of ns per cell needed to perform vel_step in the step i.
* $${W_1, W_2, . . . , W_n}$$: Where $$W_i$$ is the number of ns per cell needed to perform dens_step in the step i.
* $${Z_1, Z_2, . . . , Z_n}$$: Where $$Z_i$$ is the number of ns per cell needed to perform the three functions in the step i.

The objective is to decide, based on sample information, which of the two hypotheses is correct.
Therefore, we need a function of the sample data on which the decision (reject $$H_0$$ or do not reject $$H_0$$)
is to be based. This will be our *test statistic*, and a *rejection region*, which is the set of all
test statistic values for which $$H_0$$ will be rejected.

In order to choose a rejection region, errors that might be faced with in drawing a conclusion are
taken into account. We might take our improvement leads to a faster solution when that is not true. 
We might also affirm that the improvement is not a better solution when it actually is. These are **type 1 ($$\alpha$$)** 
and **type 2 error ($$\beta$$)** respectively.

Since a type 1 error is worst, we specified the largest value of alpha that can be tolerated and find
a rejection region having that value of alpha rather than anything smaller. In our case, a level of
significance is $$\alpha = 0.10$$

Since the amount of samples that can be produced is large, each of them represented by a random variable equally distributed in every step, a case 2 hypothesis process about a population mean can be used. 
Finally these leads to a test statistic denoted by: 
$$Z = ( \overline X - \mu_0)/(s/ \sqrt(n))$$

Let us put all the things together. This is the process that is used in our project.

1. A heuristic is proposed to make the code faster.
 
2. Two versions of our program (the **current** and the **potential** one) are obtained.
    
3. The functions *react*, *vel_step*, and *dens_step* will be executed during **N** steps. 
       Where **N** is sufficiently large.
    
4. Samples $$x_i's, y_i's, w_i's,  z_i's$$ are computed to obtain the mean amount of ns needed to perform each function.
    
5. These observed values are given to our *test statistics* in order to decide if reject or not to reject $$H_0$$ on each
       function.
    
6. The observed values are normalized and checked if they belong to the *rejection regions* previously calculated in
       terms of $$\alpha$$.
    
7. An output is produced by rejecting or failing to reject the **current program** on each function.
    
8. Go to 1 and run the process again until the decision is convincing.