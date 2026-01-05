
I need to have a mathematical formulation which I believe in + one of you supervisor's agree with, which has a good chance of converging towards something meaningful. (And it likely should not be ambitious).



Here are my thoughts about RL in this context (disregarding training time, and the fact that it probably needs to be retrained for each new contract setup/condition):



The big thing is that what RL could help with was:

* (1) Provide proxy of future which is only a small expression in the optimization model instead of 3(or many more) days of optimization.
* (2) The uncertain things about the future are: Distribution/timing of prices, correlation with PPAs.
* (3) This uncertainty hinders us in answering the questions:
* (4) What is the true value of having a given storage level at EOD/(end of step horizon)?
* (5) What is the internal value of producing X, Y, Z fuels?



(1) is kind of moot, as the solving time is not really that big of a concern and "good enough" performance is already seen.

(2) These uncertainties are not predicted by RL, but really hard for it to learn, however with our duration curve approach we have an existing approach which yields very satisfying performance - close to the optimal policy that RL could potentially find.



(4) This is still an open question, but not that important to answer (generally it is, but not for the performance in thesis) given (1).

(5) Same argument as (2) more or less.



