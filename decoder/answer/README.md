
Your documentation
------------------

We implemented Collins' description of a translator with an arbitrary ordering
We defined a "State" which contains the a history of translated words, the probability of the tranlation so far, and a record of which french words were used so far
For every French sentance we loop around for every word
  At the ith iteration we look at the best states which translate i words from the french
    (The initial state translates 0 words)
  We then consider all the additional words or phrases to each of these states
  These new states are considered at later iteration, building up to the end of the sentance, when the states are full, and the most likely (highest probability) is chosen as the translation.

Developing on this algorithm:
  We added weightings to the different elements of the probability and tuned them by hand for best results
  Added additional weighting in the form of decreasing the probability of translations which are less contiguous to the last word translated in each state
  We implemented a future cost factor that de-incentivises choosing the easiest words to translate first, with dissappointing results.
    

