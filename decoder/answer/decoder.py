#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple


class State:

    def __init__(self, words_used, last_index, logprob, lm_state):
        self.words_used = words_used
        self.last_index = last_index
        self.logprob = logprob
        self.lm_state = lm_state

    def create_new_state(self, lm_state, phrase_start, phrase_end, logprob):
        used = [False for _ in range(len(self.words_used))]
        for i in range(len(self.words_used)):
            if phrase_start <= i <= phrase_end:
                if self.words_used[i]:
                    return False
                else:
                    used[i] = True
            else:
                used[i] = self.words_used[i]
        return State(used, phrase_end, logprob, lm_state)

    def is_equal(self, state):
        return (self.lm_state[-1] == self.lm_state[-1] and
                  (len(self.lm_state) >= 2 and
                       self.lm_state[-2] == self.lm_state[-2]) and
                self.words_used == state.words_used and
                self.last_index == state.last_index)


optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="../data/input", help="File containing sentences to translate (default=../data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="../data/tm", help="File containing translation model (default=../data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="../data/lm", help="File containing ARPA-format language model (default=../data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-d", "--distortion-factor", dest="d", default=4, type="int", help="Limit on how far from each other consecutive phrases can start (default=4)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
    # The following code implements a monotone decoding
    # algorithm (one that doesn't permute the target phrases).
    # Hence all hypotheses in stacks[i] represent translations of 
    # the first i words of the input sentence. You should generalize
    # this so that they can represent translations of *any* i words.
    hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase")
    hypothesis(0.0, lm.begin(), None, None)
    initial_hypothesis = State([False for _ in range(len(f))], 0, 0, lm.begin())
  
    # stacks is an array of dictionaries one longer than the sentance
    # the i-th dict of stacks represents the partial decodings of the sentance
    #   with i words matched
    stacks = [[] for _ in f] + [[]]
    stacks[0].append(initial_hypothesis)
  
    # iterates over the array of stacks, building them as it goes
    for i, stack in enumerate(stacks[:-1]):
  
        # iterates over all the partial decodings in the stack
        # only considers a number of them specified in the option -s
        # considers them in the order of likelihood
        for state in sorted(iter(stack),key=lambda state: -state.logprob)[:opts.s]: # prune

            # find every phrase in the translation model (tm)
            # that can be found in the remaining sentance
            # to avoid:
            #    can discount phrases that begin inside the distortion model distance
            #    don't bother looking at words used in the current state
            start = max(state.last_index - opts.d, 0)
            end = min(state.last_index + opts.d, len(f))

            # looking for phrases: iterating the start index inside bounds of distortion factor
            for s in xrange(start, end):
                # if the word was already used, dont try to use it again
                if state.words_used[s]:
                    continue

                # looking for phrases: iterating the end index
                for t in xrange(s, len(f)):
                    # if we find an already used word, don't move the end index past it
                    if state.words_used[t]:
                        break

                    # is the phrase in the tranlation model
                    if f[s:t] in tm:
                        for phrase in tm[f[s:t]]:
                            # creating a new state for every translation of the phrase
                            logprob = state.logprob + phrase.logprob
                            lm_state = state.lm_state
                            for word in phrase.english.split():
                                (lm_state, word_logprob) = lm.score(lm_state, word)
                                logprob += word_logprob
                            new_hypothesis = state.create_new_state(lm_state, s, t, logprob)

                            length = s - t + 1
                            inserted = False
                            for s in range(len(stacks[i + length])):
                                if stacks[i + length][s].is_equal(new_hypothesis):
                                    if new_hypothesis.logbrob < logprob:
                                        stacks[i + length][s] = new_hypothesis
                                        inserted = True
                                        break
                            if not inserted:
                                stacks[i + length].append(new_hypothesis)
                    


            '''
            -- default code --
            for j in xrange(i+1,len(f)+1):
              if f[i:j] in tm:
                for phrase in tm[f[i:j]]:
                  logprob = h.logprob + phrase.logprob
                  lm_state = h.lm_state
                  for word in phrase.english.split():
                    (lm_state, word_logprob) = lm.score(lm_state, word)
                    logprob += word_logprob
                  logprob += lm.end(lm_state) if j == len(f) else 0.0
                  new_hypothesis = hypothesis(logprob, lm_state, h, phrase)
                  if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
                    stacks[j][lm_state] = new_hypothesis 
            '''

    winner = max(iter(stacks[-1]), key=lambda h: h.logprob)
    def extract_english(h): 
        return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
    print extract_english(winner)

    if opts.verbose:
        def extract_tm_logprob(h):
            return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
            (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
