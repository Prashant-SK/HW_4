#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple

WEIGHT_DISTORTION = 0.0
WEIGHT_LANG_MODEL = 1
WEIGHT_TRANS_MODEL = 1

class State:

    def __init__(self, phrase, words_used, last_index, predecessor, logprob, lm_state):
        self.phrase = phrase
        self.words_used = words_used
        self.last_index = last_index
        self.predecessor = predecessor
        self.logprob = logprob
        self.lm_state = lm_state

    def create_new_state(self, phrase, lm_state, phrase_start, phrase_end, logprob, distortion_max):
        used = [False for _ in range(len(self.words_used))]
        for i in range(len(self.words_used)):
            if phrase_start <= i < phrase_end:
                if self.words_used[i]:
                    return False
                else:
                    used[i] = True
            else:
                used[i] = self.words_used[i]
        i = 0
        while i < len(used) and used[i]:
            i += 1
        if i  + distortion_max < phrase_end:
            return False

        return State(phrase, used, i, self, logprob, lm_state)


    def is_equal(self, state):
        if self.last_index != state.last_index:
            return False
        for i in range(len(self.words_used)):
            if self.words_used[i] != state.words_used[i]:
                return False
        if len(self.lm_state) != len(state.lm_state):
            return False
        if len(self.lm_state) >= 1:
            if self.lm_state[-1] != state.lm_state[-1]:
                return False
        if len(self.lm_state) >= 2:
            if self.lm_state[-2] != state.lm_state[-2]:
                return False
        return True

    def get_phrase_list(self):
        if self.predecessor != None:
            l = self.predecessor.get_phrase_list()
        else:
            l = []
        if self.phrase != None:
            l.append(self.phrase.english)
        return l

    def get_sentance(self):
        l = self.get_phrase_list()
        return " ".join(l)

    def print_state(self):
        s = ""
        for i in self.words_used:
            if i:
                s += "o"
            else:
                s += "."
        return "(%s %s (%d)): %f" % (self.lm_state, s, self.last_index, self.logprob)


optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-d", "--distortion-factor", dest="d", default=6, type="int", help="Limit on how far from each other consecutive phrases can start (default=6)")
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
    initial_hypothesis = State(None, [False for _ in range(len(f))], 0, None, 0, lm.begin())
  
    # stacks is an array of dictionaries one longer than the sentance
    # the i-th dict of stacks represents the partial decodings of the sentance
    #   with i words matched
    stacks = [[] for _ in f] + [[]]
    stacks[0].append(initial_hypothesis)

  
    # iterates over the array of stacks, building them as it goes
    for i, stack in enumerate(stacks[:-1]):
        #raw_input()
        #print "Stack %d" % i
  
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
            end = min(state.last_index + opts.d - 1, len(f))

            # looking for phrases: iterating the start index inside bounds of distortion factor
            for s in xrange(start, end):
                # if the word was already used, dont try to use it again
                if state.words_used[s]:
                    continue

                # looking for phrases: iterating the end index
                for t in xrange(s + 1, end + 1):
                    # if we find an already used word, don't move the end index past it
                    if state.words_used[t - 1]:
                        break

                    # is the phrase in the tranlation model
                    if f[s:t] in tm:
                        for phrase in tm[f[s:t]]:
                            # creating a new state for every translation of the phrase
                            lm_state = state.lm_state
                            word_logprob = 0
                            for word in phrase.english.split():
                                (lm_state, w_logprob) = lm.score(lm_state, word)
                                word_logprob += w_logprob
                            distortion_logprob = -3 * abs(state.last_index - s - 1)
                            new_logprob = state.logprob 
                            new_logprob += WEIGHT_TRANS_MODEL * phrase.logprob
                            new_logprob += WEIGHT_LANG_MODEL * word_logprob
                            new_logprob += WEIGHT_DISTORTION * distortion_logprob
                            new_hypothesis = state.create_new_state(phrase, lm_state, s, t, new_logprob, opts.d)
                            if not new_hypothesis:
                                continue
                            #print "%s + (%d, %d: %s) --> %s" % (state.print_state(), s, t, phrase.english, new_hypothesis.print_state())

                            position = i + t - s
                            inserted = False
                            for st in stacks[position]:
                                if st.is_equal(new_hypothesis):
                                    if new_hypothesis.logprob < st.logprob:
                                        #del st
                                        st = new_hypothesis
                                    inserted = True
                                    break
                            if not inserted:
                                stacks[position].append(new_hypothesis)
                    


    best_stack = []
    back = 0
    while best_stack == []:
        back -= 1
        best_stack = stacks[back]
    winner = max(iter(best_stack), key=lambda h: h.logprob)
    def extract_english(h):
        return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
    if back != -1:
        print "ERROR"
    print winner.get_sentance()

    if opts.verbose:
        def extract_tm_logprob(h):
            return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
            (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
