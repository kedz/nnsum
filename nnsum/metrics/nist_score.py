from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from nltk.translate.nist_score import nist_length_penalty
from nltk.translate import bleu_score
from collections import Counter
from nltk.util import ngrams
import math


class NISTScore(Metric):
    def __init__(self, output_transform=lambda x: x, zero_ok=False):
        super(NISTScore, self).__init__(output_transform)
        self._zero_ok = zero_ok

    def reset(self):
        self._refs = []
        self._hyps = []

    def update(self, output):
        hypotheses, references = output
        self._refs.extend(references)
        self._hyps.extend(hypotheses)

    def compute(self):
        if len(self._refs) == 0:
            if self._zero_ok:
                return 0.
            else:
                raise NotComputableError(
                    'Loss must have at least one example before it can be ' \
                    'computed')
        return {"NIST score": corpus_nist(self._refs, self._hyps),
                "BLEU score": bleu_score.corpus_bleu(self._refs, self._hyps)}

def corpus_nist(list_of_references, hypotheses, n=5):
    """
    Calculate a single corpus-level NIST score (aka. system-level BLEU) for all
    the hypotheses and their respective references.

    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param n: highest n-gram order
    :type n: int
    """
    # Before proceeding to compute NIST, perform sanity checks.
    assert len(list_of_references) == len(
        hypotheses
    ), "The number of hypotheses and their reference(s) should be the same"

    # Collect the ngram coounts from the reference sentences.
    ngram_freq = Counter()
    total_reference_words = 0
    for (
        references
    ) in list_of_references:  # For each source sent, there's a list of reference sents.
        for reference in references:
            # For each order of ngram, count the ngram occurrences.
            for i in range(1, n + 1):
                ngram_freq.update(ngrams(reference, i))
            total_reference_words += len(reference)

    # Compute the information weights based on the reference sentences.
    # Eqn 2 in Doddington (2002):
    # Info(w_1 ... w_n) = log_2 [ (# of occurrences of w_1 ... w_n-1) / (# of occurrences of w_1 ... w_n) ]
    information_weights = {}
    for _ngram in ngram_freq:  # w_1 ... w_n
        _mgram = _ngram[:-1]  #  w_1 ... w_n-1
        # From https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v13a.pl#L546
        # it's computed as such:
        #     denominator = ngram_freq[_mgram] if _mgram and _mgram in ngram_freq else denominator = total_reference_words
        #     information_weights[_ngram] = -1 * math.log(ngram_freq[_ngram]/denominator) / math.log(2)
        #
        # Mathematically, it's equivalent to the our implementation:
        if _mgram and _mgram in ngram_freq:
            numerator = ngram_freq[_mgram]
        else:
            numerator = total_reference_words
        information_weights[_ngram] = math.log(numerator / ngram_freq[_ngram], 2)

    # Micro-average.
    nist_precision_numerator_per_ngram = Counter()
    nist_precision_denominator_per_ngram = Counter()
    l_ref, l_sys = 0, 0
    # For each order of ngram.
    for i in range(1, n + 1):
        # Iterate through each hypothesis and their corresponding references.
        for references, hypothesis in zip(list_of_references, hypotheses):
            hyp_len = len(hypothesis)

            # Find reference with the best NIST score.
            nist_score_per_ref = []
            for reference in references:
                _ref_len = len(reference)
                # Counter of ngrams in hypothesis.
                hyp_ngrams = (
                    Counter(ngrams(hypothesis, i))
                    if len(hypothesis) >= i
                    else Counter()
                )
                ref_ngrams = (
                    Counter(ngrams(reference, i)) if len(reference) >= i else Counter()
                )
                ngram_overlaps = hyp_ngrams & ref_ngrams
                # Precision part of the score in Eqn 3
                _numerator = sum(
                    information_weights[_ngram] * count
                    for _ngram, count in ngram_overlaps.items()
                )
                _denominator = sum(hyp_ngrams.values())
                _precision = 0 if _denominator == 0 else _numerator / _denominator
                nist_score_per_ref.append(
                    (_precision, _numerator, _denominator, _ref_len)
                )
            # Best reference.
            precision, numerator, denominator, ref_len = max(nist_score_per_ref)
            nist_precision_numerator_per_ngram[i] += numerator
            nist_precision_denominator_per_ngram[i] += denominator
            l_ref += ref_len
            l_sys += hyp_len

    # Final NIST micro-average mean aggregation.
    nist_precision = 0
    for i in nist_precision_numerator_per_ngram:
        if nist_precision_denominator_per_ngram[i] == 0:
            precision = 0.
        else:
            precision = (
                nist_precision_numerator_per_ngram[i]
                / nist_precision_denominator_per_ngram[i]
            )
        nist_precision += precision
    # Eqn 3 in Doddington(2002)
    return nist_precision * nist_length_penalty(l_ref, l_sys)
