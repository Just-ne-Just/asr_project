from typing import List, NamedTuple

import torch
import numpy as np
from hw_asr.utils import ROOT_PATH
from .char_text_encoder import CharTextEncoder
from collections import defaultdict
from pyctcdecode.decoder import build_ctcdecoder
import multiprocessing


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lm = False):
        super().__init__(alphabet)
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.lm = lm

        if self.lm:
            vocab_for_lm = [elem.upper() for elem in self.vocab[1:]]
            self.model = build_ctcdecoder(
                vocab_for_lm,
                kenlm_model_path=str(ROOT_PATH / 'data' / 'kenlm' / '4-gram.arpa')
            )

    def ctc_decode(self, inds: List[int]) -> str:
        last_char = self.char2ind[self.EMPTY_TOK]
        decoded_text = []
        for ind in inds:
            if ind == self.char2ind[self.EMPTY_TOK]:
                last_char = self.char2ind[self.EMPTY_TOK]
            elif ind != last_char:
                decoded_text.append(self.ind2char[ind])
                last_char = ind
        return ''.join(decoded_text)
    
    
    def ctc_lm(self, probs: torch.tensor, probs_length: torch.tensor,
            beam_size: int = 100) -> List[str]:
            
        # probs = torch.nn.functional.log_softmax(probs, -1)

        logits_list = np.array([probs[i][:probs_length[i]].numpy() for i in range(probs_length.shape[0])])

        with multiprocessing.Pool() as p:
            hypos = self.model.decode_batch(p, logits_list, beam_width=beam_size)

        hypos = [elem.replace("|", "").replace("??", "").replace("'", "").lower().strip() for elem in hypos]

        return hypos
    
    
    def _extend_and_merge(self, frame, state):
        new_state = defaultdict(float)
        for next_char_index, next_char_proba in enumerate(frame):
            for (pref, last_char), pref_proba in state.items():
                next_char = self.ind2char[next_char_index]
                if next_char == last_char:
                    new_pref = pref
                else:
                    if next_char != self.EMPTY_TOK:
                        new_pref = pref + next_char
                    else:
                        new_pref = pref
                    last_char = next_char
                new_state[(new_pref, last_char)] += pref_proba + next_char_proba
        return new_state

    def _truncate(self, state, beam_size):
        state_list = list(state.items())
        state_list.sort(key=lambda x: x[1], reverse=True)
        return dict(state_list[:beam_size])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        probs = torch.log(probs)
        hypos: List[Hypothesis] = []
        state = {('', self.EMPTY_TOK): 1.0}

        for frame in probs:
            state = self._extend_and_merge(frame, state)
            state = self._truncate(state, beam_size)
        
        state_list = list(state.items())
        hypos = list(map(lambda x: Hypothesis(x[0][0], np.exp(x[1])), state_list))

        return hypos
    

