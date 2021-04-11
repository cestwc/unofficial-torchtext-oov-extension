from oov import FieldOOV

import torch
from torchtext.legacy import datasets, data

def generate_bigrams(x):
	n_grams = set(zip(*[x[i:] for i in range(2)]))
	for n_gram in n_grams:
		x.append(' '.join(n_gram))
	return x

class ENGLISHTEXT(FieldOOV):

	def __init__(self, sequential=True, use_vocab=True, init_token='<sos>',
					eos_token='<eos>', fix_length=None, dtype=torch.long,
					preprocessing=generate_bigrams, postprocessing=None, lower=True,
					tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=False,
					batch_first=False, pad_token="<pad>", unk_token="<unk>",
					pad_first=False, truncate_first=False, stop_words=None,
					is_target=False, min_freq = 1, max_vocab_size=50000, vectors="glove.6B.100d", build_vocab=False):

		super().__init__(sequential, use_vocab, init_token,
					eos_token, fix_length, dtype,
					preprocessing, postprocessing, lower,
					tokenize, tokenizer_language, include_lengths,
					batch_first, pad_token, unk_token,
					pad_first, truncate_first, stop_words,
					is_target)

		if build_vocab == True:

			WikiText2 = datasets.WikiText2.splits(self)
			WikiText103 = datasets.WikiText103.splits(self, train=None)
			PennTreebank = datasets.PennTreebank.splits(self)
			SNLI = datasets.SNLI.splits(self, data.LabelField())

			self.build_vocab(WikiText2[0], WikiText2[1], WikiText2[2], 
							WikiText103[0], WikiText103[1], 
							PennTreebank[0], PennTreebank[1], PennTreebank[2],
							SNLI[0], SNLI[1], SNLI[2], 
						min_freq = min_freq, 
						max_size = max_vocab_size, 
						vectors = vectors, 
						unk_init = torch.Tensor.normal_)
