import torch
from torchtext.legacy.data import BucketIterator, Batch, Field

class BucketIteratorOOV(BucketIterator):

	def __init__(self, dataset, batch_size, sort_key=None, device=None, batch_size_fn=None, train=True, repeat=False, shuffle=None, sort=None, sort_within_batch=None):
		super().__init__(dataset, batch_size, sort_key, device, batch_size_fn, train, repeat, shuffle, sort, sort_within_batch)

	def __iter__(self):
		while True:
			self.init_epoch()
			for idx, minibatch in enumerate(self.batches):
				# fast-forward if loaded from state
				if self._iterations_this_epoch > idx:
					continue
				self.iterations += 1
				self._iterations_this_epoch += 1
				if self.sort_within_batch:
					# NOTE: `rnn.pack_padded_sequence` requires that a minibatch
					# be sorted by decreasing order, which requires reversing
					# relative to typical sort keys
					if self.sort:
						minibatch.reverse()
					else:
						minibatch.sort(key=self.sort_key, reverse=True)
				yield BatchOOV(minibatch, self.dataset, self.device)
			if not self.repeat:
				return

class BatchOOV(Batch):

	def __init__(self, data=None, dataset=None, device=None, src='src'): # name of source field is hard coded for seq2seq tasks
		super().__init__(data, dataset, device)
		if data is not None:
			self.batch_size = len(data)
			self.dataset = dataset
			self.fields = dataset.fields.keys()  # copy field names
			self.input_fields = [k for k, v in dataset.fields.items() if
								 v is not None and not v.is_target]
			self.target_fields = [k for k, v in dataset.fields.items() if
								  v is not None and v.is_target]

			if src in dataset.fields:
				srcbatch = [getattr(x, src) for x in data]
				for (name, field) in dataset.fields.items():
					if field is not None:
						batch = [getattr(x, name) for x in data]
						setattr(self, name, field.process(batch, srcBatch=srcbatch, device=device))

class FieldOOV(Field):

	def __init__(self, sequential=True, use_vocab=True, init_token=None,
				 eos_token=None, fix_length=None, dtype=torch.long,
				 preprocessing=None, postprocessing=None, lower=False,
				 tokenize=None, tokenizer_language='en', include_lengths=False,
				 batch_first=False, pad_token="<pad>", unk_token="<unk>",
				 pad_first=False, truncate_first=False, stop_words=None,
				 is_target=False):
		super().__init__(sequential, use_vocab, init_token,
				 eos_token, fix_length, dtype,
				 preprocessing, postprocessing, lower,
				 tokenize, tokenizer_language, include_lengths,
				 batch_first, pad_token, unk_token,
				 pad_first, truncate_first, stop_words,
				 is_target)

	def process(self, batch, srcBatch=None, device=None): # srcBatch is inserted before device, caution errors!
		""" Process a list of examples to create a torch.Tensor.

		Pad, numericalize, and postprocess a batch and create a tensor.

		Args:
			batch (list(object)): A list of object from a batch of examples.
		Returns:
			torch.autograd.Variable: Processed object given the input
			and custom postprocessing Pipeline.
		"""
		padded = self.pad(batch)
		if srcBatch != None:
			srcPadded = self.pad(srcBatch)
			tensor = self.numericalizeOOV(padded, srcPadded, device=device) # another numericalize function to handle OOV
		else:
			tensor = self.numericalize(padded, device=device)
		return tensor


	def numericalizeOOV(self, arr, srcArr, device=None):
		"""Turn a batch of examples that use this field into a Variable.

		If the field has include_lengths=True, a tensor of lengths will be
		included in the return value.

		Arguments:
			arr (List[List[str]], or tuple of (List[List[str]], List[int])):
				List of tokenized and padded examples, or tuple of List of
				tokenized and padded examples and List of lengths of each
				example if self.include_lengths is True.
			device (str or torch.device): A string or instance of `torch.device`
				specifying which device the Variables are going to be created on.
				If left as default, the tensors will be created on cpu. Default: None.
		"""
		if self.include_lengths and not isinstance(arr, tuple):
			raise ValueError("Field has include_lengths set to True, but "
							 "input data is not a tuple of "
							 "(data batch, batch lengths).")
		if self.include_lengths and not isinstance(srcArr, tuple): # duplicate
			raise ValueError("Field has include_lengths set to True, but "
							 "input data is not a tuple of "
							 "(data batch, batch lengths).")

		if isinstance(arr, tuple):
			arr, lengths = arr
			lengths = torch.tensor(lengths, dtype=self.dtype, device=device)
		if isinstance(srcArr, tuple): # duplicate
			srcArr, srcLengths = srcArr
			srcLengths = torch.tensor(srcLengths, dtype=self.dtype, device=device)

		def stoiOOV(arr, srcArr, vocab):
			ids = []
			for i in range(len(arr)):
				idx = vocab.stoi[arr[i]]
				if idx == vocab.stoi['<unk>'] and arr[i] in srcArr:
					idx = len(vocab) + srcArr.index(arr[i]) # Map to its temporary article OOV number
				ids.append(idx)
			return ids

		if self.use_vocab:
			if self.sequential:
				arr = [stoiOOV(arr[i], srcArr[i], self.vocab) for i in range(len(arr))]
			else:
				arr = stoiOOV(arr, srcArr, self.vocab)

			if self.postprocessing is not None:
				arr = self.postprocessing(arr, self.vocab)
		else:
			if self.dtype not in self.dtypes:
				raise ValueError(
					"Specified Field dtype {} can not be used with "
					"use_vocab=False because we do not know how to numericalize it. "
					"Please raise an issue at "
					"https://github.com/pytorch/text/issues".format(self.dtype))
			numericalization_func = self.dtypes[self.dtype]
			# It doesn't make sense to explicitly coerce to a numeric type if
			# the data is sequential, since it's unclear how to coerce padding tokens
			# to a numeric type.
			if not self.sequential:
				arr = [numericalization_func(x) if isinstance(x, str)
					   else x for x in arr]
			if self.postprocessing is not None:
				arr = self.postprocessing(arr, None)

		var = torch.tensor(arr, dtype=self.dtype, device=device)

		if self.sequential and not self.batch_first:
			var.t_()
		if self.sequential:
			var = var.contiguous()

		if self.include_lengths:
			return var, lengths
		return var