# unofficial-torchtext-oov-extension
An extension to enable pointer / copy mechanism in torchtext

Codes are adapted from [TORCHTEXT 0.8.0](https://pytorch.org/text/_modules/torchtext/data/field.html)  source code.

The extended classes include (in [legacy](https://github.com/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb)) Field, Batch, BucketIterator and etc.

Torchtext before 0.8.0 only converts vocabulary strings to indices, whereas out-of-vocabulary (OOV) words are recognized as ```'<unk>'```.

To implement pointer / copy mechanism without abandoning Torchtext, we use codes in [copynet](https://github.com/adamklec/copynet) and [pointer_summarizer](https://github.com/atulkum/pointer_summarizer) to write a light-weight extension for torchtext, though it is far from being official.

The classes in this extension are expected to be used in just the same way as standard Torchtext classes should be. Just add 'OOV' after each class in Torchtext would be enough.

Any suggestion is welcome.

A few things to note:
1. To make this copy / pointer mechanism work, There must be a batch named 'src', i.e., a source sequence that contain recognizable out-of-vocabulary words.
2. Vocabulary size of each ```Field ``` instance is preferably the same as others.
3. You may check the code to see how it works, don't panic, only 1/9 of the codes are written, and the rest is just copied from offical Torchtext sources.

## Usage
```python
from oov import BucketIteratorOOV, FieldOOV

spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = FieldOOV(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            include_lengths = True)

TRG = FieldOOV(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

from torchtext.legacy import data
from torchtext.legacy import datasets

fields = {'your_column_name': ('src', SRC), 'another_column_name': ('trg', TRG)}
train_data, test_data = data.TabularDataset.splits(
                            path = your_path,
                            train = 'your-data-train.json',
                            test = 'your-data-test.json',
                            format = 'json',
                            fields = fields
)
train_data, valid_data = train_data.split()

SRC.build_vocab(train_data.src, train_data.trg, min_freq = 2, max_size = 100)
TRG.build_vocab(train_data.src, train_data.trg, min_freq = 2, max_size = 100)

BATCH_SIZE = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIteratorOOV.splits(
    (train_data, valid_data, test_data), 
     batch_size = BATCH_SIZE,
     sort_within_batch = True,
     sort_key = lambda x : len(x.src),
     device = device)
```
then test it
```python
for i, batch in enumerate(train_iterator):
	print(batch.trg.shape)
```

## A customized object
We have also added a customized ```Field``` class, inheriting the ```FieldOOV``` class. This newly customized class is meant for common English texts, and therefore its many initializations are bonded to English, such as its tokenizer. We also let this field, called ```ENGLISHTEXT```, have its own fixed vocabulary. This vocabulary is a combined version from four popular Torchtext datasets, i.e., WikiText2, WikiText103 (part of it), PennTreebank, and SNLI. All these datasets are available on official websites. You may choose to use other datasets. The purpose of this class is mainly to maintain a same set of vocabulary in different models, and the vocabulary size has to be big enough to coverage the most common words, as "an average 20-year-old American knows 42,000 words" (just get this number from the [website](https://www.sciencemag.org/news/2016/08/average-20-year-old-american-knows-42000-words-depending-how-you-count-them)). The drawback is that it takes around 3 minutes to build your vocabulary, exclusive a 10 minutes dataset downloading waiting.

If you'd like to use this class, you no longer need to import ```oov```, as it already does for you. Just try this:
```python
SRC = ENGLISHTEXT(include_lengths = True, build_vocab = True)

TRG = ENGLISHTEXT()

TRG.vocab = SRC.vocab
```
To make it slightly faster, we only build vocabularies for one ```Field``` and copy the vocabularies for the remaining.

Now that your vocabulary is independent from your own datasets, and you will always have the consistent word-to-index pairs. Enjoy!
