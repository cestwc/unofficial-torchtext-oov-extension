# unofficial-torchtext-oov-extension
An extension to enable pointer / copy mechanism in torchtext

Codes are adapted from [TORCHTEXT 0.8.0](https://pytorch.org/text/_modules/torchtext/data/field.html)  source code.

The extended classes include (in [legacy](https://github.com/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb)) Field, Batch, BucketIterator and etc.

Torchtext before 0.8.0 only converts vocabulary strings to indices, whereas out-of-vocabulary (OOV) words are recognized as ```'<unk>'```.

To implement pointer / copy mechanism without abandoning TORCHTEXT, we use codes in [copynet](https://github.com/adamklec/copynet) and [pointer_summarizer](https://github.com/atulkum/pointer_summarizer) to write a light-weight extension for torchtext, though it is far from being official.

Any suggestion is welcome.