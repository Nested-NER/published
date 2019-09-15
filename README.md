# TOI-CNN+DTE 

### word embedding download links
>- ACE04/05 : [glove 100d](https://drive.google.com/open?id=1qDmFF0bUKHt5GpANj7jCUmDXgq50QJKw)
>- GENIA : [wikipedia-pubmed-and-PMC-w2v 200d](http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin)

### Environment
#### Python packages
>- python==3.6.1
>- pytorch==1.0.0
>- gensim==3.3.0
>- numpy==1.15.0
>- pandas==0.20.3

**(pip install -r requirements.txt)**

#### CUDA (for GPU)
>- cuda 9.0
>- cudnn 7


### Workflow

**download word embedding from the link and put raw embedding in ./model/word2vec**

>- Step 1. Run transfer_wv.py to convert the **GloVe** file to gensim file format (**only for ACE04/05 dataset**)
>- Step 2. Set parameters in config.py
>- Step 3. Run process_data.py to generate input data
>- Step 4. Run train.py to train ToI-CNN+DTE
>- Step 5. Run test.py to test ToI-CNN+DTE

### Test Best Model:
[best models for three datasets](https://drive.google.com/open?id=1Mmn7SsCMpuMrwJfowLZ75Jb6gleVIjTJ)

Download the best model and put them on "./model" path
