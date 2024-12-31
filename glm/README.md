# gLM 

## Set up python environment 
#### using conda
```
conda env create -f environment.yml python==3.10.8
conda activate glm-env
pip install torch==1.12.1+cu116  torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```
This set up was tested using python 3.10.8

## Compute gLM embeddings 
gLM embeddings can be computed using the following steps:


#### 1. Prepare two input files.

a) FASTA file of your proteins (amino acid sequences) in your contig
```
>prot_1
MNYSHDNWSAILAHIGKPEELDTSARNAGALTRRREIRDAATLLRLGLAYGPGGMSLREVTAWAQLHDVA
TLSDVALLKRLRNAADWFGILAAQTLAVRAAVTGCTSGKRLRLVDGTAISAPGGGSAEWRLHMGYDPHTC
QFTDFELTDSRDAERLDRFAQTADEIRIADRGFGSRPECIRSLAFGEADYIVRVHWRGLRWLTAEGMRFD
MMGFLRGLDCGKNGETTVMIGNSGNKKAGAPFPARLIAVSLPPEKALISKTRLLSENRRKGRVVQAETLE
AAGHVLLLTSLPEDEYSAEQVADCYRLRWQIELAFKRLKSLLHLDALRAKEPELAKAWIFANLLAAFLID
DIIQPSLDFPPRSAGSEKKN
>prot_2
MAKQDYYEILGVSKTAEEREIRKAYKRLAMKYHPDRNQGDKEAEAKFKEIKEAYEVLTDSQKRAAYDQYG
HAAFEQGGMGGGGFGGGADFSDIFGDVFGDIFGGGRGRQRAARGADLRYNMELTLEEAVRGVTKEIRIPT
LEECDVCHGSGAKPGTQPQTCPTCHGSGQVQMRQGFFAVQQTCPHCQGRGTLIKDPCNKCHGHGRVERSK
TLSVKIPAGVDTGDRIRLAGEGEAGEHGAPAGDLYVQVQVKQHPIFEREGNNLYCEVPINFAMAALGGEI
EVPTLDGRVKLKVPGETQTGKLFRMRGKGVKSVRGGAQGDLLCRVVVETPVGLNERQKQLLQELQESFGG
PTGEHNSPRSKSFFDGVKKFFDDLTR
```

b) subcontig to protein mapping with orientation
in the following format. 

Where '-' refers to reverse direction and '+' refers to forward direction relative to the rest of the contig. 

Make sure the number of proteins in subcontigs does not exceed max_seq_length = 30. 
```
contig_0  +prot_1;-prot_2;-prot_3;-prot_4;-prot_5;+prot_6;-prot_7;+prot_8;+prot_9;+prot_10;-prot_11;-prot_12;-prot_13;-prot_14;-prot_15;-prot_16;
```
see contig_to_prots.tsv and test.fa in example_data as an example.

#### 2. compute pLM embeddings. 
We use [esm2](https://github.com/facebookresearch/esm) to embed proteins. 
```
cd data
python plm_embed.py example_data/inference_example/test.fa example_data/inference_example/test.esm.embs.pkl
```
we provide the expected output example_data/inference_example/test.esm.embs.pkl for your reference and on a A100 GPU this test example took less than 2 minutes to complete. 
#### 3. batch your data for gLM inference. 
```
cd data
# make output directory
mkdir batched_data  
python batch_data.py example_data/inference_example/test.esm.embs.pkl example_data/inference_example/contig_to_prots.tsv example_data/batched_data
```
The output data directory (batched_data) now contains two files. The output directory (batched_data) which contains batch.pkl and prot_index_dict.pkl files. The former is the input containing your data input embeddings, and the latter contains the dictionary mapping from protein index to protein ID.

we provide the expected output data/example_data/batched_data/ for your reference and this particular test example took us less than 1 minutes to run. 


#### 4. compute gLM embeddings.
```
cd data
python ../gLM/glm_embed.py -d example_data/batched_data -m ../model/glm.bin -b 100 -o test_results
```
If you come across GPU memory errors, try reducing batch size (-b).

gLM embeddings will be saved as *.glm.embs.pkl file in the output directory. 

You can output all inference results (plm_embs/glm_embs/prot_ids/outputs/output_probabilitess) by adding --all_results/-a flag. This will be saved as a *.results.pkl file in the output directory. 

You can also output attention matrices by adding --attention flag. Attentions will be saved for post processing in your ourput directory *.attention.pkl

We provide the expected output in data/test_results/results/batch.pkl.glm.embs.pkl and the expected runtime for this on A100 is ~2 minutes. 

We are working on making inference code available as a colab notebook. so **stay tuned**. 




