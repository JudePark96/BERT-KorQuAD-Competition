# BERT-KorQuAD-Competition
#### Description: Machine Reading Comprehension Competition w/ Korean BERT Model

This implementation is for [AI NLP Challenge](https://challenge.enliple.com/?fbclid=IwAR2X5gqDqffUHzooGjYha1dzrxG3wsIs6qxd2naMqc6BIXwtCz7zgYqFUlk).

This code is reimplemented as a fork of [huggingface/transformers](https://github.com/huggingface/transformers)

```
@article{Wolf2019HuggingFacesTS,
  title={HuggingFace's Transformers: State-of-the-art Natural Language Processing},
  author={Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and R'emi Louf and Morgan Funtowicz and Jamie Brew},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.03771}
}
```


### 1. Post Training Data Generation 
--------

Generate KorQuAD corpus whatever the way you want. The KorQuAD corpus I've generated is on `./rsc/corpus/`.

```shellscript
sh create_post_training_data.sh
```
After, keep the output file on `./rsc/corpus`.
