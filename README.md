# BERT-KorQuAD-Competition
#### Description: Machine Reading Comprehension Competition w/ Korean BERT Model

This implementation is for [AI NLP Challenge](https://challenge.enliple.com/?fbclid=IwAR2X5gqDqffUHzooGjYha1dzrxG3wsIs6qxd2naMqc6BIXwtCz7zgYqFUlk).

This code is reimplemented as a fork of [huggingface/transformers](https://github.com/huggingface/transformers) and [korquad-challenge](https://github.com/enlipleai/korquad-challenge).

```bibtex
@article{Wolf2019HuggingFacesTS,
  title={HuggingFace's Transformers: State-of-the-art Natural Language Processing},
  author={Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and R'emi Louf and Morgan Funtowicz and Jamie Brew},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.03771}
}
```

```bibtex
@article{ham2020kornli,
  title={KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  author={Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal={arXiv preprint arXiv:2004.03289},
  year={2020}
}
```

```bibtex
@misc{whang2019domain,
    title={Domain Adaptive Training BERT for Response Selection},
    author={Taesun Whang and Dongyub Lee and Chanhee Lee and Kisu Yang and Dongsuk Oh and HeuiSeok Lim},
    year={2019},
    eprint={1908.04812},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


### 1. Post Training Data Generation 
--------

Generate KorQuAD corpus whatever the way you want. The KorQuAD corpus I've generated is on `./rsc/corpus/`.

```shellscript
sh create_post_training_data.sh
```
After, keep the output file on `./rsc/corpus`.
