<h1 align="center">
    Intents Classification for Neural Text Generation 
    <br/>
</h1>

<p align="center">The project consists of building an intent classifier which goal is pedict to the sequence of labels 
of conversations <br/> </p>

<p align="center">
    <img alt="Test & Lint" src="https://img.shields.io/github/actions/workflow/status/konkinit/intent_classification/test_lint.yaml?label=Lint%20and%20TEST&style=for-the-badge">
</p>

<p align="center">
    <img alt="Licence" src="https://img.shields.io/bower/l/MI?style=for-the-badge"> <img alt="Repo size" src="https://img.shields.io/github/repo-size/konkinit/intent_classification?style=for-the-badge">
</p>


## Getting Started

1. Clone the repository
```bash
git clone https://github.com/konkinit/intent_classification.git
```

2. Upgrade `pip` and install the required packages
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the script `./src/utilis/get_datasets.py` to download these datasets

4. Run the notebook `./notebooks/experimental_results.ipynb`


## References

See the directory `/papers`

[1] Emile Chapuis,Pierre Colombo, Matthieu Labeau, and Chloé Clavel. Code-switched inspired losses for generic spoken
dialog representations. EMNLP 2021

[2] Emile Chapuis,Pierre Colombo, Matteo Manica, Matthieu Labeau, and Chloé Clavel. Hierarchical pre-training for
sequence labelling in spoken dialog. Finding of EMNLP 2020

[3] Tanvi Dinkar, Pierre Colombo , Matthieu Labeau, and Chloé Clavel. The importance of fillers for text representations
of speech transcripts. EMNLP 2020

[4] Hamid Jalalzai, Pierre Colombo , Chloe Clavel, Eric Gaussier, Giovanna Varni, Emmanuel Vignon, and Anne Sabourin.
Heavy-tailed representations, text polarity classification & data augmentation. NeurIPS 2020

[5] Pierre Colombo, Emile Chapuis, Matteo Manica, Emmanuel Vignon, Giovanna Varni, and Chloé Clavel. Guiding attention
in sequence-to-sequence models for dialogue act prediction. (oral) AAAI 2020

[6] Alexandre Garcia,Pierre Colombo, Slim Essid, Florence d’Alché-Buc, and Chloé Clavel. From the token to the review: 
A hierarchical multimodal approach to opinion mining. EMNLP 2020

[7] Pierre Colombo, Wojciech Witon, Ashutosh Modi, James Kennedy, and Mubbasir Kapadia. Affect-driven dialog generation.
NAACL 2019
