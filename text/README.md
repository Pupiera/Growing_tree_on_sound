# Growing_tree_on_sound
Code for the paper Growing tree on sound accepted at ACL 2024.


## Installation

1. Install a 3.9.7 python environment, with [virtualenv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/).
2. clone the current repository
3. Install Parsebrain (Custom framework)
    ```
    git clone https://github.com/Pupiera/ParseBrain.git .
    pip install -e . 
    ```
4. Install python dependency : pip install -r requirement.txt 

## Launch any experiment:
    ```
    python3 <train_script.py> <hparam.yaml>
    ```
## Data Preprocessing

Follow this step to preprocess the data: 
- Download the audio file directly from the CEFC-ORFEO
repository : https://www.ortolang.fr/market/corpora/cefc-orfeo.
- split each audio file based on the given timestamp (in the conllu file)
 (you can use the splitWavOnTimestamp.py script)
    ```
    python splitWavOnTimestamp.py <path_to_CEFC-ORFEO>/11/oral/ 
    ```
- Change the path of the audio using sed

