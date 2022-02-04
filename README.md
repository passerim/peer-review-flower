# Peer-reviewed Federated Learning

## Installazione

Creazione dell'ambiente virtuale con conda.
```shell
conda create -n flenv python=3.8 ipykernel
```

Attivazione dell'ambiente virtuale appena creato.
```shell
conda activate flenv
```

Installazione dipendenze richieste dal progetto.
```shell
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install git+https://github.com/passerim/flower.git@extended_modules
```

Per eliminare l'ambiente virtuale.
```shell
conda deactivate
conda env remove --name flenv
```

## Esecuzione

Per eseguire gli esempi di training centralizzato e federato eseguire gli script ```run.sh``` nelle rispettive directory. 

## Test

Per eseguire il test centralizzato.
```shell
python -m tests.test_centralized
```

Per eseguire il test federato.
```shell
python -m tests.test_federated
```
