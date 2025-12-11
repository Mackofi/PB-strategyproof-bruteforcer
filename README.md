# PB-strategyproof-bruteforcer
A simple program that tries to find examples which break strategyproofness using different algorithms. Used in the BSc thesis "A közösségi költségvetés matematikája".

## Setup

```powershell
py -m pip install -r requirements.txt
```

## Run

Running the program and setting the configuration variables is done with the following code:
```powershell
py main.py --max-project-count 3 --max-value-per-project 4 --max-voters 6
```

The script stops as soon as it finds a configuration where the tweaked first ballot yields a higher satisfaction *under the honest ballot*. If no such configuration exists within the searched space, it prints a message at the end.
