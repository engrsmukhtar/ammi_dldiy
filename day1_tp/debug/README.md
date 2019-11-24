# Debug me!

## Context
This code, meant to train a simple network on MNIST, is bugged on purpose. Your job is to debug it. Some advice:
- Solve the bugs step by step
- Use the `print()` function or, more advanced, the `pdb` statement (see Python's official doc)

To launch the program with a specific random seed, simply run
```bash
python main.py --seed 1
```

## Random seeds
A random seed allows to generate *deterministic* random data. For example, with the same seed, `torch.randperm(10)` will always output the same permutation.

## Objective
Once you're done, the code will print the first loss, report it for the random seeds 1, 42 and 2019.
