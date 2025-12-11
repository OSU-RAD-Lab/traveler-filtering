## Running
### Pixi
1. Install pixi from [https://prefix.dev/](https://prefix.dev/)
```
curl -fsSL https://pixi.sh/install.sh | bash
```
2. Clone repo and run
```
git@github.com:gandres42/sand-sieve.git
cd sand-sieve
pixi run dynamics
```

### Conda
1. Install conda
2. Create new environment

```
conda env create -f environment.yml
```
3. Activate env
```
conda activate sand-sieve
```

4. Clone repo and run
```
git@github.com:gandres42/sand-sieve.git
cd sand-sieve
python py/main.py
```