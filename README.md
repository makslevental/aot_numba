# Running

```shell
# setup python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# compile my_module shared object
python demo.py

# compile exec that links my_module
cmake -S . -B build -GNinja
cmake --build build
 
 ./build/aot_numba
# square: 64
# mult: 64
```

# Hacks/Monkey patches

1. [Make all symbols have default visibility](https://github.com/makslevental/aot_numba/blob/4cfad8229fadc2aee0ea976e41fcae741de1f707/demo.py#L92)
2. [Knockout abi tags](https://github.com/makslevental/aot_numba/blob/4cfad8229fadc2aee0ea976e41fcae741de1f707/demo.py#L161)
3. [Build shared library instead of shared object](https://github.com/makslevental/aot_numba/blob/4cfad8229fadc2aee0ea976e41fcae741de1f707/demo.py#L282)
4. [Don't add prefix to name-mangled symbols](https://github.com/makslevental/aot_numba/blob/4cfad8229fadc2aee0ea976e41fcae741de1f707/demo.py#L22)