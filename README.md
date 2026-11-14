<div align="center">

# algernon 🐭

**A pure-Python deep learning toolbox, built from scratch.**

*Somewhere between [micrograd](https://github.com/karpathy/micrograd) and [tinygrad](https://github.com/tinygrad/tinygrad).*

![Python](https://img.shields.io/badge/python-pure-3776AB?logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)
![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

</div>

---

## 🧠 What is it?

**algernon** is a toolbox for training deep learning models, written entirely in pure Python. No NumPy, no PyTorch.

It's built in three parts:

| Layer | What it does |
| --- | --- |
| **Frontend** | Tensor creation and manipulation |
| **Autograd** | Automatic differentiation: tracking operations and computing gradients |
| **Backend** | The number crunching that actually runs the operations |

## 🤔 Why?

[micrograd](https://github.com/karpathy/micrograd) on its own is too small. [tinygrad](https://github.com/tinygrad/tinygrad) is cool, but a bit beyond what I want for the moment.

So: **algernon!**

## 🗺️ Roadmap

- [ ] Tensor frontend
- [ ] Autograd engine
- [ ] Backend
- [ ] Train a model end to end
- [ ] **v1:** an MLP built on algernon reaches ≥95% test accuracy on MNIST, with no NumPy or PyTorch in the library

## 📁 Project structure

```
algernon/
├── algernon/   # the library itself
├── examples/   # example scripts and models
└── tests/      # tests
```

## 📜 License

Released under the [MIT License](LICENSE).
