# CIFAR-10 Tools documentation

This directory contains the Sphinx documentation for the CIFAR-10 Tools package.

## Quick start

### Building locally

```bash
# Install documentation dependencies
poetry install --with docs

# Build HTML documentation
cd docs
sphinx-build -b html source build/html

# Serve locally
python3 -m http.server 8000 --directory build/html
```

Then open http://localhost:8000 in your browser.

## Structure

```
docs/
├── source/              # Documentation source files
│   ├── conf.py         # Sphinx configuration
│   ├── index.rst       # Landing page
│   ├── installation.rst
│   ├── quickstart.rst
│   ├── api/            # API reference
│   │   ├── index.rst
│   │   ├── data.rst
│   │   ├── training.rst
│   │   ├── evaluation.rst
│   │   ├── plotting.rst
│   │   └── hyperparameter_optimization.rst
│   └── notebooks/      # Notebook documentation
│       └── index.rst
├── build/              # Generated documentation (not in git)
├── serve.sh            # Local server script
├── DEPLOYMENT.md       # Deployment guide
└── README.md           # This file
```

## Features

- **Read the Docs Theme**: Professional, responsive design
- **API Documentation**: Auto-generated from docstrings
- **Notebook links**: Links to example notebooks in repository
- **Cross-references**: Intersphinx linking to PyTorch, NumPy, etc.
- **Type Hints**: Enhanced type information display
- **Search**: Full-text search capability

## GitHub Pages

Documentation is automatically deployed to:
https://gperdrizet.github.io/CIFAR10/

The deployment is handled by GitHub Actions (`.github/workflows/docs.yml`) which:
1. Builds the documentation on every push to main
2. Uploads the HTML to GitHub Pages
3. Makes it available at the URL above

## Contributing

When adding new features to the package:

1. **Add docstrings** to all functions, classes, and modules
2. **Update or create** corresponding `.rst` files in `source/api/`
3. **Add examples** to `quickstart.rst` if appropriate
4. **Create notebooks** for complex workflows
5. **Build locally** to verify everything works
6. **Push to main** to deploy

## Troubleshooting

See [DEPLOYMENT.md](DEPLOYMENT.md) for common issues and solutions.

## Extensions used

- `sphinx.ext.autodoc`: Automatic API documentation from docstrings
- `sphinx.ext.napoleon`: Google/NumPy style docstring support
- `sphinx.ext.viewcode`: Links to highlighted source code
- `sphinx.ext.intersphinx`: Links to external documentation
- `sphinx_autodoc_typehints`: Type hint documentation
- `sphinx_rtd_theme`: Read the Docs theme
