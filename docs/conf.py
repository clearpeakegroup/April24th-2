import os
import sys
sys.path.insert(0, os.path.abspath('../backend'))

project = 'FinRL Platform'
copyright = '2024, FinRL'
author = 'FinRL Contributors'
release = '1.0'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'alabaster' 