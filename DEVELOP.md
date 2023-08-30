## For Developers

To build locally: 

    python3 -m build

To upload: 

    python3 -m twine upload dist/*

To build the documentation:

    pydoc-markdown -I scgt -m scgt > Documentation.md

To install locally: 

    python3 -m pip install . 
