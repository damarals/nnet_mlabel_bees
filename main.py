#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Daniel Silva"
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Daniel Silva", "Rafael Braga", "Danielo Gomes", "Juvêncio S. Nobre",
               "João P. Vale"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Daniel A. Silva"
__email__ = "danielamaral@alu.ufc.br"
__status__ = "Development"


# Download of Segmented Bees Dataset
import requests, zipfile, io
r = requests.get('https://www.dropbox.com/s/0htmeoie69q650p/miml_dataset.zip')
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('./data')