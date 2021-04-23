To convert files build with LaTeX use

> pip install pdf2image

~~~~
from pdf2image import convert_from_path as cfp
img = cfp("name.pdf", 500)
for im in img:
    im.save("name.png", "png")
~~~~