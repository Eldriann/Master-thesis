"""Convert the scrapped html to text"""

import glob
import html2text

h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_emphasis = True
h.ignore_images = True
h.ignore_tables = True

for file_path in glob.glob("subject_*.html"):
    with open(file_path, "r") as input_file, open(file_path+'.txt', "w") as output_file:
        data = input_file.read()
        output_file.write(h.handle(data))
