"""
Strip the lines from the subject file
"""

FILE_PATH = 'subjects.txt'
with open(FILE_PATH, "r") as input_file, open(FILE_PATH+'_striped.txt', "w") as output_file:
    lines = input_file.readlines()
    for line in lines:
        output_file.write(line.strip()+'\n')
