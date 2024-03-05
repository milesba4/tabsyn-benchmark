# Open the original log file for reading
with open('output.log', 'r') as file:
    # Read the contents of the file
    contents = file.read()

# Replace every ']' with ']\n'
contents = contents.replace(']', ']\n')

# Open a new file (or overwrite the original) for writing
with open('output_modified.log', 'w') as file:
    # Write the modified contents back to the file
    file.write(contents)
