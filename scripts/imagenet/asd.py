import os
import re

def extract_parameters(file_content):
    # Regex pattern to extract parameters, handling quotes and spaces
    pattern = re.compile(r'--(\w[-\w]*)=("[^"]+"|\'[^\']+\'|\S+)')
    return pattern.findall(file_content)

def format_value(value):
    # Remove surrounding quotes from value
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    elif value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    
    # Check if the value is a list (space-separated)
    if ' ' in value:
        items = value.split(' ')
        formatted_value = "\n".join([f"  - {item}" for item in items])
        return f"\n{formatted_value}"  # Add a line break before the list
    else:
        return value

def create_md_file(sh_file, parameters):
    md_file = f"../../configs/imagenet/{sh_file.replace('.sh', '.md')}"
    with open(md_file, 'w') as f:
        f.write("# Hyperparameters\n\n")
        for param, value in parameters:
            formatted_value = format_value(value)
            f.write(f"- {param}: {formatted_value}\n")

def main():
    # List all .sh files in the current directory
    sh_files = [f for f in os.listdir('.') if f.endswith('.sh')]

    for sh_file in sh_files:
        with open(sh_file, 'r') as f:
            content = f.read()

        parameters = extract_parameters(content)
        create_md_file(sh_file, parameters)

if __name__ == "__main__":
    main()
