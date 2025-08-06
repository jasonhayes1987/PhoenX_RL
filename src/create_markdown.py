import os
import argparse
from pathlib import Path

def create_markdown_from_python_files(directory, output_file, depth=None, include_hidden=False):
    """
    Recursively scan a directory for Python files and write their content to a markdown file.
    
    Args:
        directory (str): Directory to scan
        output_file (str): Output markdown file path
        depth (int, optional): Maximum directory depth to scan
        include_hidden (bool): Whether to include hidden files and directories (starting with .)
    """
    directory_path = Path(directory).resolve()
    
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(f"# Python Files in {directory_path}\n\n")
        
        # Get all Python files in the directory and subdirectories
        all_files = []
        for root, dirs, files in os.walk(directory_path):
            # Skip hidden directories if not included
            if not include_hidden:
                dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            # Calculate current depth
            current_depth = len(Path(root).relative_to(directory_path).parts)
            if depth is not None and current_depth > depth:
                continue
                
            for file in files:
                # Skip hidden files if not included
                if not include_hidden and file.startswith('.'):
                    continue
                
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    all_files.append(file_path)
        
        # Sort files for consistent output
        all_files.sort()
        
        # Write each file to the markdown
        for file_path in all_files:
            rel_path = file_path.relative_to(directory_path)
            out_file.write(f"## {rel_path}\n\n")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as py_file:
                    content = py_file.read()
                
                out_file.write("```python\n")
                out_file.write(content)
                out_file.write("\n```\n\n")
                
                out_file.write("---\n\n")
            except Exception as e:
                out_file.write(f"Error reading file: {e}\n\n")
                out_file.write("---\n\n")
    
    print(f"Created markdown file with {len(all_files)} Python files at: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert Python files in a directory to a markdown file')
    parser.add_argument('directory', help='Directory to scan for Python files')
    parser.add_argument('--output', '-o', default='python_files.md', help='Output markdown file')
    parser.add_argument('--depth', '-d', type=int, help='Maximum directory depth to scan')
    parser.add_argument('--include-hidden', '-i', action='store_true', help='Include hidden files and directories')
    
    args = parser.parse_args()
    create_markdown_from_python_files(args.directory, args.output, args.depth, args.include_hidden)

if __name__ == "__main__":
    main()