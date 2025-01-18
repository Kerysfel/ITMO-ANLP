import os

def print_repo_structure(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in ('vectors', 'data', 'volumes','.git', 'archive')]
        level = dirpath.replace(root_dir, '').count(os.sep)
        indent = '    ' * level
        
        print(f"{indent}{os.path.basename(dirpath)}/")
        
        subindent = '    ' * (level + 1)
        for f in filenames:
            print(f"{subindent}{f}")

print_repo_structure(".")