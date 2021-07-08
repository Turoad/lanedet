import os
from sys import argv

def update_config(path):
    with open(path) as f:
        has_config=False
        for line in f:
            has_config = has_config | ('lr_update_by_epoch' in line)
        if has_config:
            return

    # with open(path, 'a') as f:
    #     f.write('lr_update_by_epoch = False')
        

def main(file_path):
    for root, dirs, files in os.walk(file_path):
        for f in files:
            path = os.path.join(root, f)
            if not path.endswith('.py'):
                continue
            update_config(path)


if __name__ == '__main__':
    main(argv[1])
