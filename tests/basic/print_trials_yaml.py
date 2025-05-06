import sys
import yaml

with open(sys.argv[1], 'r') as f:
    print(f.read())
    print('---')
    print('YAML loaded:', yaml.safe_load(f))
