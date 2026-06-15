import re

for p in ['src/chickgut/anatomy/duodenum.py', 'src/chickgut/anatomy/jejunum.py', 'src/chickgut/anatomy/ileum.py']:
    with open(p, 'r') as f:
        content = f.read()
    content = content.replace('j[0.0]', 'jnp.array([0.0])')
    with open(p, 'w') as f:
        f.write(content)
