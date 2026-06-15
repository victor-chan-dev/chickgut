import re

def fix_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # jself -> jnp.array(self) for specific attributes
    content = re.sub(r'jself\.(t_eval|iDuo_g|DuoV_cm3|JejV_cm3|IlV_cm3)', r'jnp.array(self.\1)', content)
    
    # jself.result_fore.t -> self.fore_t
    content = content.replace('jself.result_fore.t', 'self.fore_t')
    content = content.replace('jself.result_fore.y.T', 'self.fore_y')
    
    with open(path, 'w') as f:
        f.write(content)

for p in ['src/chickgut/anatomy/jejunum.py', 'src/chickgut/anatomy/ileum.py']:
    fix_file(p)

print("Fix applied.")
