import re

with open('src/chickgut/anatomy/duodenum.py', 'r') as f:
    content = f.read()

# Fix jself typos
content = content.replace('jself.iDuo_g', 'jnp.array(self.iDuo_g)')
content = content.replace('jself.DuoV_cm3', 'jnp.array(self.DuoV_cm3)')
content = content.replace('jself.t_eval', 'jnp.array(self.t_eval)')
content = content.replace('jself.result_fore.t', 'self.fore_t')
content = content.replace('jself.result_fore.y.T', 'self.fore_y')

with open('src/chickgut/anatomy/duodenum.py', 'w') as f:
    f.write(content)
