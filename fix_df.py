import re
import sys

def fix_file(path):
    with open(path, 'r') as f:
        content = f.read()
    
    # Replace float(x) with x for ys[-1, 0] or ys[-1, -1]
    content = re.sub(r'float\(([^)]+\.ys\[[^\]]+\])\)', r'\1', content)
    
    # Replace self.init_...[0] = ... with y0_... = jnp.zeros...
    content = re.sub(r'self\.init_Duo_CPu\[0\] = (.*)\n.*self\.init_Duo_CPsl\[0\] = (.*)\n.*\n.*y0_u = jnp\.array\(self\.init_Duo_CPu\)\n.*y0_sl = jnp\.array\(self\.init_Duo_CPsl\)', 
                     r'y0_u = jnp.zeros(len(self.init_Duo_CPu)).at[0].set(\1)\n        y0_sl = jnp.zeros(len(self.init_Duo_CPsl)).at[0].set(\2)', content)
    
    content = re.sub(r'self\.init_Duo_CPr\[0\] = (.*)\n.*y0_r = jnp\.array\(self\.init_Duo_CPr\)', 
                     r'y0_r = jnp.zeros(len(self.init_Duo_CPr)).at[0].set(\1)', content)

    content = re.sub(r'self\.init_Jej_CPu\[0\] = (.*)\n.*self\.init_Jej_CPsl\[0\] = (.*)\n.*\n.*y0_u = jnp\.array\(self\.init_Jej_CPu\)\n.*y0_sl = jnp\.array\(self\.init_Jej_CPsl\)', 
                     r'y0_u = jnp.zeros(len(self.init_Jej_CPu)).at[0].set(\1)\n        y0_sl = jnp.zeros(len(self.init_Jej_CPsl)).at[0].set(\2)', content)
    
    content = re.sub(r'self\.init_Jej_CPr\[0\] = (.*)\n.*\n.*y0_r = jnp\.array\(self\.init_Jej_CPr\)', 
                     r'y0_r = jnp.zeros(len(self.init_Jej_CPr)).at[0].set(\1)', content)

    content = re.sub(r'self\.init_Il_CPu\[0\] = (.*)\n.*self\.init_Il_CPsl\[0\] = (.*)\n.*\n.*y0_u = jnp\.array\(self\.init_Il_CPu\)\n.*y0_sl = jnp\.array\(self\.init_Il_CPsl\)', 
                     r'y0_u = jnp.zeros(len(self.init_Il_CPu)).at[0].set(\1)\n        y0_sl = jnp.zeros(len(self.init_Il_CPsl)).at[0].set(\2)', content)
    
    content = re.sub(r'self\.init_Il_CPr\[0\] = (.*)\n.*\n.*y0_r = jnp\.array\(self\.init_Il_CPr\)', 
                     r'y0_r = jnp.zeros(len(self.init_Il_CPr)).at[0].set(\1)', content)

    # Replace pd.DataFrame({ with {
    content = content.replace('pd.DataFrame({', '{')
    # Replace np.array(X) with X
    content = re.sub(r'np\.array\(([^)]+)\)', r'\1', content)
    # Replace }) with }
    content = re.sub(r'\}\)', r'}', content)

    with open(path, 'w') as f:
        f.write(content)

for p in ['src/chickgut/anatomy/duodenum.py', 'src/chickgut/anatomy/jejunum.py', 'src/chickgut/anatomy/ileum.py']:
    fix_file(p)
