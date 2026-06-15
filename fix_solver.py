import os

files = [
    "src/chickgut/anatomy/duodenum.py",
    "src/chickgut/anatomy/jejunum.py",
    "src/chickgut/anatomy/ileum.py"
]

for f in files:
    with open(f, "r") as file:
        content = file.read()
    
    content = content.replace("solver = diffrax.Kvaerno5()", "solver = diffrax.Tsit5()")
    
    with open(f, "w") as file:
        file.write(content)

print("Replaced Kvaerno5 with Tsit5")
