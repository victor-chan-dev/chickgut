import re

files = [
    "src/chickgut/anatomy/duodenum.py",
    "src/chickgut/anatomy/jejunum.py",
    "src/chickgut/anatomy/ileum.py"
]

for f in files:
    with open(f, "r") as file:
        content = file.read()
    
    # Check if adjoint is already injected
    if "diffrax.DirectAdjoint()" not in content:
        # We find "diffrax.diffeqsolve(" and append the adjoint inside
        # Since diffeqsolve might be split across lines, we can just replace "diffrax.diffeqsolve(" 
        # with "diffrax.diffeqsolve(adjoint=diffrax.DirectAdjoint(),"
        content = content.replace("diffrax.diffeqsolve(", "diffrax.diffeqsolve(adjoint=diffrax.DirectAdjoint(), ")
        
        with open(f, "w") as file:
            file.write(content)
        print(f"Injected adjoint into {f}")

