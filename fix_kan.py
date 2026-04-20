with open("Finance/02_核心代码/源代码/khaos/模型定义/kan.py", "r") as f:
    lines = f.readlines()

for i in range(588, 615):
    line = lines[i]
    if line.strip():
        # find the right indentation, should be 12 spaces for everything under if
        # except some comments
        if not line.startswith("            "):
            lines[i] = "            " + line.lstrip()

with open("Finance/02_核心代码/源代码/khaos/模型定义/kan.py", "w") as f:
    f.writelines(lines)
