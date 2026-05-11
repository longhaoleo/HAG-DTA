import subprocess
import sys

def export_env_yaml(file_name="environment.yaml"):
    # 1. 获取当前Python版本
    py_version = sys.version.split()[0]
    
    # 2. 获取所有pip安装的包（原生命令）
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    
    # 3. 解析包列表
    dependencies = []
    for line in result.stdout.splitlines():
        if line.strip() and "==" in line:
            dependencies.append(f"    - {line.strip()}")

    # 4. 手动拼接YAML格式（原生字符串，无需任何库）
    yaml_content = f"""python: {py_version}
dependencies:
{chr(10).join(dependencies)}
"""
    # 5. 写入文件
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print(f"✅ 环境导出成功！文件：{file_name}")

if __name__ == "__main__":
    export_env_yaml()