import subprocess

# 定义要执行的命令列表，每个命令都是一个包含其参数的列表
experiments = [
    {
        "name": "Default Parameters",
        "command": ["uv", "run", "python", "final_train.py"]
    },
     {
        "name": "Batch Size 128",
        "command": ["python", "final_train.py", "--batch_size", "128"]
     },
    {
        "name": "Without RMSNorm",
        "command": ["python", "final_train.py", "--no-rmsnorm"]
    }
]

#依次执行每个实验
for i , exp in enumerate(experiments):
    print(" = " * 60) # 打印 60 个 " = " 做分隔线
    print(f"Running Experiment {i + 1} : {exp['name']}") # "Running Experiment 1 : Default Parameters"
    print(f"Command : {' '.join(exp['command'])}") # 把命令列表拼回字符串
    print(" = " * 60)

    try:
        # 使用 subprocess.run 执行命令
        # check=True 会在命令返回非零退出码（即发生错误）时抛出异常
        subprocess.run(exp["command"], check=True) # subprocess.run 启动一个子进程跑命令，并且阻塞等待——父脚本卡在这一行，直到 final_train.py 跑完（可能是几小时）才继续 check=True：子进程退出时检查退出码（exit code）。惯例是 0 = 成功，非 0 = 出错。正常跑完返回 0 什么都不发生；final_train.py 崩溃（Python 异常、OOM 等）退出码非 0，subprocess 立刻抛出 CalledProcessError 异常。
        print(f"\n--- Experiment '{exp['name']}' finished successfully. ---\n")
    except subprocess.CalledProcessError as e:
        print(f"\n--- Experiment '{exp['name']}' failed with exit code {e.returncode}. ---\n")
        break # 如果一个实验失败，可以选择停止后续实验
    except FileNotFoundError:
        print("\n--- Error: 'uv' not found. Make sure uv is installed and in your PATH. ---\n")
        break

print("All experiments have been run.")