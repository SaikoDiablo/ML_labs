import platform
import sys

packages = [
    "torch",
    "torchvision",
    "mlflow",
    "yaml",
    "matplotlib",
    "numpy",
    "pandas",
    "sklearn",
]

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print()

for name in packages:
    try:
        if name == "yaml":
            import yaml as mod
        elif name == "sklearn":
            import sklearn as mod
        else:
            mod = __import__(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name}=={version}")
    except Exception as e:
        print(f"{name}: not installed ({e})")
