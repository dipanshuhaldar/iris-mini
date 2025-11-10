from .utils import load_cfg
from .train import train

def main():
    cfg = load_cfg()
    acc = train(cfg)
    print(f"CV accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
