import os
import sys
import matplotlib.pyplot as plt

def load_csv(path):
    csv_path = os.path.join(path, "train.csv")
    with open(csv_path, "r") as f:
        csv_str = f.read()
    csv = [s.split(",") for s in csv_str.split("\n")]

    if csv[-1] == [""]:
        csv = csv[:-1]

    return csv

def save_fig(path, csv):
    save_path = os.path.join(path, "plot.png")
    x = [int(s[0]) for s in csv]
    train_acc = [float(s[1]) for s in csv]
    train_loss = [float(s[2]) for s in csv]
    val_acc = [float(s[3]) for s in csv]
    val_loss = [float(s[4]) for s in csv]
    lr = [float(s[5]) for s in csv]

    plt.figure(figsize=[12, 12])
    plt.subplots_adjust(hspace=0.5, left=0.0625)

    plt.subplot(3, 1, 1)
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(x, train_acc, label="train_accuracy")
    plt.plot(x, val_acc, label="val_accuracy")
    plt.legend(loc="lower left", bbox_to_anchor=[1.0, 0.5])

    plt.subplot(3, 1, 2)
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(x, train_loss, label="train_loss")
    plt.plot(x, val_loss, label="val_loss")
    plt.legend(loc="lower left", bbox_to_anchor=[1.0, 0.5])

    plt.subplot(3, 1, 3)
    plt.title("Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.plot(x, lr, label="lr")
    plt.legend(loc="lower left", bbox_to_anchor=[1.0, 0.5])

    plt.savefig(save_path, pad_inches=0)

def main(path):
    print("Plotting...")
    csv = load_csv(path)
    save_fig(path, csv)
    print("Finished. saved at", os.path.join(path, "plot.png"))

if __name__ == "__main__":
    path = "runs/train"

    for arg in sys.argv:
        if arg.startswith("path"):
            path = arg.split("=")[1]
            continue
    
    if not os.path.isdir(path):
        print("invalid path:", path)
        sys.exit()
    
    main(path)
    