import matplotlib.pyplot as plt


def plot_heuristics_vs_ml(heuristics_acc, ml_acc, save_path):
    x_axis = [x / 10 for x in range(1, 10)]

    fig, ax = plt.subplots()
    ax.plot(x_axis, heuristics_acc, marker="o", label="Heuristics")
    ax.plot(x_axis, ml_acc, marker="o", label="ML")

    ax.set_xlabel("Threshold")
    ax.set_xlim(0, 1)
    ax.set_xticks([x / 10 for x in range(1, 10)])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.1, 1)
    ax.legend()

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    name_acc = [0.4314456, 0.40685544, 0.38450075, 0.35842027, 0.32414307, 0.27645306, 0.25111773, 0.22503726, 0.21684054]
    value_acc = [0.94560358, 0.82414307, 0.68107303, 0.58122206, 0.50223547, 0.44783905, 0.38971684, 0.32861401, 0.26453055]
    ml_acc = [0.86] * 9

    plot_heuristics_vs_ml(name_acc, ml_acc, "./plots/name_acc.png")
    plot_heuristics_vs_ml(value_acc, ml_acc, "./plots/value_acc.png")
