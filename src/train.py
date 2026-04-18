import torch # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
import torchvision # type: ignore
import torchvision.transforms as transforms # type: ignore
import matplotlib.pyplot as plt # type: ignore

from model import PrunableNet
from utils import sparsity_loss, evaluate, calculate_sparsity

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)


# 🔥 Plot function (gate distribution)
def plot_gates(model):
    all_gates = []

    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.show()


def train_model(lambda_val):
    model = PrunableNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    for epoch in range(epochs):
        model.train()

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # Classification loss
            cls_loss = F.cross_entropy(outputs, labels)

            # Sparsity loss
            sp_loss = sparsity_loss(model)

            # Total loss
            loss = cls_loss + lambda_val * sp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} done")

    acc = evaluate(model, testloader, device)
    sparsity = calculate_sparsity(model)

    return acc, sparsity, model


if __name__ == "__main__":
    lambdas = [1e-5, 1e-4, 1e-3]

    results = []
    best_model = None
    best_acc = 0

    for lam in lambdas:
        print(f"\n🚀 Training with lambda = {lam}")

        acc, sparsity, model = train_model(lam)

        print(f"Accuracy: {acc:.2f}%")
        print(f"Sparsity: {sparsity:.2f}%")

        results.append((lam, acc, sparsity))

        # Save best model for plotting
        if acc > best_acc:
            best_acc = acc
            best_model = model

    print("\n📊 Final Results:")
    for r in results:
        print(f"Lambda={r[0]} | Accuracy={r[1]:.2f}% | Sparsity={r[2]:.2f}%")

    # 🔥 Plot gate distribution for best model
    print("\n📈 Plotting gate distribution for best model...")
    plot_gates(best_model)