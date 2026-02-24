import torchxrayvision as xrv
import sys

print("Iniciando verificação de pesos TorchXRayVision...")

try:
    print("1. Tentando carregar DenseNet121 (res224)...")
    model1 = xrv.models.DenseNet(weights="densenet121-res224-all")
    print("   -> SUCESSO: DenseNet carregado.")
except Exception as e:
    print(f"   -> FALHA DenseNet: {e}")

try:
    print("2. Tentando carregar ResNet50 (res512)...")
    model2 = xrv.models.ResNet(weights="resnet50-res512-all")
    print("   -> SUCESSO: ResNet carregado.")
except Exception as e:
    print(f"   -> FALHA ResNet: {e}")

print("Verificação concluída.")
