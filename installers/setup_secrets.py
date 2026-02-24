#!/usr/bin/env python3
import sys
from pathlib import Path

# Adiciona o diretório atual ao path para importar o utilitário
# O script está em installers/ e precisamos acessar a raiz
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from backend.utils.security import vault, VAULT_FILE

def setup():
    print("=== SLM MedTeacher Secret Setup ===")
    print(f"Este script criará um cofre em: {VAULT_FILE}")
    print("O arquivo ficará FORA da pasta do Google Drive.")
    print("")
    print("Você precisará da sua chave do OpenRouter.")
    print("Obtenha em: https://openrouter.ai/keys")
    print("A chave começa com: sk-or-v1-...")
    print("")

    api_key = input("Cole sua OPENROUTER_API_KEY e pressione Enter: ").strip()

    if not api_key:
        print("❌ Erro: Chave vazia.")
        return

    if not api_key.startswith("sk-or-"):
        print("⚠ Aviso: a chave não começa com 'sk-or-'. Verifique se copiou corretamente.")
        confirm = input("Continuar mesmo assim? (s/N): ").strip().lower()
        if confirm != "s":
            print("Operação cancelada.")
            return

    if vault.encrypt_and_save(api_key):
        print("")
        print("✅ Sucesso! Chave cifrada com Fernet (AES-128-CBC) e salva com segurança.")
        print(f"   Cofre: {VAULT_FILE}")
        print("   Você já pode apagar qualquer menção à chave no arquivo .env.")
    else:
        print("")
        print("❌ Erro ao salvar a chave.")

if __name__ == "__main__":
    setup()
