#!/usr/bin/env python3
"""
Instalador Universal - SLM Local Medteacher
Windows, Linux e macOS

Autor: Dr. Aydamari Faria Jr.
Data: 2026-02-14
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

# Cores para terminal (compatível multiplataforma)
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text, color=Colors.OKGREEN):
    """Imprime texto colorido (funciona no Windows 10+)"""
    if platform.system() == "Windows":
        # Habilita ANSI no Windows
        os.system('')
    print(f"{color}{text}{Colors.ENDC}")

def check_python_version():
    """Verifica se Python >= 3.9"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_colored(
            f"❌ Erro: Python 3.9+ é necessário. Versão atual: {version.major}.{version.minor}",
            Colors.FAIL
        )
        sys.exit(1)
    print_colored(f"✓ Python {version.major}.{version.minor}.{version.micro} detectado", Colors.OKGREEN)

def check_ollama():
    """Verifica se Ollama está instalado"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_colored(f"✓ Ollama instalado: {result.stdout.strip()}", Colors.OKGREEN)
            return True
    except FileNotFoundError:
        pass
    
    print_colored("⚠️  Ollama não detectado", Colors.WARNING)
    print_colored("   Baixe em: https://ollama.ai/download", Colors.OKBLUE)
    return False

def create_venv():
    """Cria ambiente virtual Python"""
    print_colored("\n📦 Criando ambiente virtual...", Colors.HEADER)
    
    venv_path = Path("venv")
    if venv_path.exists():
        print_colored("   Ambiente virtual já existe. Removendo...", Colors.WARNING)
        import shutil
        shutil.rmtree(venv_path)
    
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    print_colored("✓ Ambiente virtual criado", Colors.OKGREEN)

def get_pip_executable():
    """Retorna caminho do pip no venv conforme SO"""
    if platform.system() == "Windows":
        return Path("venv") / "Scripts" / "pip.exe"
    else:
        return Path("venv") / "bin" / "pip"

def install_dependencies():
    """Instala dependências Python"""
    print_colored("\n📚 Instalando dependências...", Colors.HEADER)
    
    pip_path = get_pip_executable()
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
    
    print_colored("✓ Dependências instaladas", Colors.OKGREEN)

def download_model():
    """Baixa modelo MedGemma via Ollama"""
    print_colored("\n🧠 Baixando modelo MedGemma 1.5 4B (Q4_K_M)...", Colors.HEADER)
    print_colored("   Tamanho: ~3GB - Pode demorar alguns minutos", Colors.WARNING)

    model_name = "thiagomoraes/medgemma-1.5-4b-it:Q4_K_M"
    
    try:
        subprocess.run(['ollama', 'pull', model_name], check=True)
        print_colored(f"✓ Modelo {model_name} baixado", Colors.OKGREEN)
    except subprocess.CalledProcessError:
        print_colored(f"❌ Erro ao baixar modelo", Colors.FAIL)
        return False
    return True

def create_modelfiles():
    """Cria modelfiles customizados do Ollama"""
    print_colored("\n⚙️  Criando modelfiles customizados...", Colors.HEADER)
    
    modelfiles_dir = Path("modelfiles")
    model_base = "thiagomoraes/medgemma-1.5-4b-it:Q4_K_M"
    
    modelfiles = {
        "medgemma-simulador": """FROM {model}

SYSTEM \"\"\"Você é um simulador de casos clínicos para estudantes de medicina.
Seu papel é gerar casos realistas, atuar como paciente virtual e fornecer feedback estruturado.
Mantenha alta fidelidade clínica e adapte-se ao nível do estudante.
\"\"\"

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
""",
        
        "medgemma-raciocinio": """FROM {model}

SYSTEM \"\"\"Você é um preceptor clínico especializado em raciocínio diagnóstico.
Analise casos clínicos complexos, imagens médicas e exames complementares.
Guie o raciocínio usando método socrático, sem dar diagnósticos diretos.
Cite fundamentos fisiopatológicos e princípios de medicina baseada em evidências.
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER num_ctx 16384
""",
        
        "medgemma-comunicacao": """FROM {model}

SYSTEM \"\"\"Você é um especialista em comunicação médico-paciente.
Ajude estudantes a explicar conceitos médicos em linguagem acessível.
Use princípios de Calgary-Cambridge, SPIKES e health literacy.
Sugira analogias, teach-back questions e adaptações culturais.
\"\"\"

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
"""
    }
    
    created_count = 0
    for name, content in modelfiles.items():
        filepath = modelfiles_dir / f"{name}.modelfile"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.format(model=model_base))
        
        # Criar modelo no Ollama
        try:
            subprocess.run(['ollama', 'create', name, '-f', str(filepath)], check=True)
            print_colored(f"   ✓ {name}", Colors.OKGREEN)
            created_count += 1
        except subprocess.CalledProcessError:
            print_colored(f"   ❌ Erro ao criar {name}", Colors.FAIL)
    
    if created_count == len(modelfiles):
        print_colored(f"✓ {created_count} modelfiles criados", Colors.OKGREEN)
        return True
    return False

def create_directories():
    """Cria diretórios necessários"""
    print_colored("\n📁 Criando estrutura de diretórios...", Colors.HEADER)
    
    dirs = [
        "css", "js", "backend/models", "backend/utils",
        "prompts", "modelfiles", "sessions"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Criar __init__.py nos pacotes Python
    (Path("backend") / "__init__.py").touch()
    (Path("backend/models") / "__init__.py").touch()
    (Path("backend/utils") / "__init__.py").touch()
    
    print_colored("✓ Diretórios criados", Colors.OKGREEN)

def verify_installation():
    """Verifica se instalação está completa"""
    print_colored("\n🔍 Verificando instalação...", Colors.HEADER)
    
    checks = {
        "Ambiente virtual": Path("venv").exists(),
        "Dependências": get_pip_executable().exists(),
        "Modelfiles": all((Path("modelfiles") / f"{m}.modelfile").exists() 
                         for m in ["medgemma-simulador", "medgemma-raciocinio", "medgemma-comunicacao"]),
    }
    
    all_ok = True
    for item, status in checks.items():
        if status:
            print_colored(f"   ✓ {item}", Colors.OKGREEN)
        else:
            print_colored(f"   ❌ {item}", Colors.FAIL)
            all_ok = False
    
    return all_ok

def main():
    """Fluxo principal de instalação"""
    print_colored("=" * 60, Colors.HEADER)
    print_colored("    INSTALADOR - SLM MÉDICO LOCAL", Colors.HEADER + Colors.BOLD)
    print_colored("    MedGemma 1.5 4B + FastAPI", Colors.HEADER)
    print_colored("=" * 60, Colors.HEADER)
    
    print_colored(f"\nSistema operacional: {platform.system()} {platform.release()}", Colors.OKBLUE)
    
    # Verificações
    check_python_version()
    ollama_installed = check_ollama()
    
    if not ollama_installed:
        print_colored("\n⚠️  Instale o Ollama antes de continuar", Colors.WARNING)
        response = input("Continuar mesmo assim? (s/N): ").lower()
        if response != 's':
            sys.exit(0)
    
    # Instalação
    try:
        create_directories()
        create_venv()
        install_dependencies()
        
        if ollama_installed:
            if download_model():
                create_modelfiles()
        
        # Verificação final
        if verify_installation():
            print_colored("\n" + "=" * 60, Colors.OKGREEN)
            print_colored("✓ INSTALAÇÃO CONCLUÍDA COM SUCESSO!", Colors.OKGREEN + Colors.BOLD)
            print_colored("=" * 60, Colors.OKGREEN)
            
            print_colored("\n📖 PRÓXIMOS PASSOS:", Colors.HEADER)
            
            if platform.system() == "Windows":
                print_colored("   1. Ativar ambiente: venv\\Scripts\\activate", Colors.OKBLUE)
            else:
                print_colored("   1. Ativar ambiente: source venv/bin/activate", Colors.OKBLUE)
            
            print_colored("   2. Iniciar servidor: python backend/main.py", Colors.OKBLUE)
            print_colored("   3. Abrir navegador: http://localhost:8000/medteacher.html", Colors.OKBLUE)
            
        else:
            print_colored("\n❌ Instalação incompleta. Verifique os erros acima.", Colors.FAIL)
            sys.exit(1)
    
    except KeyboardInterrupt:
        print_colored("\n\n⚠️  Instalação cancelada pelo usuário", Colors.WARNING)
        sys.exit(0)
    except Exception as e:
        print_colored(f"\n❌ Erro durante instalação: {e}", Colors.FAIL)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
