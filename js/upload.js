/**
 * Upload Manager
 * Gerencia drag-and-drop e seleção de arquivos
 */

class UploadManager {
    constructor() {
        this.files = [];
        this.maxFiles = 5;
        this.maxSizeMB = 10;
        
        this.dropzone = null;
        this.fileInput = null;
        this.filePreview = null;
        this.uploadZone = null;
        
        this.onFilesChanged = null; // Callback
    }
    
    /**
     * Inicializa o gerenciador de uploads
     */
    init() {
        this.dropzone = document.getElementById('dropzone');
        this.fileInput = document.getElementById('file-input');
        this.filePreview = document.getElementById('file-preview');
        this.uploadZone = document.getElementById('upload-zone');
        
        if (!this.dropzone || !this.fileInput) {
            console.warn('Upload elements not found');
            return;
        }
        
        this.setupEventListeners();
    }
    
    /**
     * Configura event listeners
     */
    setupEventListeners() {
        // Click na dropzone abre file picker
        this.dropzone.addEventListener('click', () => {
            this.fileInput.click();
        });
        
        // Drag and drop
        this.dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropzone.classList.add('drag-over');
        });
        
        this.dropzone.addEventListener('dragleave', () => {
            this.dropzone.classList.remove('drag-over');
        });
        
        this.dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropzone.classList.remove('drag-over');
            
            const files = Array.from(e.dataTransfer.files);
            this.handleFiles(files);
        });
        
        // File input change
        this.fileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            this.handleFiles(files);
            
            // Reset input para permitir selecionar mesmo arquivo novamente
            e.target.value = '';
        });
    }
    
    /**
     * Processa arquivos selecionados
     */
    handleFiles(newFiles) {
        // Validar número total de arquivos
        if (this.files.length + newFiles.length > this.maxFiles) {
            this.showError(`Maximum ${this.maxFiles} files allowed`);
            return;
        }
        
        // Validar cada arquivo
        const validFiles = [];
        
        for (const file of newFiles) {
            // Validar tamanho
            const sizeMB = file.size / (1024 * 1024);
            if (sizeMB > this.maxSizeMB) {
                this.showError(`File "${file.name}" is too large (${sizeMB.toFixed(1)}MB). Max: ${this.maxSizeMB}MB`);
                continue;
            }
            
            // Validar tipo
            const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'application/pdf'];
            if (!validTypes.includes(file.type)) {
                this.showError(`File "${file.name}" has unsupported format`);
                continue;
            }
            
            validFiles.push(file);
        }
        
        // Adicionar arquivos válidos
        this.files.push(...validFiles);
        
        // Atualizar preview
        this.updatePreview();
        
        // Notificar callback
        if (this.onFilesChanged) {
            this.onFilesChanged(this.files);
        }
    }
    
    /**
     * Atualiza preview de arquivos
     */
    updatePreview() {
        if (this.files.length === 0) {
            this.filePreview.style.display = 'none';
            this.filePreview.innerHTML = '';
            return;
        }
        
        this.filePreview.style.display = 'flex';
        this.filePreview.innerHTML = '';
        
        this.files.forEach((file, index) => {
            const item = document.createElement('div');
            item.className = 'file-preview-item';
            
            // Ícone baseado no tipo
            const icon = file.type.startsWith('image/') ? '🖼️' : '📄';
            
            // Nome truncado
            let displayName = file.name;
            if (displayName.length > 30) {
                displayName = displayName.substring(0, 27) + '...';
            }
            
            // Tamanho
            const sizeKB = (file.size / 1024).toFixed(0);
            
            item.innerHTML = `
                <span>${icon} ${displayName} (${sizeKB}KB)</span>
                <button class="file-remove-btn" data-index="${index}" title="Remove file">✕</button>
            `;
            
            this.filePreview.appendChild(item);
        });
        
        // Event listeners para botões de remover
        this.filePreview.querySelectorAll('.file-remove-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const index = parseInt(btn.dataset.index);
                this.removeFile(index);
            });
        });
    }
    
    /**
     * Remove arquivo por índice
     */
    removeFile(index) {
        this.files.splice(index, 1);
        this.updatePreview();
        
        if (this.onFilesChanged) {
            this.onFilesChanged(this.files);
        }
    }
    
    /**
     * Limpa todos os arquivos
     */
    clearFiles() {
        this.files = [];
        this.updatePreview();
        
        if (this.onFilesChanged) {
            this.onFilesChanged(this.files);
        }
    }
    
    /**
     * Obtém arquivos atuais
     */
    getFiles() {
        return this.files;
    }
    
    /**
     * Verifica se há arquivos
     */
    hasFiles() {
        return this.files.length > 0;
    }
    
    /**
     * Mostra/esconde zona de upload
     */
    toggle() {
        if (this.uploadZone.style.display === 'none') {
            this.uploadZone.style.display = 'block';
        } else {
            this.uploadZone.style.display = 'none';
        }
    }
    
    /**
     * Mostra mensagem de erro
     */
    showError(message) {
        // Usar sistema de toast se disponível
        if (window.AppManager && window.AppManager.showToast) {
            window.AppManager.showToast(message, 'error');
        } else {
            alert(message);
        }
    }
}

// Disponibilizar globalmente
window.UploadManager = UploadManager;
