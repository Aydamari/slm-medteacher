/**
 * SLM MedTeacher - Main Application
 * Gerencia toda a lógica da aplicação
 */

class AppManager {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentSessionId = null;
        this.currentLanguage = 'en';
        this.currentMode = 'clinical-reasoning';
        this.currentTier = 'local_4b';
        this.currentLlmModel = 'gemini_2.5_flash';
        this.selectedExams = [];
        this.elements = {};
        this.uploadManager = new UploadManager();
        
        this.messages = {
            en: {
                processingMessage: 'Processing your message...',
                processingFiles: 'Processing files...',
                sessionCreated: 'Session created successfully',
                sessionExported: 'Session exported successfully',
                errorOccurred: 'An error occurred',
                connectionError: 'Cannot connect to server. Is it running?'
            },
            pt: {
                processingMessage: 'Processando sua mensagem...',
                processingFiles: 'Processando arquivos...',
                sessionCreated: 'Sessão criada com sucesso',
                sessionExported: 'Sessão exportada com sucesso',
                errorOccurred: 'Ocorreu um erro',
                connectionError: 'Não foi possível conectar ao servidor. Ele está rodando?'
            },
        };
    }
    
    init() {
        this.cacheElements();
        this.setupEventListeners();
        this.uploadManager.init();
        this.uploadManager.onFilesChanged = (files) => {
            console.log('Files changed:', files.length);
        };
        console.log('✅ MedTeacher initialized');
    }
    
    cacheElements() {
        this.elements = {
            welcomeScreen: document.getElementById('welcome-screen'),
            chatInterface: document.getElementById('chat-interface'),
            languageSelect: document.getElementById('language-select'),
            tierSelect: document.getElementById('tier-select'),
            llmModelGroup: document.getElementById('llm-model-group'),
            llmModelSelect: document.getElementById('llm-model-select'),
            modeSelect: document.getElementById('mode-select'),
            examSelectionGroup: document.getElementById('exam-selection-group'),
            startSessionBtn: document.getElementById('start-session-btn'),
            newSessionBtn: document.getElementById('new-session-btn'),
            exportSessionBtn: document.getElementById('export-session-btn'),
            sendBtn: document.getElementById('send-btn'),
            attachBtn: document.getElementById('attach-btn'),
            currentSessionIdSpan: document.getElementById('current-session-id'),
            turnCountSpan: document.getElementById('turn-count'),
            contextPercentSpan: document.getElementById('context-percent'),
            contextIndicator: document.getElementById('context-indicator'),
            messagesContainer: document.getElementById('messages-container'),
            messageInput: document.getElementById('message-input'),
            loadingOverlay: document.getElementById('loading-overlay'),
            loadingMessage: document.getElementById('loading-message'),
            toastContainer: document.getElementById('toast-container')
        };
    }
    
    setupEventListeners() {
        this.elements.languageSelect.addEventListener('change', (e) => {
            this.currentLanguage = e.target.value;
        });

        this.elements.tierSelect.addEventListener('change', (e) => {
            this.currentTier = e.target.value;
            const isLlmCloud = this.currentTier === 'llm_cloud';
            this.elements.llmModelGroup.style.display = isLlmCloud ? 'flex' : 'none';
        });

        this.elements.llmModelSelect.addEventListener('change', (e) => {
            this.currentLlmModel = e.target.value;
        });
        
        this.elements.modeSelect.addEventListener('change', (e) => {
            this.currentMode = e.target.value;
            if (this.currentMode === 'clinical-reasoning') {
                this.elements.examSelectionGroup.style.display = 'flex';
            } else {
                this.elements.examSelectionGroup.style.display = 'none';
            }
        });
        
        // Listener para os checkboxes de exames
        document.querySelectorAll('input[name="exam-type"]').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.selectedExams = Array.from(document.querySelectorAll('input[name="exam-type"]:checked'))
                    .map(cb => cb.value);
            });
        });
        
        this.elements.startSessionBtn.addEventListener('click', () => {
            this.createSession();
        });
        
        this.elements.newSessionBtn.addEventListener('click', () => {
            if (confirm('Start a new session? Current session will be saved.')) {
                this.showWelcomeScreen();
            }
        });
        
        this.elements.exportSessionBtn.addEventListener('click', () => {
            this.exportSession();
        });
        
        this.elements.sendBtn.addEventListener('click', () => {
            this.sendMessage();
        });
        
        this.elements.attachBtn.addEventListener('click', () => {
            this.uploadManager.toggle();
        });
        
        this.elements.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.elements.messageInput.addEventListener('input', () => {
            this.autoResizeTextarea(this.elements.messageInput);
        });
    }
    
    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }
    
    showWelcomeScreen() {
        this.elements.welcomeScreen.style.display = 'flex';
        this.elements.chatInterface.style.display = 'none';
        this.currentSessionId = null;
    }
    
    showChatInterface() {
        this.elements.welcomeScreen.style.display = 'none';
        this.elements.chatInterface.style.display = 'flex';
    }
    
    async createSession() {
        this.showLoading(this.getMessage('processingMessage'));
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/session/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mode: this.currentMode,
                    language: this.currentLanguage,
                    model_tier: this.currentTier,
                    llm_model: this.currentLlmModel,
                    exam_types: this.currentMode === 'clinical-reasoning' ? this.selectedExams : []
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            this.currentSessionId = data.session_id;
            this.currentLanguage = data.language;
            this.updateSessionInfo(data);
            this.showChatInterface();
            this.clearMessages();
            this.showToast(this.getMessage('sessionCreated'), 'success');
            
        } catch (error) {
            console.error('Error creating session:', error);
            this.showToast(this.getMessage('connectionError'), 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async sendMessage() {
        const message = this.elements.messageInput.value.trim();
        
        if (!message && !this.uploadManager.hasFiles()) {
            return;
        }
        
        if (!this.currentSessionId) {
            this.showToast('No active session', 'error');
            return;
        }
        
        this.elements.sendBtn.disabled = true;
        this.elements.messageInput.disabled = true;
        
        const filenames = this.uploadManager.hasFiles() 
            ? this.uploadManager.getFiles().map(f => f.name) 
            : [];

        if (message || filenames.length > 0) {
            this.addMessage('user', message, filenames);
        }
        
        this.elements.messageInput.value = '';
        this.autoResizeTextarea(this.elements.messageInput);
        
        const loadingMsg = this.uploadManager.hasFiles() 
            ? this.getMessage('processingFiles')
            : this.getMessage('processingMessage');
        this.showLoading(loadingMsg);
        
        try {
            let response;
            
            if (this.uploadManager.hasFiles()) {
                const formData = new FormData();
                formData.append('session_id', this.currentSessionId);
                formData.append('message', message);
                formData.append('language', this.currentLanguage);
                
                // Enviar exames selecionados no momento do upload
                formData.append('exam_types', JSON.stringify(this.selectedExams));
                
                this.uploadManager.getFiles().forEach(file => {
                    formData.append('files', file);
                });
                
                response = await fetch(`${this.apiBaseUrl}/chat/multimodal`, {
                    method: 'POST',
                    body: formData
                });
                
                this.uploadManager.clearFiles();
                
            } else {
                response = await fetch(`${this.apiBaseUrl}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: this.currentSessionId,
                        message: message,
                        language: this.currentLanguage
                    })
                });
            }
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            this.addMessage('assistant', data.response);
            this.updateSessionInfo({ turn_count: data.turn_count });
            
            if (data.context_usage) {
                this.updateContextUsage(data.context_usage);
            }
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.showToast(this.getMessage('errorOccurred'), 'error');
        } finally {
            this.hideLoading();
            this.elements.sendBtn.disabled = false;
            this.elements.messageInput.disabled = false;
            this.elements.messageInput.focus();
        }
    }
    
    addMessage(role, content, filenames = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = role === 'user' ? '👤' : '🤖';
        const renderedContent = window.MarkdownRenderer.render(content);
        
        let filesHtml = '';
        if (filenames && filenames.length > 0) {
            filesHtml = `
                <div class="message-files">
                    ${filenames.map(f => `<span class="file-tag">📎 ${this.maskFilename(f)}</span>`).join('')}
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-bubble">
                ${filesHtml}
                <div class="message-content">${renderedContent}</div>
            </div>
        `;
        
        this.elements.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    clearMessages() {
        this.elements.messagesContainer.innerHTML = '';
    }
    
    scrollToBottom() {
        this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
    }
    
    updateSessionInfo(data) {
        if (data.session_id) {
            this.elements.currentSessionIdSpan.textContent = data.session_id.substring(0, 8);
        }
        if (data.turn_count !== undefined) {
            this.elements.turnCountSpan.textContent = data.turn_count;
        }
    }
    
    updateContextUsage(contextUsage) {
        const percent = contextUsage.usage_percent || 0;
        this.elements.contextPercentSpan.textContent = percent.toFixed(0);
        
        this.elements.contextIndicator.className = 'context-indicator';
        if (percent >= 80) {
            this.elements.contextIndicator.classList.add('high');
        } else if (percent >= 50) {
            this.elements.contextIndicator.classList.add('moderate');
        }
    }
    
    async exportSession() {
        if (!this.currentSessionId) {
            this.showToast('No active session', 'error');
            return;
        }
        
        this.showLoading('Exporting session...');
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/session/${this.currentSessionId}/export`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `medteacher_session_${this.currentSessionId}.md`;
            a.click();
            URL.revokeObjectURL(url);
            
            this.showToast(this.getMessage('sessionExported'), 'success');
            
        } catch (error) {
            console.error('Error exporting session:', error);
            this.showToast(this.getMessage('errorOccurred'), 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    showLoading(message) {
        this.elements.loadingMessage.textContent = message;
        this.elements.loadingOverlay.style.display = 'flex';
    }
    
    hideLoading() {
        this.elements.loadingOverlay.style.display = 'none';
    }
    
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        this.elements.toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, 3000);
    }
    
    getMessage(key) {
        return this.messages[this.currentLanguage]?.[key] || this.messages.en[key] || key;
    }

    maskFilename(filename) {
        if (!filename) return '';
        const parts = filename.split('.');
        const ext = parts.pop();
        const name = parts.join('.');
        
        if (name.length <= 6) {
            return `***.${ext}`;
        }
        
        const start = name.substring(0, 3);
        const end = name.substring(name.length - 3);
        return `${start}...${end}.${ext}`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.AppManager = new AppManager();
    window.AppManager.init();
});
