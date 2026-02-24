/**
 * Simple Markdown Renderer
 * Converte Markdown básico para HTML de forma segura
 * Sem dependências externas
 */

const MarkdownRenderer = {
    /**
     * Renderiza Markdown para HTML
     * @param {string} markdown - Texto em Markdown
     * @returns {string} HTML renderizado
     */
    render(markdown) {
        if (!markdown) return '';
        
        let html = markdown;
        
        // Escape HTML para segurança
        html = this.escapeHtml(html);
        
        // Processar blocos de código (antes de outras conversões)
        html = this.processCodeBlocks(html);
        
        // Processar código inline
        html = this.processInlineCode(html);
        
        // Headers (# ## ### etc)
        html = this.processHeaders(html);
        
        // Bold (**texto**)
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        
        // Italic (*texto*)
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        
        // Links [texto](url)
        html = this.processLinks(html);
        
        // Listas
        html = this.processLists(html);
        
        // Quebras de linha (dois espaços + newline ou duplo newline)
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/  \n/g, '<br>');
        
        // Envolver em parágrafos
        html = `<p>${html}</p>`;
        
        // Limpar parágrafos vazios
        html = html.replace(/<p><\/p>/g, '');
        
        return html;
    },
    
    /**
     * Escape HTML para prevenir XSS
     */
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        
        // Não escapar dentro de blocos de código
        return text.replace(/[&<>"']/g, m => map[m]);
    },
    
    /**
     * Processa blocos de código ```
     */
    processCodeBlocks(text) {
        // Substituir blocos de código por placeholders
        const codeBlocks = [];
        
        text = text.replace(/```(\w+)?\n([\s\S]+?)```/g, (match, lang, code) => {
            const placeholder = `___CODEBLOCK_${codeBlocks.length}___`;
            codeBlocks.push({
                lang: lang || 'plaintext',
                code: code.trim()
            });
            return placeholder;
        });
        
        // Restaurar blocos de código como HTML
        codeBlocks.forEach((block, index) => {
            const html = `<pre><code class="language-${block.lang}">${block.code}</code></pre>`;
            text = text.replace(`___CODEBLOCK_${index}___`, html);
        });
        
        return text;
    },
    
    /**
     * Processa código inline `code`
     */
    processInlineCode(text) {
        return text.replace(/`(.+?)`/g, '<code>$1</code>');
    },
    
    /**
     * Processa headers # ## ### etc
     */
    processHeaders(text) {
        const lines = text.split('\n');
        
        return lines.map(line => {
            // H1
            if (line.startsWith('# ')) {
                return `<h1>${line.substring(2)}</h1>`;
            }
            // H2
            if (line.startsWith('## ')) {
                return `<h2>${line.substring(3)}</h2>`;
            }
            // H3
            if (line.startsWith('### ')) {
                return `<h3>${line.substring(4)}</h3>`;
            }
            // H4
            if (line.startsWith('#### ')) {
                return `<h4>${line.substring(5)}</h4>`;
            }
            
            return line;
        }).join('\n');
    },
    
    /**
     * Processa links [texto](url)
     */
    processLinks(text) {
        return text.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    },
    
    /**
     * Processa listas - e *
     */
    processLists(text) {
        const lines = text.split('\n');
        let inList = false;
        let listType = null;
        const result = [];
        
        lines.forEach(line => {
            // Lista não ordenada
            if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
                if (!inList) {
                    result.push('<ul>');
                    inList = true;
                    listType = 'ul';
                }
                const content = line.trim().substring(2);
                result.push(`<li>${content}</li>`);
            }
            // Lista ordenada
            else if (/^\d+\.\s/.test(line.trim())) {
                if (!inList) {
                    result.push('<ol>');
                    inList = true;
                    listType = 'ol';
                } else if (listType === 'ul') {
                    result.push('</ul>');
                    result.push('<ol>');
                    listType = 'ol';
                }
                const content = line.trim().replace(/^\d+\.\s/, '');
                result.push(`<li>${content}</li>`);
            }
            // Não é item de lista
            else {
                if (inList) {
                    result.push(listType === 'ul' ? '</ul>' : '</ol>');
                    inList = false;
                    listType = null;
                }
                result.push(line);
            }
        });
        
        // Fechar lista se ainda aberta
        if (inList) {
            result.push(listType === 'ul' ? '</ul>' : '</ol>');
        }
        
        return result.join('\n');
    }
};

// Disponibilizar globalmente
window.MarkdownRenderer = MarkdownRenderer;
