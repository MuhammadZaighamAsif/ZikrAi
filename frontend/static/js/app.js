// API Base URL
const API_BASE = window.location.origin;

// DOM Elements
const chatContainer = document.getElementById('chatContainer');
const queryForm = document.getElementById('queryForm');
const queryInput = document.getElementById('queryInput');
const submitBtn = document.getElementById('submitBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const themeToggle = document.getElementById('themeToggle');
const fontInc = document.getElementById('fontInc');
const fontDec = document.getElementById('fontDec');

// Stats elements
const totalChunks = document.getElementById('totalChunks');
const quranChunks = document.getElementById('quranChunks');
const hadithChunks = document.getElementById('hadithChunks');
const statusIndicator = document.getElementById('status');
const toastContainer = document.getElementById('toastContainer');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    setupEventListeners();
    updateWelcomeTime();
    updateCharCounter();
    initTheme();
    
    // Allow Shift+Enter for new line, Enter to submit
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            queryForm.dispatchEvent(new Event('submit'));
        }
    });
    
    // Character counter
    queryInput.addEventListener('input', () => {
        updateCharCounter();
        autoResize();
    });
});

// Update character counter
function updateCharCounter() {
    const charCounter = document.getElementById('charCounter');
    if (charCounter) {
        const length = queryInput.value.length;
        charCounter.textContent = `${length}/500`;
        charCounter.style.color = length > 450 ? '#EF4444' : '#6B7280';
    }
}

// Auto-resize textarea
function autoResize() {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
}

// Update welcome time
function updateWelcomeTime() {
    const welcomeTime = document.getElementById('welcomeTime');
    if (welcomeTime) {
        const now = new Date();
        welcomeTime.textContent = now.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }
}

// Load system stats
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();
        
        totalChunks.textContent = data.total_chunks.toLocaleString();
        quranChunks.textContent = data.quran_chunks.toLocaleString();
        hadithChunks.textContent = data.hadith_chunks.toLocaleString();
        
        if (data.model_available) {
            statusIndicator.style.color = '#10B981';
            statusIndicator.title = 'System Ready';
        } else {
            statusIndicator.style.color = '#F59E0B';
            statusIndicator.title = 'Limited Functionality';
        }
    } catch (error) {
        console.error('Error loading stats:', error);
        statusIndicator.style.color = '#EF4444';
        statusIndicator.title = 'System Error';
        totalChunks.textContent = '--';
        quranChunks.textContent = '--';
        hadithChunks.textContent = '--';
    }
}

// Setup event listeners
function setupEventListeners() {
    queryForm.addEventListener('submit', handleSubmit);
    
    // Quick action buttons
    document.querySelectorAll('.quick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.dataset.question;
            queryInput.value = question;
            queryInput.focus();
            updateCharCounter();
            autoResize();
            showToast('Loaded quick question', 'success');
        });
    });
    
    // Clear button
    const clearBtn = document.getElementById('clearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (confirm('Clear the conversation?')) {
                chatContainer.innerHTML = '';
                addWelcomeMessage();
                showToast('Conversation cleared', 'success');
            }
        });
    }

    // Theme toggle
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const isDark = document.body.classList.toggle('dark');
            localStorage.setItem('zikrai_theme', isDark ? 'dark' : 'light');
            themeToggle.textContent = isDark ? '☀️ Light' : '🌙 Dark';
        });
    }

    // Font size controls
    const applyFontSize = (size) => {
        document.documentElement.style.setProperty('--message-font-size', `${size}px`);
        localStorage.setItem('zikrai_font_size', String(size));
    };
    const currentSize = parseInt(localStorage.getItem('zikrai_font_size') || '16', 10);
    applyFontSize(currentSize);
    if (fontInc) {
        fontInc.addEventListener('click', () => {
            const size = Math.min((parseInt(localStorage.getItem('zikrai_font_size') || '16', 10) + 1), 22);
            applyFontSize(size);
        });
    }
    if (fontDec) {
        fontDec.addEventListener('click', () => {
            const size = Math.max((parseInt(localStorage.getItem('zikrai_font_size') || '16', 10) - 1), 14);
            applyFontSize(size);
        });
    }
}

// Add welcome message
function addWelcomeMessage() {
    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = 'message bot-message';
    welcomeDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-header">
                <span class="message-sender">ZikrAi Assistant</span>
                <span class="message-time">${new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</span>
            </div>
            <p class="intro"><strong>السلام عليكم ورحمة الله وبركاته</strong></p>
            <p class="intro">Peace be upon you! Welcome to ZikrAi.</p>
            <p class="intro">I'm your Islamic Knowledge Assistant, here to help you explore the wisdom of the Quran and Hadith. Simply type your question below or click one of the quick topic buttons above!</p>
        </div>
    `;
    chatContainer.appendChild(welcomeDiv);
}

// Initialize theme from localStorage or system preference
function initTheme() {
    const saved = localStorage.getItem('zikrai_theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const useDark = saved ? saved === 'dark' : prefersDark;
    document.body.classList.toggle('dark', useDark);
    if (themeToggle) {
        themeToggle.textContent = useDark ? '☀️ Light' : '🌙 Dark';
    }
}

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();
    
    const query = queryInput.value.trim();
    if (!query) return;
    
    // Add user message
    addMessage(query, 'user');
    
    // Clear input
    queryInput.value = '';
    queryInput.style.height = 'auto';
    updateCharCounter();
    
    // Disable input
    setLoading(true);
    
    try {
        // Call API
        const response = await fetch(`${API_BASE}/api/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query, k: 5 })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Add bot response with sources
            addMessage(data.answer, 'bot', data.sources);
            showToast('Answer ready', 'success');
        } else {
            addMessage(`Error: ${data.error}`, 'bot');
            showToast('Error from server', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, there was an error processing your request. Please try again.', 'bot');
        showToast('Network error', 'error');
    } finally {
        setLoading(false);
    }
}

// Add message to chat
function addMessage(text, type, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    // Add avatar
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = type === 'user' ? '👤' : '🤖';
    messageDiv.appendChild(avatar);
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Add header with sender and time
    const header = document.createElement('div');
    header.className = 'message-header';
    const sender = document.createElement('span');
    sender.className = 'message-sender';
    sender.textContent = type === 'user' ? 'You' : 'ZikrAi Assistant';
    const time = document.createElement('span');
    time.className = 'message-time';
    time.textContent = new Date().toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    header.appendChild(sender);
    // Actions container (time + copy)
    const actions = document.createElement('div');
    actions.style.display = 'flex';
    actions.style.alignItems = 'center';
    actions.style.gap = '6px';
    actions.appendChild(time);
    if (type === 'bot') {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.title = 'Copy response';
        copyBtn.textContent = 'Copy';
        copyBtn.addEventListener('click', async () => {
            try {
                const plainText = paragraphs.join('\n');
                await navigator.clipboard.writeText(plainText);
                copyBtn.textContent = 'Copied!';
                showToast('Response copied', 'success');
                setTimeout(() => (copyBtn.textContent = 'Copy'), 1500);
            } catch (err) {
                console.error('Copy failed:', err);
                copyBtn.textContent = 'Failed';
                showToast('Copy failed', 'error');
                setTimeout(() => (copyBtn.textContent = 'Copy'), 1500);
            }
        });
        actions.appendChild(copyBtn);
    }
    header.appendChild(actions);
    contentDiv.appendChild(header);
    
    // Format text: turn newlines into paragraphs, simple emphasis
    const paragraphs = text.split('\n').filter(p => p.trim());
    paragraphs.forEach(p => {
        const pElement = document.createElement('p');
        pElement.textContent = p.trim();
        contentDiv.appendChild(pElement);
    });
    
    // Add sources if available
    if (sources && sources.length > 0) {
        const sourcesDiv = createSourcesDiv(sources);
        contentDiv.appendChild(sourcesDiv);
    }
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom smoothly
    setTimeout(() => {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    }, 100);
}

// Create sources display
function createSourcesDiv(sources) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'sources';
    
    const header = document.createElement('div');
    header.className = 'sources-header';
    header.innerHTML = '📚 Sources Used:';
    sourcesDiv.appendChild(header);
    
    sources.forEach((source, index) => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';

        const reference = document.createElement('div');
        reference.className = 'source-reference';
        reference.textContent = `${index + 1}. ${source.source}`;

        const text = document.createElement('div');
        text.className = 'source-text';
        const fullText = source.text || '';
        const preview = fullText.substring(0, 180) + (fullText.length > 180 ? '…' : '');
        text.textContent = preview;

        // Expand/collapse control
        if (fullText.length > 180) {
            const toggle = document.createElement('button');
            toggle.textContent = 'Show more';
            toggle.style.marginLeft = '8px';
            toggle.style.fontSize = '0.85rem';
            toggle.style.cursor = 'pointer';
            let expanded = false;
            toggle.addEventListener('click', () => {
                expanded = !expanded;
                text.textContent = expanded ? fullText : preview;
                toggle.textContent = expanded ? 'Show less' : 'Show more';
            });
            text.appendChild(toggle);
        }

        sourceItem.appendChild(reference);
        sourceItem.appendChild(text);
        sourcesDiv.appendChild(sourceItem);
    });
    
    return sourcesDiv;
}

// Set loading state
function setLoading(isLoading) {
    if (isLoading) {
        // add skeleton message for perceived loading
        const skeleton = document.createElement('div');
        skeleton.className = 'skeleton-message';
        skeleton.id = 'skeletonMessage';
        skeleton.innerHTML = `
            <div class="skeleton-avatar"></div>
            <div class="skeleton-content">
                <div class="skeleton-line short"></div>
                <div class="skeleton-line long"></div>
                <div class="skeleton-line medium"></div>
            </div>
        `;
        chatContainer.appendChild(skeleton);
        loadingOverlay.classList.add('active');
        submitBtn.disabled = true;
        queryInput.disabled = true;
    } else {
        loadingOverlay.classList.remove('active');
        submitBtn.disabled = false;
        queryInput.disabled = false;
        queryInput.focus();
        // remove skeleton
        const sk = document.getElementById('skeletonMessage');
        if (sk) sk.remove();
    }
}

// Toast helper
function showToast(message, type = 'success') {
    if (!toastContainer) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('fade');
        toast.remove();
    }, 2000);
}
