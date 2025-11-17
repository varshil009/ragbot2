// Add these at the start of script.js
let pdfViewerReady = false;
let pdfViewer = null;

// Stop words list for highlighting
const STOP_WORDS = new Set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
    'where', 'who', 'which', 'why', 'how'
]);

window.addEventListener('message', function(event) {
    // Store the PDF viewer reference when it's ready
    if (event.data && event.data.type === 'READY') {
        pdfViewer = document.getElementById('pdf-viewer').contentWindow;
        pdfViewerReady = true;
    }
});

function uploadPDF() {
    const fileInput = document.getElementById('pdf-upload');
    if (!fileInput.files || !fileInput.files[0]) {
        alert('Please select a PDF file first');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    document.getElementById('processing-message').style.display = 'block';
    document.getElementById('upload-section').style.display = 'none';

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.message === 'PDF uploaded and processed successfully') {
            document.getElementById('processing-message').style.display = 'none';
            document.getElementById('chat-interface').style.display = 'block';
            
            const pdfUrl = URL.createObjectURL(fileInput.files[0]);
            const viewer = document.getElementById('pdf-viewer');
            
            // Set up PDF viewer with event listener
            viewer.src = `/static/pdfjs/web/viewer.html?file=${encodeURIComponent(pdfUrl)}`;
            
            // Add load event listener to the iframe
            viewer.onload = function() {
                // Wait for PDF.js to fully initialize
                const checkPDFApplication = setInterval(() => {
                    if (viewer.contentWindow.PDFViewerApplication) {
                        clearInterval(checkPDFApplication);
                        pdfViewer = viewer.contentWindow;
                        pdfViewerReady = true;
                    }
                }, 100);
            };
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('processing-message').style.display = 'none';
        document.getElementById('upload-section').style.display = 'block';
        alert('Error uploading PDF. Please try again.');
    });
}

function askQuestion() {
    const question = document.getElementById('question-input').value;
    if (!question) return;

    addMessage(question, 'user-message');
    document.getElementById('question-input').value = '';

    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        if (data.answer) {
            addMessage(data.answer, 'bot-message', true);
        }
        if (data.image) {
            addImage(data.image);
        }
        if (data.page_numbers && data.page_numbers.length > 0) {
            addPageNavigation(data.page_numbers, data.answer);
        }
    })
    .catch(error => console.error('Error:', error));
}

function addMessage(text, className, isMarkdown = false) {
    if (!text) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;

    if (isMarkdown && typeof marked !== 'undefined') {
        messageDiv.innerHTML = marked.parse(text);
    } else {
        messageDiv.textContent = text;
    }

    document.getElementById('chat-messages').appendChild(messageDiv);
    messageDiv.scrollIntoView({ behavior: "smooth" });
}

function addImage(imageBase64) {
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${imageBase64}`;
    img.style.maxWidth = '100%';
    document.getElementById('chat-messages').appendChild(img);
    img.scrollIntoView({ behavior: "smooth" });
}

function addPageNavigation(pageNumbers, answerText) {
    const navContainer = document.createElement('div');
    navContainer.className = 'page-navigation';

    pageNumbers.forEach(page => {
        const button = document.createElement('button');
        button.textContent = `Page ${page}`;
        button.onclick = () => {
            if (pdfViewerReady) {
                jumpToPageAndHighlight(page, answerText);
            } else {
                console.log('PDF viewer not ready yet');
            }
        };
        navContainer.appendChild(button);
    });

    document.getElementById('chat-messages').appendChild(navContainer);
    navContainer.scrollIntoView({ behavior: "smooth" });
}

function jumpToPageAndHighlight(pageNumber, query) {
    const viewer = document.getElementById('pdf-viewer');
    if (!viewer || !pdfViewerReady) return;

    pageNumber = parseInt(pageNumber);

    try {
        // Jump to the specified page
        viewer.contentWindow.PDFViewerApplication.page = pageNumber;

        // Wait for the text layer to be rendered
        const checkTextLayer = setInterval(() => {
            const pageElement = viewer.contentWindow.document.querySelector(`#viewer .page[data-page-number="${pageNumber}"]`);
            if (pageElement) {
                const textLayer = pageElement.querySelector('.textLayer');
                if (textLayer) {
                    clearInterval(checkTextLayer);
                    highlightTextOnPage(pageNumber, query);
                }
            }
        }, 100); // Check every 100ms
    } catch (error) {
        console.error('Error jumping to page:', error);
    }
}
function highlightTextOnPage(pageNumber, query) {
    const viewer = document.getElementById('pdf-viewer');
    if (!viewer || !query) return;

    try {
        const pageElement = viewer.contentWindow.document.querySelector(`#viewer .page[data-page-number="${pageNumber}"]`);
        if (!pageElement) {
            console.error(`Page ${pageNumber} not found`);
            return;
        }

        // Remove previous highlights
        const previousHighlights = pageElement.querySelectorAll('.highlight-text');
        previousHighlights.forEach(highlight => highlight.remove());

        // Process the query to get meaningful words
        const wordsToHighlight = query
            .toLowerCase()
            .split(/\s+/)
            .filter(word => {
                // Remove punctuation and check if it's not a stop word
                const cleanWord = word.replace(/[.,!?;:()"']/g, '');
                return cleanWord.length > 2 && !STOP_WORDS.has(cleanWord);
            });

        const textLayer = pageElement.querySelector('.textLayer');
        if (!textLayer) {
            console.error('Text layer not found');
            return;
        }

        const textElements = textLayer.querySelectorAll('span');
        textElements.forEach(span => {
            const spanText = span.textContent.toLowerCase();

            // Check if any of the query words appear in this span
            const shouldHighlight = wordsToHighlight.some(word => 
                spanText.includes(word)
            );

            if (shouldHighlight) {
                const highlightDiv = document.createElement('div');
                highlightDiv.className = 'highlight-text';

                const rect = span.getBoundingClientRect();
                const textLayerRect = textLayer.getBoundingClientRect();

                // Position the highlight relative to the text layer
                highlightDiv.style.left = `${rect.left - textLayerRect.left}px`;
                highlightDiv.style.top = `${rect.top - textLayerRect.top}px`;
                highlightDiv.style.width = `${rect.width}px`;
                highlightDiv.style.height = `${rect.height}px`;

                textLayer.appendChild(highlightDiv);
            }
        });
    } catch (error) {
        console.error('Error highlighting text:', error);
    }
}// Add event listener for Enter key in question input
document.getElementById('question-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        askQuestion();
    }
});

// Add this event listener
window.addEventListener('message', function(event) {
    if (event.data && event.data.type === 'pdfViewer-ready') {
        pdfViewerReady = true;
        pdfViewer = document.getElementById('pdf-viewer').contentWindow;
    }
});

// Add CSS for highlights
const style = document.createElement('style');
style.textContent = `
.highlight-text {
    position: absolute;
    background-color: yellow;
    opacity: 0.5;
    pointer-events: none;
    mix-blend-mode: multiply;
}`;
document.head.appendChild(style);