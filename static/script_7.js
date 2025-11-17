// Add these at the start of script.js
let bookSelected = false;

// Function to handle book selection
function selectBook() {
    const bookSelect = document.getElementById('book-select');
    const bookName = bookSelect.value;

    if (!bookName) {
        alert('Please select a book first');
        return;
    }

    document.getElementById('processing-message').style.display = 'block';
    document.getElementById('book-selection').style.display = 'none';

    fetch('/select_book', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ book_name: bookName }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            document.getElementById('processing-message').style.display = 'none';
            document.getElementById('book-selection').style.display = 'block';
        } else {
            bookSelected = true;
            checkStatus();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('processing-message').style.display = 'none';
        document.getElementById('book-selection').style.display = 'block';
        alert('Error selecting book. Please try again.');
    });
}

// Function to check the status of the RAG model
function checkStatus() {
    fetch('/status', {
        method: 'GET',
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'ready') {
            document.getElementById('processing-message').style.display = 'none';
            document.getElementById('chat-interface').style.display = 'block';
        } else {
            setTimeout(checkStatus, 1000); // Check again after 1 second
        }
    });
}
/*
// Function to handle user questions
function askQuestion() {
    const question = document.getElementById('question-input').value;
    if (!question) {
        alert('Please enter a question.');
        return;
    }

    addMessage(question, 'user-message');
    document.getElementById('question-input').value = '';

    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            addMessage(data.answer, 'bot-message', true);
        }
    })
    .catch(error => console.error('Error:', error));
}
*/

function askQuestion() {
    const questionInput = document.getElementById('question-input');
    const question = questionInput.value.trim();

    if (!question) {
        alert('Please enter a question.');
        return;
    }

    addMessage(question, 'user-message');
    questionInput.value = '';

    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        timeout: 35000, // 35 seconds timeout
        body: JSON.stringify({ question: question }),
    })
    .then(response => {
        if (!response.ok) {
            // Handle HTTP errors
            return response.json().then(errorData => {
                throw new Error(errorData.error || 'Unknown server error');
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            addMessage(`Error: ${data.error}`, 'bot-message');
        } else {
            addMessage(data.answer, 'bot-message', true);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        addMessage(`Sorry, there was an error processing your question: ${error.message}`, 'bot-message');
    });
}
// Function to add a message to the chat interface
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

// Add event listener for Enter key in question input
document.getElementById('question-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        askQuestion();
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



// Function to handle book selection
function selectBook() {
    const bookSelect = document.getElementById('book-select');
    const bookName = bookSelect.value;

    fetch('/select_book', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ book_name: bookName }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            document.getElementById('book-selection').style.display = 'none';
            document.getElementById('processing-message').style.display = 'block';
            checkStatus();
        }
    });
}

// Function to check the status of the RAG model
function checkStatus() {
    fetch('/status', {
        method: 'GET',
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'ready') {
            document.getElementById('processing-message').style.display = 'none';
            document.getElementById('chat-interface').style.display = 'block';
        } else {
            setTimeout(checkStatus, 1000); // Check again after 1 second
        }
    });
}

// Function to handle user questions
function askQuestion() {
    const questionInput = document.getElementById('question-input');
    const question = questionInput.value;

    if (!question) {
        alert('Please enter a question.');
        return;
    }

    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            const chatMessages = document.getElementById('chat-messages');
            const messageElement = document.createElement('div');
            messageElement.innerHTML = `<strong>You:</strong> ${question}<br><strong>Bot:</strong> ${data.answer}`;
            chatMessages.appendChild(messageElement);
            questionInput.value = ''; // Clear the input field
        }
    });
}

// Function to end the session
function endSession() {
    fetch('/end_session', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        location.reload(); // Reload the page to reset the UI
    });
}