from flask import Flask, request, render_template, jsonify, session
from flask_session import Session
import os
from werkzeug.utils import secure_filename
from base_rag2 import rag_process
import base64
import io


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Add a secret key for session encryption

# Configure session management
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './session_data'
app.config['SESSION_COOKIE_NAME'] = 'my_session_cookie'
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
Session(app)

REFERENCE_BOOKS = ['deep learning with python', 'python data science handbook']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/select_book', methods=['POST'])
def select_book():
    try:
        book_name = request.json.get('book_name')
        if not book_name or book_name not in REFERENCE_BOOKS:
            return jsonify({'error': 'Invalid book selection'}), 400

        # Store only the book name in the session (not the entire rag_model)
        session['current_book'] = book_name
        print(f"Book '{book_name}' selected successfully")
        return jsonify({'message': f'{book_name} selected successfully'})

    except Exception as e:
        print(f"Error in select_book: {str(e)}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/status', methods=['GET'])
def status():
    # Check if a book is selected (no need to store rag_model in session)
    if 'current_book' in session:
        return jsonify({'status': 'ready'})
    return jsonify({'status': 'processing'})

"""@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        if 'current_book' not in session:
            return jsonify({'error': 'No book selected'}), 400

        # Reinitialize the RAG model using the stored book name
        book_name = session['current_book']
        rag_model = rag_process(book_name)  # Recreate the model here
        answer = rag_model.execute(question)
        return jsonify({'answer': answer})

    except Exception as e:
        print(f"Error in ask: {str(e)}")
        return jsonify({'error': 'Server error'}), 500"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import concurrent.futures

@app.route('/ask', methods=['POST'])
def ask():
    import traceback  # Add this import for detailed error tracing
    
    try:
        question = request.json.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        if 'current_book' not in session:
            return jsonify({'error': 'No book selected'}), 400

        book_name = session['current_book']
        
        try:
            print(f"Attempting to process RAG for book: {book_name}")
            print(f"Question: {question}")
            
            # Detailed logging before model processing
            rag_model = rag_process(book_name)
            print("RAG model successfully loaded")
            
            # Add more verbose logging during execution
            answer = rag_model.execute(question)
            print("Answer successfully generated")
            
            return jsonify({'answer': answer})
        
        except Exception as model_error:
            # Comprehensive error logging
            print("RAG Model Processing Error:")
            print(f"Error Type: {type(model_error)}")
            print(f"Error Details: {str(model_error)}")
            traceback.print_exc()  # Print full stack trace
            
            return jsonify({
                'error': f'Error processing question: {str(model_error)}',
                'error_type': str(type(model_error))
            }), 500

    except Exception as e:
        print("Unexpected Error in ask route:")
        traceback.print_exc()
        return jsonify({'error': f'Unexpected server error: {str(e)}'}), 500

def process_rag_question(book_name, question):
    """
    Separate function to process the RAG question.
    This allows us to easily apply a timeout.
    """
    rag_model = rag_process(book_name)
    answer = rag_model.execute(question)
    return answer

@app.route('/end_session', methods=['POST'])
def end_session():
    session.pop('current_book', None)
    return jsonify({'message': 'Session ended successfully'})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
