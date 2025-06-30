from flask import Flask, render_template, request, jsonify
from Model import InsuranceLLM, ModelConfig
from Database import init_nonpersistent_cosine_db
import Utility as util
from sentence_transformers import SentenceTransformer

config = ModelConfig()
llm = InsuranceLLM(config)
llm.load_model()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
db = init_nonpersistent_cosine_db()

app = Flask(__name__)

# Mock RAG implementation
def rag_pipeline(user_query):
    q_embed = embedder.encode(user_query)
    context,metadata = util.get_context(q_embed,db)
    response,metastring = llm.generate_answer(user_query,context,metadata)
    return f"{response}\n{metastring}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get('user_input')
    if not user_input:
        return jsonify({'response': 'No input provided.'})
    
    response = rag_pipeline(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
