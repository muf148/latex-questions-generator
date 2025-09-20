#!/usr/bin/env python3
"""
Question Generator from Book/Lecture Notes
==========================================

This script processes text content (books, lecture notes, PDFs) and generates
questions on specified topics with LaTeX-formatted output.

Requirements:
- nltk (for text processing)
- PyPDF2 or pdfplumber (for PDF processing)
- transformers (optional, for advanced NLP)

Install dependencies:
pip install nltk PyPDF2 pdfplumber
"""

import re
import random
import argparse
from typing import List, Dict, Tuple
from pathlib import Path
import io

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk_available = True
except ImportError:
    nltk_available = False

# PDF processing imports
pdf_available = False
try:
    import PyPDF2
    pdf_library = 'PyPDF2'
    pdf_available = True
except ImportError:
    try:
        import pdfplumber
        pdf_library = 'pdfplumber'
        pdf_available = True
    except ImportError:
        pdf_library = None

class QuestionGenerator:
    def __init__(self):
        self.question_templates = {
            'definition': [
                "What is {topic}?",
                "Define {topic}.",
                "Explain the concept of {topic}.",
                "How would you describe {topic}?"
            ],
            'explanation': [
                "Explain how {topic} works.",
                "Describe the process of {topic}.",
                "What are the key principles behind {topic}?",
                "How does {topic} function?"
            ],
            'comparison': [
                "Compare and contrast {topic1} and {topic2}.",
                "What are the similarities and differences between {topic1} and {topic2}?",
                "How does {topic1} differ from {topic2}?"
            ],
            'application': [
                "Provide examples of {topic} in real-world scenarios.",
                "How can {topic} be applied in practice?",
                "What are the practical applications of {topic}?",
                "Give an example where {topic} would be useful."
            ],
            'analysis': [
                "Analyze the importance of {topic}.",
                "What are the advantages and disadvantages of {topic}?",
                "Critically evaluate {topic}.",
                "Discuss the implications of {topic}."
            ],
            'cause_effect': [
                "What causes {topic}?",
                "What are the effects of {topic}?",
                "Explain the relationship between cause and effect in {topic}.",
                "What happens when {topic} occurs?"
            ]
        }
        
        if nltk_available:
            try:
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
            except LookupError:
                print("NLTK data not found. Run: python -m nltk.downloader punkt stopwords wordnet")
                self.stop_words = set()
                self.lemmatizer = None

    def load_text(self, file_path: str) -> str:
        """Load text content from a file (supports .txt, .md, .pdf)."""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                return self._load_pdf(file_path)
            else:
                # Handle text files
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading file: {e}")
    
    def _load_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        if not pdf_available:
            raise ImportError("PDF processing not available. Install PyPDF2 or pdfplumber: pip install PyPDF2 pdfplumber")
        
        text = ""
        
        if pdf_library == 'PyPDF2':
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                raise Exception(f"Error reading PDF with PyPDF2: {e}")
                
        elif pdf_library == 'pdfplumber':
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                raise Exception(f"Error reading PDF with pdfplumber: {e}")
        
        if not text.strip():
            raise Exception("No text could be extracted from the PDF file")
            
        return text
    
    def load_text_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        """Load text content from bytes (for Streamlit file uploads)."""
        file_extension = Path(filename).suffix.lower()
        
        if file_extension == '.pdf':
            return self._load_pdf_from_bytes(file_bytes)
        else:
            # Handle text files
            try:
                return file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # Try other encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        return file_bytes.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                raise Exception("Could not decode text file. Please ensure it's in a supported encoding.")
    
    def _load_pdf_from_bytes(self, file_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        if not pdf_available:
            raise ImportError("PDF processing not available. Install PyPDF2 or pdfplumber: pip install PyPDF2 pdfplumber")
        
        text = ""
        
        if pdf_library == 'PyPDF2':
            try:
                pdf_file = io.BytesIO(file_bytes)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except Exception as e:
                raise Exception(f"Error reading PDF with PyPDF2: {e}")
                
        elif pdf_library == 'pdfplumber':
            try:
                pdf_file = io.BytesIO(file_bytes)
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                raise Exception(f"Error reading PDF with pdfplumber: {e}")
        
        if not text.strip():
            raise Exception("No text could be extracted from the PDF file")
            
        return text

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        return text

    def extract_sentences_about_topic(self, text: str, topic: str, context_size: int = 3) -> List[str]:
        """Extract sentences that mention the specified topic."""
        if not nltk_available:
            # Simple sentence splitting fallback
            sentences = re.split(r'[.!?]+', text)
        else:
            sentences = sent_tokenize(text)
        
        topic_sentences = []
        topic_lower = topic.lower()
        
        for i, sentence in enumerate(sentences):
            if topic_lower in sentence.lower():
                # Get context around the sentence
                start = max(0, i - context_size)
                end = min(len(sentences), i + context_size + 1)
                context = ' '.join(sentences[start:end])
                topic_sentences.append(context.strip())
        
        return topic_sentences

    def find_key_terms(self, text: str, topic: str) -> List[str]:
        """Extract key terms related to the topic from the text."""
        if not nltk_available:
            # Simple word extraction fallback
            words = re.findall(r'\b\w+\b', text.lower())
            return [word for word in words if len(word) > 3 and word != topic.lower()][:10]
        
        words = word_tokenize(text.lower())
        
        # Filter out stop words and short words
        key_terms = []
        for word in words:
            if (len(word) > 3 and 
                word not in self.stop_words and 
                word.isalpha() and 
                word != topic.lower()):
                if self.lemmatizer:
                    word = self.lemmatizer.lemmatize(word)
                key_terms.append(word)
        
        # Get most frequent terms
        from collections import Counter
        term_freq = Counter(key_terms)
        return [term for term, freq in term_freq.most_common(10)]

    def generate_question(self, topic: str, question_type: str = None, 
                         related_terms: List[str] = None) -> str:
        """Generate a question based on the topic and type."""
        if question_type is None:
            question_type = random.choice(list(self.question_templates.keys()))
        
        if question_type not in self.question_templates:
            question_type = 'definition'
        
        templates = self.question_templates[question_type]
        template = random.choice(templates)
        
        # Handle comparison questions
        if question_type == 'comparison' and related_terms:
            topic2 = random.choice(related_terms)
            question = template.format(topic1=topic, topic2=topic2)
        else:
            question = template.format(topic=topic)
        
        return question

    def format_latex_question(self, question: str, topic: str, 
                            difficulty: str = "Medium", 
                            points: int = 10,
                            include_space: bool = True) -> str:
        """Format the question in LaTeX."""
        latex_output = []
        
        # Question header
        latex_output.append("\\begin{question}")
        latex_output.append(f"\\textbf{{Topic:}} {topic}")
        latex_output.append(f"\\textbf{{Difficulty:}} {difficulty}")
        latex_output.append(f"\\textbf{{Points:}} {points}")
        latex_output.append("")
        
        # Question text
        latex_output.append("\\textit{Question:}")
        latex_output.append(question)
        latex_output.append("")
        
        # Answer space
        if include_space:
            latex_output.append("\\textit{Answer:}")
            latex_output.append("\\vspace{4cm}")
            latex_output.append("")
        
        latex_output.append("\\end{question}")
        latex_output.append("\\newpage")
        
        return "\n".join(latex_output)

    def create_latex_document(self, questions: List[str], title: str = "Generated Questions") -> str:
        """Create a complete LaTeX document with questions."""
        latex_doc = []
        
        # Document preamble
        latex_doc.append("\\documentclass[12pt]{article}")
        latex_doc.append("\\usepackage[utf8]{inputenc}")
        latex_doc.append("\\usepackage[margin=1in]{geometry}")
        latex_doc.append("\\usepackage{amsmath}")
        latex_doc.append("\\usepackage{amsfonts}")
        latex_doc.append("\\usepackage{amssymb}")
        latex_doc.append("")
        
        # Define question environment
        latex_doc.append("\\newcounter{questionnumber}")
        latex_doc.append("\\newenvironment{question}{%")
        latex_doc.append("  \\stepcounter{questionnumber}%")
        latex_doc.append("  \\section*{Question \\thequestionnumber}%")
        latex_doc.append("}{}")
        latex_doc.append("")
        
        # Document content
        latex_doc.append("\\begin{document}")
        latex_doc.append(f"\\title{{{title}}}")
        latex_doc.append("\\author{Generated by Question Generator}")
        latex_doc.append("\\date{\\today}")
        latex_doc.append("\\maketitle")
        latex_doc.append("")
        
        # Add questions
        for question in questions:
            latex_doc.append(question)
            latex_doc.append("")
        
        latex_doc.append("\\end{document}")
        
        return "\n".join(latex_doc)

    def generate_questions_from_text(self, text: str, topic: str, 
                                   num_questions: int = 5,
                                   question_types: List[str] = None) -> List[Dict]:
        """Generate multiple questions from text about a specific topic."""
        preprocessed_text = self.preprocess_text(text)
        topic_sentences = self.extract_sentences_about_topic(preprocessed_text, topic)
        
        if not topic_sentences:
            print(f"Warning: No content found related to '{topic}' in the text.")
            return []
        
        key_terms = self.find_key_terms(' '.join(topic_sentences), topic)
        
        questions = []
        available_types = question_types or list(self.question_templates.keys())
        
        for i in range(num_questions):
            question_type = random.choice(available_types)
            question_text = self.generate_question(topic, question_type, key_terms)
            
            difficulty = random.choice(["Easy", "Medium", "Hard"])
            points = random.choice([5, 10, 15, 20])
            
            latex_question = self.format_latex_question(
                question_text, topic, difficulty, points
            )
            
            questions.append({
                'question': question_text,
                'type': question_type,
                'topic': topic,
                'difficulty': difficulty,
                'points': points,
                'latex': latex_question
            })
        
        return questions

def main():
    parser = argparse.ArgumentParser(description='Generate questions from book/lecture notes')
    parser.add_argument('input_file', help='Path to the text/PDF file containing notes')
    parser.add_argument('topic', help='Topic to generate questions about')
    parser.add_argument('-n', '--num-questions', type=int, default=5,
                       help='Number of questions to generate (default: 5)')
    parser.add_argument('-o', '--output', help='Output LaTeX file path')
    parser.add_argument('--question-types', nargs='+', 
                       choices=['definition', 'explanation', 'comparison', 
                               'application', 'analysis', 'cause_effect'],
                       help='Specific question types to generate')
    parser.add_argument('--title', default='Generated Questions',
                       help='Title for the LaTeX document')
    
    args = parser.parse_args()
    
    # Initialize the question generator
    generator = QuestionGenerator()
    
    try:
        # Load the text
        print(f"Loading text from: {args.input_file}")
        text = generator.load_text(args.input_file)
        
        # Generate questions
        print(f"Generating {args.num_questions} questions about '{args.topic}'...")
        questions = generator.generate_questions_from_text(
            text, args.topic, args.num_questions, args.question_types
        )
        
        if not questions:
            print("No questions could be generated. Check if the topic exists in your text.")
            return
        
        # Create LaTeX document
        latex_questions = [q['latex'] for q in questions]
        latex_doc = generator.create_latex_document(latex_questions, args.title)
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(latex_doc)
            print(f"LaTeX document saved to: {args.output}")
        else:
            print("\n" + "="*50)
            print("GENERATED LATEX DOCUMENT:")
            print("="*50)
            print(latex_doc)
        
        # Display summary
        print(f"\nSummary:")
        print(f"- Generated {len(questions)} questions")
        print(f"- Topic: {args.topic}")
        print(f"- Question types: {set(q['type'] for q in questions)}")
        
    except Exception as e:
        print(f"Error: {e}")

# Example usage function
def example_usage():
    """Example of how to use the QuestionGenerator class programmatically."""
    generator = QuestionGenerator()
    
    # Sample text (you would normally load this from a file)
    sample_text = """
    Machine learning is a subset of artificial intelligence that enables computers 
    to learn and make decisions without being explicitly programmed. It involves 
    algorithms that can identify patterns in data and make predictions or classifications 
    based on those patterns. There are three main types of machine learning: 
    supervised learning, unsupervised learning, and reinforcement learning.
    
    Supervised learning uses labeled training data to learn a mapping function 
    from input variables to output variables. Common examples include regression 
    and classification problems. Unsupervised learning finds hidden patterns 
    in data without labeled examples. Reinforcement learning involves an agent 
    learning to make decisions by interacting with an environment and receiving 
    rewards or penalties.
    """
    
    # Generate questions
    questions = generator.generate_questions_from_text(
        sample_text, "machine learning", num_questions=3
    )
    
    # Create LaTeX document
    latex_questions = [q['latex'] for q in questions]
    latex_doc = generator.create_latex_document(latex_questions, "Machine Learning Quiz")
    
    print(latex_doc)

# Streamlit Web Interface
def streamlit_app():
    """Streamlit web interface for the Question Generator."""
    import streamlit as st
    import io
    
    st.set_page_config(
        page_title="Question Generator",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Question Generator from Notes")
    st.markdown("Generate LaTeX-formatted questions from your books or lecture notes!")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Initialize session state
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'latex_doc' not in st.session_state:
        st.session_state.latex_doc = ""
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your notes, book text, or PDF",
        type=['txt', 'md', 'pdf'],
        help="Upload a text file or PDF containing your notes or book content"
    )
    
    # Text input area as alternative
    st.markdown("**Or paste your text directly:**")
    direct_text = st.text_area(
        "Paste your text here",
        height=200,
        placeholder="Paste your lecture notes or book content here..."
    )
    
    # Topic input
    topic = st.text_input(
        "Topic to generate questions about",
        placeholder="e.g., machine learning, photosynthesis, calculus"
    )
    
    # Configuration options in sidebar
    num_questions = st.sidebar.slider(
        "Number of questions",
        min_value=1,
        max_value=20,
        value=5
    )
    
    question_types = st.sidebar.multiselect(
        "Question types",
        options=['definition', 'explanation', 'comparison', 'application', 'analysis', 'cause_effect'],
        default=['definition', 'explanation', 'application'],
        help="Select the types of questions you want to generate"
    )
    
    document_title = st.sidebar.text_input(
        "Document title",
        value="Generated Questions"
    )
    
    include_answer_space = st.sidebar.checkbox(
        "Include answer spaces",
        value=True,
        help="Add space for writing answers in the LaTeX output"
    )
    
    # Initialize the question generator
    generator = QuestionGenerator()
    
    # Generate questions button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_btn = st.button(
            "🚀 Generate Questions",
            type="primary",
            use_container_width=True
        )
    
    # Process text and generate questions
    if generate_btn:
        if not topic:
            st.error("Please enter a topic to generate questions about!")
            return
        
        # Get text content
        text_content = ""
        if uploaded_file is not None:
            try:
                with st.spinner("Processing uploaded file..."):
                    file_bytes = uploaded_file.read()
                    text_content = generator.load_text_from_bytes(file_bytes, uploaded_file.name)
                    
                    # Show file info
                    file_size = len(file_bytes) / 1024  # KB
                    st.success(f"✅ Successfully loaded {uploaded_file.name} ({file_size:.1f} KB)")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                if "PDF processing not available" in str(e):
                    st.info("💡 To process PDF files, install: `pip install PyPDF2 pdfplumber`")
                return
                
        elif direct_text.strip():
            text_content = direct_text
        else:
            st.error("Please upload a file or paste text content!")
            return
        
        # Generate questions
        with st.spinner("Analyzing text and generating questions..."):
            
        # Generate questions
        with st.spinner("Analyzing text and generating questions..."):
            try:
                questions = generator.generate_questions_from_text(
                    text_content,
                    topic,
                    num_questions,
                    question_types if question_types else None
                )
                
                if not questions:
                    st.error(f"No content found related to '{topic}' in the provided text. Please check your topic or text content.")
                    return
                
                # Store in session state
                st.session_state.questions = questions
                
                # Generate LaTeX document
                latex_questions = []
                for q in questions:
                    latex_q = generator.format_latex_question(
                        q['question'], 
                        q['topic'], 
                        q['difficulty'], 
                        q['points'],
                        include_answer_space
                    )
                    latex_questions.append(latex_q)
                
                st.session_state.latex_doc = generator.create_latex_document(
                    latex_questions, 
                    document_title
                )
                
                st.success(f"✅ Generated {len(questions)} questions successfully!")
                
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")
                return
    
    # Display results if questions exist
    if st.session_state.questions:
        st.markdown("---")
        st.header("📋 Generated Questions")
        
        # Display questions in tabs
        tab1, tab2, tab3 = st.tabs(["📝 Questions Preview", "📄 LaTeX Code", "📊 Summary"])
        
        with tab1:
            for i, question in enumerate(st.session_state.questions, 1):
                with st.expander(f"Question {i} - {question['type'].title()} ({question['difficulty']})"):
                    st.markdown(f"**Topic:** {question['topic']}")
                    st.markdown(f"**Type:** {question['type'].replace('_', ' ').title()}")
                    st.markdown(f"**Difficulty:** {question['difficulty']}")
                    st.markdown(f"**Points:** {question['points']}")
                    st.markdown("**Question:**")
                    st.info(question['question'])
        
        with tab2:
            st.markdown("**Complete LaTeX Document:**")
            st.code(st.session_state.latex_doc, language='latex')
            
            # Download button for LaTeX
            st.download_button(
                label="📥 Download LaTeX File",
                data=st.session_state.latex_doc,
                file_name=f"{document_title.lower().replace(' ', '_')}.tex",
                mime="text/plain"
            )
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Questions", len(st.session_state.questions))
                st.metric("Topic", topic)
                
                # Question types distribution
                type_counts = {}
                for q in st.session_state.questions:
                    q_type = q['type'].replace('_', ' ').title()
                    type_counts[q_type] = type_counts.get(q_type, 0) + 1
                
                st.markdown("**Question Types:**")
                for q_type, count in type_counts.items():
                    st.write(f"• {q_type}: {count}")
            
            with col2:
                # Difficulty distribution
                diff_counts = {}
                for q in st.session_state.questions:
                    diff = q['difficulty']
                    diff_counts[diff] = diff_counts.get(diff, 0) + 1
                
                st.markdown("**Difficulty Distribution:**")
                for diff, count in diff_counts.items():
                    st.write(f"• {diff}: {count}")
                
                # Points distribution
                total_points = sum(q['points'] for q in st.session_state.questions)
                st.metric("Total Points", total_points)
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        ### How to Use the Question Generator:
        
        1. **Upload your content**: Upload a text file (.txt, .md) or PDF file, or paste your notes directly
        2. **Specify a topic**: Enter the specific topic you want questions about
        3. **Configure options**: Use the sidebar to set number of questions, types, etc.
        4. **Generate**: Click the generate button and wait for processing
        5. **Download**: Use the LaTeX code to create a PDF document
        
        ### Supported File Formats:
        - **Text Files**: .txt, .md (plain text, markdown)
        - **PDF Files**: .pdf (text will be extracted automatically)
        - **Direct Input**: Copy and paste text directly into the text area
        
        ### Question Types:
        - **Definition**: Basic concept definitions
        - **Explanation**: How something works or functions
        - **Comparison**: Comparing different concepts
        - **Application**: Real-world applications and examples
        - **Analysis**: Critical thinking and evaluation
        - **Cause-Effect**: Relationships and consequences
        
        ### Tips:
        - Make sure your topic appears in the text content
        - Longer, more detailed text produces better questions
        - Use specific topics rather than very broad ones
        - The tool works best with educational content
        - **PDF Support**: Install `PyPDF2` or `pdfplumber` for PDF processing
        - **Large PDFs**: May take longer to process, be patient during upload
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit 🎈 | Question Generator v1.0"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # Check if running with streamlit
    try:
        import streamlit as st
        # If streamlit is available and we're in streamlit context
        if hasattr(st, 'runtime') and st.runtime.exists():
            streamlit_app()
        else:
            # Running as regular script
            if not nltk_available:
                print("Warning: NLTK not available. Using basic text processing.")
                print("For better results, install NLTK: pip install nltk")
                print()
            if not pdf_available:
                print("Warning: PDF processing not available.")
                print("For PDF support, install: pip install PyPDF2 pdfplumber")
                print()
            main()
    except ImportError:
        # Streamlit not available, run as regular script
        print("Streamlit not available. Running in command-line mode.")
        print("To use the web interface, install streamlit: pip install streamlit")
        if not nltk_available:
            print("Warning: NLTK not available. Using basic text processing.")
            print("For better results, install NLTK: pip install nltk")
            print()
        if not pdf_available:
            print("Warning: PDF processing not available.")
            print("For PDF support, install: pip install PyPDF2 pdfplumber")
            print()
        main()
