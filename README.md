# 📝 Question Generator from Notes

A powerful Python application that automatically generates LaTeX-formatted questions from your books, lecture notes, or any text content. Perfect for educators, students, and content creators who need to quickly create quizzes, exams, or study materials.

## ✨ Features

- 🎯 **Smart Topic Detection**: Automatically finds content related to your specified topic
- 📚 **Multiple Question Types**: Generate various types of questions including definitions, explanations, comparisons, applications, analysis, and cause-effect
- 🖥️ **Dual Interface**: Web-based Streamlit interface + Command-line tool
- 📄 **LaTeX Output**: Professional, ready-to-compile LaTeX documents
- 🎨 **Customizable**: Configure difficulty levels, question types, and document formatting
- 📊 **Analytics**: View question distribution and statistics
- 💾 **Easy Export**: Download LaTeX files directly from the web interface

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/question-generator.git
cd question-generator
```

2. **Install dependencies:**
```bash
pip install streamlit nltk
```

3. **Download NLTK data (recommended for better performance):**
```bash
python -m nltk.downloader punkt stopwords wordnet
```

### Running the Application

#### Web Interface (Recommended)
```bash
streamlit run question_generator.py
```
Then open your browser to `http://localhost:8501`

#### Command Line Interface
```bash
python question_generator.py notes.txt "machine learning" -n 5 -o quiz.tex
```

## 🖥️ Web Interface Guide

### 1. Upload Content
- **File Upload**: Upload `.txt` or `.md` files containing your notes
- **Direct Input**: Paste text directly into the text area

### 2. Configure Settings
- **Topic**: Specify what you want questions about
- **Number of Questions**: Choose 1-20 questions
- **Question Types**: Select from 6 different question categories
- **Document Title**: Customize your LaTeX document title
- **Answer Spaces**: Toggle answer spaces in the output

### 3. Generate & Download
- Click "Generate Questions" to process your content
- Preview questions in the "Questions Preview" tab
- View LaTeX code in the "LaTeX Code" tab
- Check statistics in the "Summary" tab
- Download the LaTeX file directly

## 💻 Command Line Usage

### Basic Usage
```bash
python question_generator.py input_file.txt "topic_name"
```

### Advanced Options
```bash
python question_generator.py notes.txt "quantum mechanics" \
    --num-questions 10 \
    --question-types definition explanation analysis \
    --output physics_quiz.tex \
    --title "Physics Midterm Quiz"
```

### Command Line Arguments
- `input_file`: Path to your text file
- `topic`: Topic to generate questions about
- `-n, --num-questions`: Number of questions (default: 5)
- `-o, --output`: Output LaTeX file path
- `--question-types`: Specific question types to generate
- `--title`: LaTeX document title

## 📋 Question Types

| Type | Description | Example |
|------|-------------|---------|
| **Definition** | Basic concept definitions | "What is machine learning?" |
| **Explanation** | How something works | "Explain how neural networks function." |
| **Comparison** | Comparing concepts | "Compare supervised and unsupervised learning." |
| **Application** | Real-world usage | "Provide examples of AI in healthcare." |
| **Analysis** | Critical evaluation | "Analyze the advantages of deep learning." |
| **Cause-Effect** | Relationships | "What causes overfitting in models?" |

## 📁 Project Structure

```
question-generator/
├── question_generator.py      # Main application file
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── examples/                 # Example input files
│   ├── ml_notes.txt         # Sample machine learning notes
│   └── physics_notes.txt    # Sample physics notes
└── output/                  # Generated LaTeX files
    └── sample_quiz.tex      # Example output
```

## 🔧 Requirements

### Required
- Python 3.7+
- streamlit
- Basic text processing capabilities

### Optional (Recommended)
- nltk - Enhanced text processing and better question generation
- LaTeX distribution (for compiling generated files to PDF)

### Full Requirements
```
streamlit>=1.28.0
nltk>=3.8
```

## 📖 Usage Examples

### Example 1: Machine Learning Quiz
```python
from question_generator import QuestionGenerator

generator = QuestionGenerator()
text = generator.load_text("ml_textbook.txt")
questions = generator.generate_questions_from_text(
    text, "neural networks", num_questions=5
)
```

### Example 2: History Exam
```bash
python question_generator.py history_notes.txt "World War II" \
    --num-questions 8 \
    --question-types definition explanation analysis cause_effect \
    --output history_exam.tex
```

### Example 3: Biology Study Guide
Using the web interface:
1. Upload your biology textbook chapter
2. Set topic to "photosynthesis"
3. Select question types: definition, explanation, application
4. Generate 6 questions
5. Download LaTeX file and compile to PDF

## 🎯 Tips for Best Results

### Input Text Quality
- **Detailed Content**: More comprehensive text produces better questions
- **Topic Coverage**: Ensure your topic is well-covered in the source material
- **Clear Structure**: Well-organized notes work better than fragmented text

### Topic Selection
- **Specific Topics**: "machine learning algorithms" works better than just "AI"
- **Consistent Terminology**: Use terms that appear in your source text
- **Multiple Contexts**: Topics mentioned in various contexts generate diverse questions

### Question Generation
- **Mixed Types**: Combine different question types for comprehensive coverage
- **Appropriate Difficulty**: Match difficulty to your audience level
- **Review Generated Content**: Always review questions before using in assessments

## 🔨 LaTeX Compilation

To convert generated LaTeX to PDF:

```bash
# Using pdflatex
pdflatex your_quiz.tex

# Using overleaf (recommended for beginners)
# Upload the .tex file to overleaf.com and compile online
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution
- Additional question types
- Support for more input formats (PDF, DOCX)
- Advanced NLP features
- UI/UX improvements
- Better LaTeX templates
- Question difficulty assessment

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Bug Reports & Feature Requests

- **Bug Reports**: Open an issue with detailed steps to reproduce
- **Feature Requests**: Open an issue with your proposed enhancement
- **Questions**: Check existing issues or open a new discussion

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Uses [NLTK](https://www.nltk.org/) for advanced text processing
- LaTeX formatting for professional document output
- Inspired by the need for automated educational content generation

## 📊 Changelog

### v1.0.0 (Current)
- ✅ Initial release
- ✅ Streamlit web interface
- ✅ Command-line interface
- ✅ Six question types
- ✅ LaTeX output format
- ✅ File upload and direct text input
- ✅ Configurable difficulty and points

### Planned Features
- 🔄 PDF and DOCX input support
- 🔄 Question difficulty auto-assessment
- 🔄 Multiple output formats (Word, HTML)
- 🔄 Advanced NLP with transformers
- 🔄 Question answer generation
- 🔄 Bulk processing capabilities

---

**Made with ❤️ for educators and learners everywhere**

*If you find this tool helpful, please ⭐ star this repository and share it with others!*
