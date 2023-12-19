import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import re
from constants import openai_key  # Assuming you have a constants file
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

class ChatbotApp:
    def __init__(self):
        self.pdf_file = None
        self.excel_file = None
        self.question_column = None
        self.raw_text = None

    def set_openai_api_key(self):
        os.environ["OPENAI_API_KEY"] = openai_key

    def process_pdf_file(self):
        if self.pdf_file is not None:
            pdf_reader = PdfReader(self.pdf_file)
            raw_text = ''
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    raw_text += content
            return raw_text
        return ''

    def extract_quantities(self, text):
        # Define the regex pattern to match numbers, optionally with commas, and decimals
        regex_pattern = r'\b\d{1,3}(?:,\d{3})*\.?\d*\b'
        matches = re.findall(regex_pattern, text, re.IGNORECASE)
        filtered_matches = [match[0] for match in matches if match[0] not in ['360','2021','2022','tonnes (tCO2-e)','tonnes (tCO 2-e)','t CO2-e','kgCO 2-e/sqm','tonnes (tCO 2-e)','year 1','Scope 2','Scope 1','Scope 3']]

        standardized_matches = [str(int(match.replace(',', ''))) for match in matches]
        return ",".join(standardized_matches)

    def extract_and_convert_to_float(self, s):
        pattern = r'\b\d{1,3}(?:,\d{3})*\.?\d*\b'
        match = re.search(pattern, s)
        if match:
            numeric_part = match.group()
            numeric_part = numeric_part.replace(',', '')
            try:
                return int(numeric_part)
            except ValueError:
                return "Error: Could not convert to float"
        return "No valid number found"
        example_string = "2,347t"
        extracted_float = extract_and_convert_to_float(example_string)
        extracted_float
    def extract_and_standardize_units(self, text):
        unit_patterns = {
            'million MWh': r'(?:32\.3\s+)?million megawatt hours',
            # ... [Other unit patterns] ...
        }

        unit = ''
        for standard_unit, unit_pattern in unit_patterns.items():
            if re.search(unit_pattern, text, re.IGNORECASE):
                unit = standard_unit
                break

        return unit

    def process_excel_file(self):
        if self.excel_file is not None:
            df = pd.read_excel(self.excel_file)
            questions = df[self.question_column].tolist()

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=800,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_text(self.raw_text)
            embeddings = OpenAIEmbeddings()
            document_search = FAISS.from_texts(texts, embeddings)
            chain = load_qa_chain(OpenAI(), chain_type="stuff")

            quantities = []
            units = []
            answers = []

            for question in questions:
                docs = document_search.similarity_search(question)
                answer = chain.run(input_documents=docs, question=question)
                quantity = self.extract_quantities(answer)
                unit = self.extract_and_standardize_units(answer)

                quantities.append(quantity)
                units.append(unit)
                answers.append(answer)

            df['Answers'] = answers
            df['Quantity'] = quantities
            df['Unit'] = units

            df.to_excel(self.excel_file, index=False)

    def run(self):
        st.title('Chatbot with File Processing')
        self.pdf_file = st.file_uploader('Upload PDF File', type='pdf')
        self.excel_file = st.file_uploader('Upload Excel File', type=['xlsx', 'xls'])
        self.question_column = st.text_input('Enter the column name for questions in the Excel file')

        if st.button('Process Files'):
            if self.pdf_file is not None and self.excel_file is not None:
                self.set_openai_api_key()
                self.raw_text = self.process_pdf_file()
                if self.raw_text:
                    self.process_excel_file()
                    st.download_button('Download Processed Excel', data=self.excel_file.getvalue(), file_name=self.excel_file.name)
                else:
                    st.warning('Please upload a PDF file.')
            else:
                st.warning('Please upload both a PDF and an Excel file.')

if __name__ == '__main__':
    app = ChatbotApp()
    app.run()
