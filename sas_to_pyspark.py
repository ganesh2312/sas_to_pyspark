import os
import re
import logging
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from openai import OpenAI
 
# Set up logging
logging.basicConfig(level=logging.INFO)
# Set API key
os.environ["OPENAI_API_KEY"] = " your api key here "
 
MODEL = "gpt-4-turbo"
def conversion_agent(file_path, prompt):
    llm = ChatOpenAI(model=MODEL)
 
    # Initialize Metadata Extraction Agent
    metadata_extractor_agent = Agent(
        role="Metadata Specialist",
        goal="Extract structured technical specifications from SAS procedures",
        backstory=(
            "Expert in SAS procedure analysis with deep understanding of statistical methods. "
            "Specializes in identifying key parameters, output requirements, and statistical measures."
        ),
        llm=llm,
        max_iter=3,
        verbose=True
    )
 
    # SAS Procedure Analyzer Agent
    sas_analyzer = Agent(
        role="Procedure Decomposer",
        goal="Identify core components of SAS procedures",
        backstory=(
            "Expert in SAS syntax analysis focusing on parameter extraction and output specification. "
            "Identifies procedure type, input parameters, statistical operations, and expected outputs."
        ),
        llm=llm,
        max_iter=5,
        verbose=True
    )
 
    # Enhanced PySpark Converter Agent
    pyspark_converter = Agent(
        role="Statistical Code Translator",
        goal="Generate PySpark code using appropriate statistical libraries",
        backstory=(
            "Expert in PySpark and scientific Python stack. Specializes in translating statistical procedures "
            "using optimal combinations of PySpark DataFrame operations, SciPy, statsmodels, and visualization libraries."
        ),
        llm=llm,
        max_iter=7,
        verbose=True
    )
 
    # Redefined Tasks
    # Task 1: Technical Metadata Extraction
    metadata_task = Task(
        description=(
            "Extract structured metadata from SAS code with focus on statistical requirements:\n"
            "1. Identify procedure type and parameters\n"
            "2. List required statistical measures\n"
            "3. Note any visualization requirements\n"
            "SAS Code: {sas_chunk}"
        ),
        expected_output=(
            "string containing: procedure_name (UPPERCASE), input_params, statistical_measures, "
            "visualization_type, required_libraries"
        ),
        agent=metadata_extractor_agent
    )
 
    # Task 2: Procedure Analysis
    analyze_task = Task(
        description=(
            "Analyze SAS procedure to identify:\n"
            "1. Primary statistical operation type\n"
            "2. Input parameters and data transformations\n"
            "3. Expected output format and precision requirements\n"
            "SAS Code: {sas_chunk}"
        ),
        expected_output=(
            "procedure_analysis: Structured analysis of SAS procedure components "
            "including data inputs, statistical operations, and output specs"
        ),
        agent=sas_analyzer
    )
 
    # Task 3: Library-aware Code Conversion
    convert_task = Task(
        description=(
            "Generate PySpark code using metadata and procedure analysis:\n"
            "1. Select appropriate libraries based on metadata\n"
            "2. Implement statistical measures using optimal combinations of:\n"
            "   - PySpark DataFrame operations\n"
            "   - SciPy/statsmodels for advanced statistics\n"
            "   - Seaborn/Matplotlib for visualizations\n"
            "3. Include necessary data type conversions\n"
            "4. Add performance optimizations for large datasets\n"
            " SAS Code: {sas_chunk}"
        ),
        expected_output=(
            "Complete PySpark implementation with:\n"
            "- Proper library imports\n"
            "- Statistical calculations matching SAS output\n"
            "- Visualizations when required\n"
            "- Memory optimization techniques"
        ),
        agent=pyspark_converter
    )
 
    # Adjust text splitter to preserve complete procedures
    separators = ["\nproc ", "\ndata ", "\n%macro "]
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=15000,
        chunk_overlap=1000
    )
 
    with open(file_path, "r") as f:
        sas_code = f.read()
 
    chunks = splitter.create_documents([sas_code])
    final_code = ""
 
    # Process each SAS code chunk sequentially
    for chunk in chunks:
        crew = Crew(
            agents=[metadata_extractor_agent, sas_analyzer, pyspark_converter],
            tasks=[metadata_task, analyze_task, convert_task],
            process=Process.sequential,
            verbose=True,
            max_rpm=3,
            cache=False
        )
 
        # Pass the SAS code chunk using the placeholder 'sas_chunk'
        result = crew.kickoff(inputs={'sas_chunk': chunk.page_content})
 
        try:
            converted_code = result.output
        except AttributeError:
            converted_code = str(result)
 
        # Clean formatting markers from the output
        converted_code = (
            converted_code.replace("```python", "")
                          .replace("```pyspark", "")
                          .replace("```", "")
                          .strip()
        )
 
        # Append the converted code; deduplicate overlapping import headers if necessary
        final_code += "\n" + converted_code
        """if final_code:
            header, *rest = converted_code.split('\n')
            if 'import' in header and not final_code.startswith(header):
                final_code = header + '\n' + final_code
            else:
                final_code += '\n' + converted_code
        else:
            final_code = converted_code"""
    print(chunks)
    return final_code
def main():
    # Accept SAS file path as input from the user
    file_path = input("Enter the path to your SAS code text file: ")
    conversion_prompt = """
    f"Convert the provided SAS code to PySpark, ensuring the following:\n"
        f"1. Accurate translation of statistical procedures (e.g., `proc univariate`) using PySpark DataFrame operations or appropriate libraries.\n"
        f"2. Conversion of SAS-generated visualizations (e.g., pie charts from `proc gchart`) to equivalent code using libraries like matplotlib or seaborn.\n"
        f"3. Proper handling of data types, column names, and date formats.\n"
        f"4. Utilization of a single SparkSession instance across the entire code, unless a compelling reason dictates otherwise.\n"
        f"5. Consideration of sampling or caching when converting data to Pandas with `.toPandas()` to avoid memory issues with large datasets.\n"
        f"6. To ensure accurate PySpark code generation for prebuilt SAS procedures like PROC MEANS and PROC UNIVARIATE, proper handling is required.\n"
        f"7. For predefined SAS libraries (e.g., SASHELP, SASUSER), ensure their contents are accurately referenced and handled in the PySpark environment.\n"
        f"8. **Do not include import statements or Spark session initialization in the generated code.**\n"
    Convert SAS code to PySpark code, ensuring that statistical procedures and charting are accurately converted.
    Specifically:
    - For `proc univariate`, include conversion for statistical measures using PySpark DataFrame operations or appropriate libraries.
    - For pie charts generated via SAS procedures (e.g., `proc gchart`), generate equivalent visualization code using matplotlib or a similar library.
    - Use additional packages such as SciPy, seaborn, matplotlib etc. whenever needed.
    - Ensure that data types, column names, and date formats are handled correctly.
    - Use a single SparkSession instance across the entire code, unless there is a compelling reason otherwise.
    - When converting data to Pandas with .toPandas(), consider sampling or caching if the dataset is large to avoid memory issues.
    """
 
    try:
        pyspark_code = conversion_agent(file_path, conversion_prompt)
        print("Converted PySpark Code:\n")
        print(pyspark_code)
 
        # Optionally, write the output to a file
        output_file = "converted_code.py"
        with open(output_file, "w") as f:
            f.write(pyspark_code)
        print(f"\nConverted code has been written to {output_file}")
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
 
if __name__ == "__main__":
 
    main()
 
 