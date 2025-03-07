# SAS to PySpark Conversion using CrewAI

## 📌 Project Overview
This project provides an **automated pipeline** for converting **SAS procedures** to **PySpark** using CrewAI agents. It handles **dataset creation, statistical procedures, visualization**, and **large-scale data processing** efficiently.

## 🚀 Features
- **Automated SAS to PySpark Conversion** 🛠️
- **Handles SAS Procedures (e.g., PROC UNIVARIATE, PROC MEANS)** 📊
- **Supports Data Visualization with Matplotlib & Seaborn** 📈
- **Preserves Statistical Transformations & Data Formats** 🔢
- **Optimized for Large SAS Files Using Chunking** 🏗️
- **Prevents Duplicate Imports & Multiple SparkSessions** ⚡

## 🏗️ Project Structure
```
SAS_TO_PYSPARK_CREWAI/
│-- sas_to_pyspark.py          # Main script for SAS to PySpark conversion
│-- SAS_Procdure_1.sas        # Example SAS file
│-- README.md                 # Project documentation
│-- .gitignore                # Git ignore file
```

## 📜 How It Works
1️⃣ **Splits SAS Code into Logical Chunks** using `RecursiveCharacterTextSplitter` 📝  
2️⃣ **Extracts Metadata & Identifies Procedures** (e.g., PROC UNIVARIATE, PROC MEANS) 🔍  
3️⃣ **Converts to PySpark** using CrewAI agents 🤖  
4️⃣ **Handles Statistical & Visualization Libraries** (SciPy, Statsmodels, Matplotlib) 📊  
5️⃣ **Ensures Optimized Execution & Correct Data Formatting** ✅  

## ⚙️ Setup Instructions
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/ganesh2312/sas_to_pyspark.git
cd sas_to_pyspark
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Run the Conversion Script**
```sh
python sas_to_pyspark.py
```

## 🛠 Technologies Used
- **Python** 🐍
- **PySpark** 🔥
- **CrewAI** 🤖
- **LangChain** 🏗️
- **SciPy, Statsmodels, Matplotlib, Seaborn** 📊
- **Git & GitHub** 🛠️

## ⚡ Example Usage
```sh
python sas_to_pyspark.py --file example.sas
```

## 🛠 Issues & Troubleshooting
- **Permission Errors?** Run with `sudo` or check GitHub access rights 🔐
- **Large Files Skipping Lines?** Increase chunk overlap (`chunk_overlap=3000`) 📄
- **Multiple SparkSessions?** Ensure single SparkSession is used across chunks ⚡

## 📬 Contact
For queries, reach out at **ganeshnjr2312@gmail.com** 📩

---
🚀 **Transform your SAS workflows to scalable PySpark solutions!** 🔥

