### **Python Expense Categorizer**



An automatic, machine learning powered tool that is helpful in the categorization of daily transactions into 7 categories: Bills, Food, Essentials, Academics, Entertainment and Travel. This project is an application of NLP to solve the problem of real world data organisation challenges. It aims to make the manual bookkeeping redundant by using a calibrated SVM, to predict the categories of transaction based on a text description. A fallback logic has been used as well to avoid wrong confident answers. A confidence meter is also built in, to make the user know how sure the AI is of the category so decided.



##### Technical Architecture:

1. Core Algorithm: Linear Support Vector Classifier
2. Feature Extraction: Term Frequency-Inverse Document Frequency
3. Scaling: Sublinear Term Frequency to account for keyword frequency variance.
4. Reliability/Fallback: A threshold of 25% confidence is applied to flag ambiguous or new words as miscellaneous rather than being put in a random category, hence misclassified.
5. Employing joblib: The model can be instantly deployed without retraining using serialisation.



##### Execution instructions:

This project is fully compatible with both the terminal or Command Line Interface (CLI) environments as well as Integrated Development Environments (IDEs). To run the project, make sure that python 3 and libraries mentioned in requirements are installed on your computer.



Option 1: Command Line Interface - Obtain the source code(.py) and training data(.csv), or clone the repository. Ensure that the source code and training data files are in the same folder. In the terminal environment, navigate to the project folder and execute "python python-expense-categorizer.py"



Option 2: Integrated Development Environments (like VS Code or PyCharm) - After opening the script and ensuring that the program and training data are in the same folder, select the python interpreter on your IDE (if applicable). Then run the script. The project will start working in a console or integrated terminal.



##### License:

The project is submitted to VITyarthi for academic evaluation purposes for the course Fundamentals in AI and ML (CSA2001) for winter semester 2025-26.



##### Contact Information:

Developer Name: Maanas Srivastava (25BCE10735), Student, VIT Bhopal

Email: maanas.25bce10735@vitbhopal.ac.in



##### Additional Notes:

The current model is trained on a localised data set of about 120 entries. The accuracy of the model can be increased by either expanding the training data or by using a separate dataset from the internet. Also, the miscellaneous category is a deliberate choice to maintain the project integrity when working with words that are unknown. It avoids misclassification.

