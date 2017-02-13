#  
#Name : Aditya Ramachandra Desai
#USC ID : 5246-4961-15
#Script : The following script activates the standard perceptron to classify the emails based on the learning model
#Dataset 100% : /home/aditya/CSCI544Corpus/dev
#
from GenericPerceptron import GenericPerceptron
import sys

if __name__ == "__main__":
    classifyPerceptron = GenericPerceptron()
    classifyPerceptron.readModel()
    classifyPerceptron.labeldFiles = list(classifyPerceptron.labelData(sys.argv[1]))
    classifyPerceptron.classifyData()
    classifyPerceptron.writeResult(sys.argv[2])#"per_output.txt"
    classifyPerceptron.evaluate(sys.argv[2])

