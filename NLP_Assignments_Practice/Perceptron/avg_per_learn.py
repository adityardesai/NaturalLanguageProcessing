#  
#Name : Aditya Ramachandra Desai
#USC ID : 5246-4961-15
#Script : The following script activates the average perceptron to learn and build learning model
#Dataset 100% : /home/aditya/CSCI544Corpus/train
#Dataset 10% : /home/aditya/CSCI544Corpus/tenPerc
#
from GenericPerceptron import GenericPerceptron
import sys
from datetime import datetime

if __name__ == "__main__":
    avgPerceptron = GenericPerceptron()
    #startTime = datetime.now()
    avgPerceptron.labeldFiles = list(avgPerceptron.labelData(sys.argv[1]))
    #print(" Learning Label Fields ")
    #print(datetime.now() - startTime)
    #startTime2 = datetime.now()
    avgPerceptron.averageProcessLabels()
    #print(" Avg Processing Label Fields ")
    #print(datetime.now() - startTime2)
    avgPerceptron.updateAvgModel()
    avgPerceptron.writeModel()
   
